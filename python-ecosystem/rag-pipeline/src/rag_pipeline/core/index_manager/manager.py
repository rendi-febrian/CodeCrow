"""
Main structural repository index manager.

Composes all index management components and provides the public API.
"""

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Callable, Optional, List

from qdrant_client import QdrantClient

from ...models.config import RAGConfig, IndexStats
from ..splitter import ASTCodeSplitter
from ..loader import DocumentLoader
from ..coordination import ProjectMutationCoordinator
from ..index_representation import (
    branch_splitter_kwargs,
    index_representation_fingerprint,
)
from ..pr_overlay_representation import (
    pr_overlay_representation_fingerprint,
)
from .. import revision_preflight
from ..exact_index import ExactIndexPreconditionError
from ..revision_preflight_cache import (
    RevisionPreflightCache,
    RevisionPreflightKey,
)
from ..source_tree import (
    attest_repository_source_tree,
    verify_repository_source_tree,
)

from .collection_manager import CollectionManager
from .branch_manager import BranchManager
from .point_operations import PointOperations
from .stats_manager import StatsManager
from .indexer import PrOverlayOperations, RepositoryIndexer

logger = logging.getLogger(__name__)


def read_repository_revision_preflight(*args, **kwargs):
    """Patchable module boundary around strict generation verification."""
    return revision_preflight.read_repository_revision_preflight(
        *args, **kwargs
    )


def read_repository_generation_manifest_receipt(*args, **kwargs):
    """Patchable boundary around bounded exact-target verification."""
    return revision_preflight.read_repository_generation_manifest_receipt(
        *args, **kwargs
    )


def _config_int(config, name: str, default: int) -> int:
    value = getattr(config, name, default)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return default
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _config_nonnegative_int(config, name: str, default: int) -> int:
    value = getattr(config, name, default)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _config_float(config, name: str, default: float) -> float:
    value = getattr(config, name, default)
    if not isinstance(value, (int, float, str)):
        return default
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


class RAGIndexManager:
    """Manage structural indices for code repositories using Qdrant.
    
    This is the main entry point for all indexing operations.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._full_index_capacity = threading.BoundedSemaphore(
            _config_int(config, "full_index_concurrency", 1)
        )
        self._mutation_coordinator = ProjectMutationCoordinator(
            os.getenv("REDIS_URL", "redis://redis:6379/1"),
            lease_seconds=_config_int(
                config,
                "rag_mutation_lease_seconds",
                300,
            ),
            acquire_timeout_seconds=_config_float(
                config,
                "rag_mutation_acquire_timeout_seconds",
                5.0,
            ),
        )
        self._revision_preflight_cache = RevisionPreflightCache(
            max_entries=_config_int(
                config,
                "revision_preflight_cache_entries",
                512,
            ),
            ttl_seconds=_config_nonnegative_int(
                config,
                "revision_preflight_cache_ttl_seconds",
                0,
            ),
            max_concurrent_loads=_config_int(
                config,
                "revision_preflight_max_concurrency",
                2,
            ),
        )
        self.index_representation_fingerprint = (
            index_representation_fingerprint(config)
        )
        self.pr_overlay_representation_fingerprint = (
            pr_overlay_representation_fingerprint(config)
        )

        plugin_catalog = None
        plugin_runtime = None
        plugin_selector = None
        try:
            from codecrow_plugins import PluginRuntime, ProjectSelector
            from codecrow_plugins.bootstrap import discover_builtin_plugins

            plugin_catalog = discover_builtin_plugins()
            plugin_runtime = PluginRuntime(plugin_catalog)
            plugin_selector = ProjectSelector(plugin_catalog.registry)
            logger.info("Loaded plugins: %s", ", ".join(plugin_catalog.registry.ordered_ids))
        except ModuleNotFoundError as exception:
            if exception.name != "codecrow_plugins":
                raise
            logger.warning("Plugin package is not installed; using the generic structural fallback")

        self.plugin_catalog = plugin_catalog
        self.plugin_runtime = plugin_runtime
        self.plugin_selector = plugin_selector

        # Qdrant client
        self.qdrant_client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key or None,
            timeout=_config_int(config, "qdrant_timeout_seconds", 30),
        )
        logger.info(f"Connected to Qdrant at {config.qdrant_url}")

        # Splitter and loader
        logger.info("Using ASTCodeSplitter for code chunking (tree-sitter query-based)")
        self.splitter = ASTCodeSplitter(
            **branch_splitter_kwargs(config),
            plugin_runtime=plugin_runtime,
        )
        self.loader = DocumentLoader(config)

        # Component managers
        self._collection_manager = CollectionManager(self.qdrant_client)
        self._branch_manager = BranchManager(self.qdrant_client)
        self._point_ops = PointOperations(
            self.qdrant_client,
            batch_size=_config_int(config, "qdrant_upsert_batch_size", 128),
            max_upsert_payload_bytes=_config_int(
                config,
                "qdrant_upsert_max_payload_bytes",
                8 * 1024 * 1024,
            ),
        )
        self._stats_manager = StatsManager(self.qdrant_client)
        
        # Higher-level operations
        self._indexer = RepositoryIndexer(
            config=config,
            collection_manager=self._collection_manager,
            branch_manager=self._branch_manager,
            point_ops=self._point_ops,
            stats_manager=self._stats_manager,
            splitter=self.splitter,
            loader=self.loader,
            plugin_catalog=plugin_catalog,
            plugin_runtime=plugin_runtime,
            plugin_selector=plugin_selector,
        )
        self._pr_overlay_ops = PrOverlayOperations(
            client=self.qdrant_client,
            point_ops=self._point_ops,
        )

    # Repository indexing

    @contextmanager
    def _admit_full_index(
        self,
        workspace: str,
        project: str,
        branch: str,
        progress_callback: Optional[Callable[[dict], None]],
    ):
        """Serialize memory-heavy full builds inside one RAG worker process."""
        wait_started = time.monotonic()
        acquired = self._full_index_capacity.acquire(blocking=False)
        if not acquired:
            logger.info(
                "RAG full index waiting for process capacity "
                "workspace=%s project=%s branch=%s",
                workspace,
                project,
                branch,
            )
            if progress_callback is not None:
                try:
                    progress_callback({
                        "stage": "waiting_capacity",
                        "message": "Waiting for RAG full-index capacity",
                        "progress": 0,
                    })
                except Exception as exception:
                    logger.warning(
                        "RAG capacity progress callback failed: %s",
                        exception,
                    )
            self._full_index_capacity.acquire()
        waited_ms = round((time.monotonic() - wait_started) * 1000)
        logger.info(
            "RAG full index admitted workspace=%s project=%s branch=%s "
            "wait_ms=%s",
            workspace,
            project,
            branch,
            waited_ms,
        )
        try:
            yield
        finally:
            self._full_index_capacity.release()

    def estimate_repository_size(
        self,
        repo_path: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> tuple[int, int]:
        """Estimate repository size (file count and chunk count)."""
        return self._indexer.estimate_repository_size(repo_path, include_patterns, exclude_patterns)

    def index_repository(
        self,
        repo_path: str,
        workspace: str,
        project: str,
        branch: str,
        commit: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        source_tree_sha256: Optional[str] = None,
        collection_target: str = "",
        progress_callback: Optional[Callable[[dict], None]] = None,
        project_type: Optional[str] = None,
        source_root: Optional[str] = None,
    ) -> IndexStats:
        """Index entire repository for a branch using atomic swap strategy."""
        if not collection_target:
            raise ValueError(
                "full repository indexing requires an immutable collection target"
            )
        with self._admit_full_index(
            workspace,
            project,
            branch,
            progress_callback,
        ):
            return self._index_repository_admitted(
                repo_path=repo_path,
                workspace=workspace,
                project=project,
                branch=branch,
                commit=commit,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                source_tree_sha256=source_tree_sha256,
                collection_target=collection_target,
                progress_callback=progress_callback,
                project_type=project_type,
                source_root=source_root,
            )

    def _index_repository_admitted(
        self,
        repo_path: str,
        workspace: str,
        project: str,
        branch: str,
        commit: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        source_tree_sha256: Optional[str] = None,
        collection_target: str = "",
        progress_callback: Optional[Callable[[dict], None]] = None,
        project_type: Optional[str] = None,
        source_root: Optional[str] = None,
    ) -> IndexStats:
        """Run a full build after process-wide capacity has been acquired."""
        if not collection_target:
            raise ValueError(
                "full repository indexing requires an exact target"
            )
        alias_name = collection_target
        source_tree = (
            verify_repository_source_tree(
                repo_path,
                commit,
                source_tree_sha256,
            )
            if source_tree_sha256
            else attest_repository_source_tree(repo_path, commit)
        )
        with self._mutation_coordinator.acquire(
            workspace,
            project,
            "full-index",
            collection_target=collection_target,
        ) as lease:
            return self._indexer.index_repository(
                repo_path=repo_path,
                workspace=workspace,
                project=project,
                branch=branch,
                commit=commit,
                alias_name=alias_name,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                source_tree_sha256=source_tree.tree_sha256,
                source_tree=source_tree,
                operation_id=lease.token,
                activation_guard=lease.assert_owned,
                progress_callback=progress_callback,
                project_type=project_type,
                source_root=source_root,
            )

    def get_revision_preflight(
        self,
        workspace: str,
        project: str,
        branch: str,
        commit: str,
        *,
        collection_target: str,
    ):
        """Verify one immutable revision in an explicitly selected target."""
        physical = self._collection_manager.resolve_collection_target(
            collection_target
        )
        if physical is None:
            return None
        if not self._collection_manager.is_structural_collection(physical):
            logger.info(
                "Ignoring pre-structural repository generation target %s; "
                "a full index will publish its replacement",
                physical,
            )
            return None
        cache = getattr(self, "_revision_preflight_cache", None)
        if cache is None:
            # Preserve lightweight construction used by tooling and tests.
            cache = RevisionPreflightCache(
                max_entries=_config_int(
                    self.config,
                    "revision_preflight_cache_entries",
                    512,
                ),
                ttl_seconds=_config_nonnegative_int(
                    self.config,
                    "revision_preflight_cache_ttl_seconds",
                    0,
                ),
                max_concurrent_loads=_config_int(
                    self.config,
                    "revision_preflight_max_concurrency",
                    2,
                ),
            )
            self._revision_preflight_cache = cache

        def verify():
            result = read_repository_revision_preflight(
                self.qdrant_client,
                physical,
                branch,
                commit,
            )
            if result is None:
                return None
            if (
                result.get("workspace") != workspace
                or result.get("project") != project
            ):
                raise ExactIndexPreconditionError(
                    "repository generation coordinates do not match the requested tenant"
                )
            return {
                **result,
                "current_index_representation_fingerprint": (
                    self.index_representation_fingerprint
                ),
            }

        return cache.get_or_load(
            RevisionPreflightKey(
                collection=physical,
                workspace=workspace,
                project=project,
                branch=branch,
                commit=commit,
            ),
            verify,
        )

    # Branch operations

    def delete_branch(
        self,
        workspace: str,
        project: str,
        branch: str,
        collection_target: str,
        generation_revision: str,
        generation_manifest_sha256: str,
    ) -> bool:
        """Delete one registry-selected sealed branch generation."""
        if not generation_revision or not generation_manifest_sha256:
            raise ExactIndexPreconditionError(
                "exact generation deletion requires its revision and manifest receipt"
            )
        with self._mutation_coordinator.acquire(
            workspace,
            project,
            "delete-generation",
            collection_target=collection_target,
        ) as lease:
            physical = self._collection_manager.resolve_collection_target(
                collection_target
            )
            if physical is None:
                return False
            receipt = read_repository_generation_manifest_receipt(
                self.qdrant_client,
                physical,
                workspace,
                project,
                branch,
                generation_revision,
                generation_manifest_sha256,
            )
            if receipt is None:
                raise ExactIndexPreconditionError(
                    "collection target does not match the registry generation receipt"
                )
            lease.assert_owned()
            alias_target = self._collection_manager.read_alias_targets(
                [collection_target]
            )[collection_target]
            if alias_target is not None:
                if alias_target != physical:
                    raise ExactIndexPreconditionError(
                        "collection target changed before exact generation deletion"
                    )
                if not self._collection_manager.delete_alias(collection_target):
                    return False
            return self._collection_manager.delete_collection(physical)

    def cleanup_expired_pending_collections(self) -> int:
        """Remove only old pending collections without a live operation lease."""
        return self._collection_manager.cleanup_expired_pending_collections(
            is_operation_active=self._mutation_coordinator.is_operation_active,
        )

    def pr_overlay_mutation(
        self,
        workspace: str,
        project: str,
        pr_number: int,
        operation: str,
    ):
        """Serialize one PR overlay without blocking unrelated PRs.

        Index replacement and deletion for the same PR must not overlap, even
        when the target generation changes between attempts. Different PR
        numbers have disjoint point identities and can mutate concurrently.
        """
        return self._mutation_coordinator.acquire(
            workspace,
            project,
            operation,
            publication_scope=f"pr-overlay:{pr_number}",
        )

    def close(self) -> None:
        try:
            self._mutation_coordinator.close()
        finally:
            self.qdrant_client.close()
