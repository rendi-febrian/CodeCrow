"""
Repository indexing operations.

Handles full repository indexing with atomic swap and streaming processing.
"""

import gc
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, List

from ..documents import TextNode
from qdrant_client.models import (
    PointIdsList,
    PointStruct,
)

from ...models.config import RAGConfig, IndexStats
from ...utils.utils import clean_archive_path, make_namespace
from ..index_representation import (
    INDEX_REPRESENTATION_PAYLOAD_KEY,
    index_representation_fingerprint,
)
from ..generation_manifest import (
    build_generation_manifest_node,
    compute_generation_members_digest,
    seal_generation_members,
)
from ..loader import (
    REPOSITORY_FILE_SIZE_LIMIT_CODE,
    RepositoryFileSkip,
)
from ..source_tree import require_repository_source_tree_unchanged
from .collection_manager import CollectionManager
from .branch_manager import BranchManager
from .point_operations import PointOperations, STORAGE_MARKER_VECTOR
from .stats_manager import StatsManager

logger = logging.getLogger(__name__)

# Memory-efficient batch sizes
DOCUMENT_BATCH_SIZE = 50
INSERT_BATCH_SIZE = 128
OPAQUE_STATE_PART_SIZE = 100_000
ARCHITECTURE_SOURCE_PART_SIZE = 50_000


def _bounded_source_parts(content: str) -> list[tuple[str, int, int]]:
    """Split source deterministically and retain the covered one-based lines."""
    parts = []
    offset = 0
    start_line = 1
    while offset < len(content):
        hard_end = min(offset + ARCHITECTURE_SOURCE_PART_SIZE, len(content))
        end = hard_end
        if hard_end < len(content):
            newline = content.rfind("\n", offset, hard_end)
            if newline >= offset:
                end = newline + 1
        if end <= offset:
            end = hard_end

        part = content[offset:end]
        newline_count = part.count("\n")
        end_line = start_line + newline_count
        if part.endswith("\n"):
            end_line -= 1
        parts.append((part, start_line, max(start_line, end_line)))
        start_line += newline_count
        offset = end
    return parts


def _plugin_identity_metadata(
    capabilities,
    implementation_fingerprint: str,
    representation_fingerprint: Optional[str] = None,
):
    metadata = {
        "plugin_ids": [],
        "plugin_fingerprint": "sha256:" + "0" * 64,
        "plugin_descriptor_fingerprint": "sha256:" + "0" * 64,
        "plugin_implementation_fingerprint": "sha256:" + "0" * 64,
        INDEX_REPRESENTATION_PAYLOAD_KEY: (
            representation_fingerprint
            or index_representation_fingerprint()
        ),
    }
    if capabilities is None:
        return metadata
    metadata.update({
        "plugin_ids": list(capabilities.repository_plugins),
        "plugin_fingerprint": capabilities.fingerprint,
        "plugin_descriptor_fingerprint": capabilities.descriptor_fingerprint,
        "plugin_implementation_fingerprint": implementation_fingerprint,
    })
    return metadata


class RepositoryIndexer:
    """Handles repository indexing operations."""
    
    def __init__(
        self,
        config: RAGConfig,
        collection_manager: CollectionManager,
        branch_manager: BranchManager,
        point_ops: PointOperations,
        stats_manager: StatsManager,
        splitter,
        loader,
        plugin_catalog=None,
        plugin_runtime=None,
        plugin_selector=None,
    ):
        self.config = config
        self.collection_manager = collection_manager
        self.branch_manager = branch_manager
        self.point_ops = point_ops
        self.stats_manager = stats_manager
        self.splitter = splitter
        self.loader = loader
        self.plugin_catalog = plugin_catalog
        self.plugin_runtime = plugin_runtime
        self.plugin_selector = plugin_selector
        self.representation_fingerprint = index_representation_fingerprint(
            config
        )

    @staticmethod
    def accept_recoverable_repository_diagnostics(
        diagnostics,
        phase: str,
    ) -> set[str]:
        """Log quarantined project inputs and reject only runtime-level faults."""
        recoverable = [
            diagnostic for diagnostic in diagnostics
            if diagnostic.recoverable
        ]
        fatal = [
            diagnostic for diagnostic in diagnostics
            if not diagnostic.recoverable
        ]
        for diagnostic in recoverable:
            logger.warning(
                "Skipping invalid repository input during %s "
                "(plugin=%s code=%s path=%s): %s",
                phase,
                diagnostic.plugin_id or "plugin",
                diagnostic.code,
                diagnostic.path or "<repository>",
                diagnostic.message,
            )
        if fatal:
            summary = "; ".join(
                f"{diagnostic.plugin_id or 'plugin'}:{diagnostic.code}: "
                f"{diagnostic.message}"
                for diagnostic in fatal[:10]
            )
            raise RuntimeError(f"{phase} failed: {summary}")
        return {
            diagnostic.path
            for diagnostic in recoverable
            if diagnostic.path is not None
        }

    @staticmethod
    def _architecture_nodes(
        analysis,
        capabilities,
        workspace: str,
        project: str,
        branch: str,
        commit: str,
        implementation_fingerprint: str = "sha256:" + "0" * 64,
        representation_fingerprint: Optional[str] = None,
    ) -> List[TextNode]:
        """Serialize neutral architecture packets into bounded retrieval nodes."""
        nodes: List[TextNode] = []
        facts_per_node = 25
        grouped = {}
        from ..repository_overlay import architecture_group_id

        for packet in analysis.packets:
            for fact in packet.facts:
                identity = (packet.plugin_id, packet.kind, fact.path)
                grouped.setdefault(identity, []).append((packet, fact))

        for (plugin_id, packet_kind, source_path), records in sorted(grouped.items()):
            group_id = architecture_group_id((plugin_id, packet_kind, source_path))
            records = sorted(records, key=lambda item: (item[0].key, item[1]))
            for offset in range(0, len(records), facts_per_node):
                segment = records[offset:offset + facts_per_node]
                identity = f"{plugin_id}\0{packet_kind}\0{source_path}\0{offset // facts_per_node}"
                digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                synthetic_path = f"__analysis_architecture__/{plugin_id}/{digest}.context"
                related_paths = sorted({
                    path
                    for _, fact in segment
                    for path in (fact.path, *fact.related_paths)
                })
                fact_payload = [
                    {
                        **dict(fact.as_metadata()),
                        "packetKey": packet.key,
                        "packetAttributes": dict(packet.attributes),
                    }
                    for packet, fact in segment
                ]
                identifiers = sorted({
                    value
                    for _, fact in segment
                    for value in (fact.source, fact.target)
                    if value
                })
                packet_keys = sorted({packet.key for packet, _ in segment})
                header = [
                    "Deterministic repository architecture context",
                    f"Plugin: {plugin_id}",
                    f"Kind: {packet_kind}",
                    f"Source: {source_path}",
                    "Packet keys: " + ", ".join(packet_keys),
                ]
                header.append("Related paths: " + ", ".join(related_paths))
                fact_lines = [
                    (
                        f"- Packet {packet.key}: [{fact.kind}] "
                        f"{fact.source} {fact.relation} {fact.target} "
                        f"({fact.path}:{fact.line})"
                        + (
                            " {" + ", ".join(
                                f"{key}={value}" for key, value in fact.attributes
                            ) + "}"
                            if fact.attributes else ""
                        )
                    )
                    for packet, fact in segment
                ]
                nodes.append(TextNode(
                    text="\n".join((*header, "Facts:", *fact_lines)),
                    metadata={
                        "workspace": workspace,
                        "project": project,
                        "branch": branch,
                        "commit": commit,
                        "path": synthetic_path,
                        "language": "architecture-context",
                        "filetype": "context",
                        "architecture_context": True,
                        "architecture_plugin": plugin_id,
                        "architecture_kind": packet_kind,
                        "architecture_source_path": source_path,
                        "architecture_group": group_id,
                        "architecture_key": f"{packet_kind}:{source_path}:{offset // facts_per_node}",
                        "architecture_keys": packet_keys,
                        "architecture_paths": related_paths,
                        "architecture_identifiers": identifiers,
                        **_plugin_identity_metadata(
                            capabilities,
                            implementation_fingerprint,
                            representation_fingerprint,
                        ),
                        "plugin_graph_facts": fact_payload,
                    },
                ))
        return nodes

    @staticmethod
    def _snapshot_nodes(
        analysis,
        capabilities,
        workspace: str,
        project: str,
        branch: str,
        commit: str,
        implementation_fingerprint: str = "sha256:" + "0" * 64,
        representation_fingerprint: Optional[str] = None,
    ) -> List[TextNode]:
        """Store opaque plugin snapshots in bounded payload records."""
        nodes: List[TextNode] = []
        part_size = OPAQUE_STATE_PART_SIZE
        for snapshot in analysis.snapshots:
            identity = f"{snapshot.plugin_id}\0{snapshot.kind}"
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            content_digest = hashlib.sha256(snapshot.content.encode("utf-8")).hexdigest()
            parts = [
                snapshot.content[offset:offset + part_size]
                for offset in range(0, len(snapshot.content), part_size)
            ]
            for part_index, content in enumerate(parts):
                nodes.append(TextNode(
                    text=content,
                    metadata={
                        "workspace": workspace,
                        "project": project,
                        "branch": branch,
                        "commit": commit,
                        "path": (
                            f"__analysis_state__/{snapshot.plugin_id}/{digest}/"
                            f"{part_index:06d}.state"
                        ),
                        "language": "repository-state",
                        "filetype": "state",
                        "repository_snapshot": True,
                        "snapshot_plugin": snapshot.plugin_id,
                        "snapshot_kind": snapshot.kind,
                        "snapshot_part": part_index,
                        "snapshot_parts": len(parts),
                        "snapshot_content_sha256": content_digest,
                        **_plugin_identity_metadata(
                            capabilities,
                            implementation_fingerprint,
                            representation_fingerprint,
                        ),
                    },
                ))
        return nodes

    @staticmethod
    def _symbol_nodes(
        analysis,
        capabilities,
        workspace: str,
        project: str,
        branch: str,
        commit: str,
        implementation_fingerprint: str = "sha256:" + "0" * 64,
        representation_fingerprint: Optional[str] = None,
    ) -> List[TextNode]:
        """Persist neutral plugin symbols as independently searchable records."""
        nodes: List[TextNode] = []
        for symbol in analysis.symbols:
            short_name = re.split(r"[\\.:]+", symbol.qualified_name)[-1]
            identity = (
                f"symbol\0{symbol.qualified_name}\0{symbol.kind}\0"
                f"{symbol.path}\0{symbol.line}"
            )
            nodes.append(TextNode(
                text=f"{symbol.kind} {symbol.qualified_name}",
                metadata={
                    "workspace": workspace,
                    "project": project,
                    "branch": branch,
                    "commit": commit,
                    "path": symbol.path,
                    "language": "structural-symbol",
                    "filetype": (
                        symbol.path.rsplit(".", 1)[-1]
                        if "." in symbol.path else ""
                    ),
                    "start_line": symbol.line,
                    "end_line": symbol.line,
                    "symbol_definition": True,
                    "symbol_qualified_name": symbol.qualified_name,
                    "symbol_kind": symbol.kind,
                    "symbol_parents": list(symbol.parents),
                    "symbol_methods": list(symbol.methods),
                    "symbol_constructor_types": list(symbol.constructor_types),
                    "symbol_attributes": dict(symbol.attributes),
                    "primary_name": short_name,
                    "symbol_names": [short_name, symbol.qualified_name],
                    "architecture_identifiers": [
                        short_name,
                        symbol.qualified_name,
                        *symbol.parents,
                    ],
                    "storage_identity": identity,
                    **_plugin_identity_metadata(
                        capabilities,
                        implementation_fingerprint,
                        representation_fingerprint,
                    ),
                },
            ))
        return nodes

    @staticmethod
    def _repository_facts_nodes(
        repository_facts,
        capabilities,
        workspace: str,
        project: str,
        branch: str,
        commit: str,
        implementation_fingerprint: str = "sha256:" + "0" * 64,
        representation_fingerprint: Optional[str] = None,
    ) -> List[TextNode]:
        """Persist the complete neutral inventory used for plugin selection."""
        content = json.dumps(
            {
                "revision": repository_facts.revision,
                "paths": list(repository_facts.paths),
                "markerContents": dict(repository_facts.marker_contents),
                "projectType": repository_facts.project_type,
                "sourceRoot": repository_facts.source_root,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        content_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        part_size = OPAQUE_STATE_PART_SIZE
        parts = [
            content[offset:offset + part_size]
            for offset in range(0, len(content), part_size)
        ]
        return [
            TextNode(
                text=part,
                metadata={
                    "workspace": workspace,
                    "project": project,
                    "branch": branch,
                    "commit": commit,
                    "path": (
                        "__analysis_state__/repository-facts/"
                        f"{part_index:06d}.state"
                    ),
                    "language": "repository-state",
                    "filetype": "state",
                    "repository_facts_state": True,
                    "facts_part": part_index,
                    "facts_parts": len(parts),
                    "facts_content_sha256": content_digest,
                    **_plugin_identity_metadata(
                        capabilities,
                        implementation_fingerprint,
                        representation_fingerprint,
                    ),
                },
            )
            for part_index, part in enumerate(parts)
        ]

    @staticmethod
    def _repository_context_nodes(
        analysis,
        capabilities,
        workspace: str,
        project: str,
        branch: str,
        commit: str,
        implementation_fingerprint: str = "sha256:" + "0" * 64,
        representation_fingerprint: Optional[str] = None,
        excluded_paths: Optional[set[str]] = None,
    ) -> List[TextNode]:
        """Serialize exact plugin-selected sources as structural records."""
        nodes: List[TextNode] = []
        excluded_paths = excluded_paths or set()
        for context in analysis.contexts:
            if context.path in excluded_paths:
                continue
            parts = _bounded_source_parts(context.content)
            for part_index, (content, start_line, end_line) in enumerate(parts):
                nodes.append(TextNode(
                    text=content,
                    metadata={
                        "workspace": workspace,
                        "project": project,
                        "branch": branch,
                        "commit": commit,
                        "path": context.path,
                        "language": "architecture-source",
                        "filetype": context.path.rsplit(".", 1)[-1],
                        "architecture_source": True,
                        "architecture_plugin": context.plugin_id,
                        "architecture_source_kind": context.kind,
                        "architecture_source_part": part_index,
                        "architecture_source_parts": len(parts),
                        "start_line": start_line,
                        "end_line": end_line,
                        **_plugin_identity_metadata(
                            capabilities,
                            implementation_fingerprint,
                            representation_fingerprint,
                        ),
                        **dict(context.attributes),
                    },
                ))
        return nodes

    @staticmethod
    def _architecture_only_source_nodes(
        documents,
        capabilities,
        workspace: str,
        project: str,
        branch: str,
        commit: str,
        implementation_fingerprint: str = "sha256:" + "0" * 64,
        representation_fingerprint: Optional[str] = None,
    ) -> List[TextNode]:
        """Store bounded raw text for files reserved for plugin analysis.

        Architecture-only disposition suppresses generic syntax splitting, not
        exact source availability. Architecture facts can name these files as
        relation targets, so their already-loaded text must remain hydratable.
        """
        nodes: List[TextNode] = []
        identity_metadata = _plugin_identity_metadata(
            capabilities,
            implementation_fingerprint,
            representation_fingerprint,
        )
        for document in sorted(
            documents,
            key=lambda item: str(item.metadata.get("path", "")),
        ):
            path = clean_archive_path(str(document.metadata.get("path", "")))
            content = document.text
            if not path or not isinstance(content, str) or not content.strip():
                continue
            parts = _bounded_source_parts(content)
            for part_index, (part, start_line, end_line) in enumerate(parts):
                nodes.append(TextNode(
                    text=part,
                    metadata={
                        **document.metadata,
                        "workspace": workspace,
                        "project": project,
                        "branch": branch,
                        "commit": commit,
                        "path": path,
                        "architecture_source": True,
                        "architecture_source_kind": (
                            "plugin-selected-raw-source"
                        ),
                        "architecture_source_part": part_index,
                        "architecture_source_parts": len(parts),
                        "start_line": start_line,
                        "end_line": end_line,
                        "storage_identity": f"architecture-source:{path}",
                        **identity_metadata,
                    },
                ))
        return nodes
    
    def estimate_repository_size(
        self,
        repo_path: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> tuple[int, int]:
        """Estimate repository size (file count and chunk count) without actually indexing."""
        logger.info(f"Estimating repository size for: {repo_path}")

        repo_path_obj = Path(repo_path)
        file_list = list(self.loader.iter_repository_files(repo_path_obj, include_patterns, exclude_patterns))
        if self.plugin_catalog is not None and self.plugin_selector is not None:
            from codecrow_plugins import FileDisposition, build_repository_facts

            estimate_capabilities = self.plugin_selector.select(build_repository_facts(
                repo_path_obj,
                "estimate",
                file_list,
                self.plugin_catalog.registry,
            ))
            if self.plugin_runtime is not None:
                file_list = [
                    path
                    for path in file_list
                    if self.plugin_runtime.file_disposition(
                        clean_archive_path(Path(path).as_posix()), estimate_capabilities
                    ) is FileDisposition.FULL
                ]
        file_count = len(file_list)
        logger.info(
            "RAG capacity scan found %s repository files "
            "(this is not the LLM review scope)",
            file_count,
        )

        if file_count == 0:
            return 0, 0

        SAMPLE_SIZE = 100
        chunk_count = 0
        
        if file_count <= SAMPLE_SIZE:
            for i in range(0, file_count, DOCUMENT_BATCH_SIZE):
                batch = file_list[i:i + DOCUMENT_BATCH_SIZE]
                documents = self.loader.load_file_batch(
                    batch, repo_path_obj, "estimate", "estimate", "estimate", "estimate"
                )
                if documents:
                    chunks = self.splitter.split_documents(documents)
                    chunk_count += len(chunks)
                    del chunks
                del documents
                gc.collect()
        else:
            # Stable spread across the normalized list. Sampling must not change
            # index admission decisions between identical runs.
            ordered_files = sorted(file_list, key=lambda path: str(path).replace("\\", "/"))
            sample_files = [
                ordered_files[(index * len(ordered_files)) // SAMPLE_SIZE]
                for index in range(SAMPLE_SIZE)
            ]
            sample_chunk_count = 0
            
            for i in range(0, len(sample_files), DOCUMENT_BATCH_SIZE):
                batch = sample_files[i:i + DOCUMENT_BATCH_SIZE]
                documents = self.loader.load_file_batch(
                    batch, repo_path_obj, "estimate", "estimate", "estimate", "estimate"
                )
                if documents:
                    chunks = self.splitter.split_documents(documents)
                    sample_chunk_count += len(chunks)
                    del chunks
                del documents
                gc.collect()
            
            avg_chunks_per_file = sample_chunk_count / SAMPLE_SIZE
            chunk_count = int(avg_chunks_per_file * file_count)
            logger.info(f"Estimated ~{avg_chunks_per_file:.1f} chunks/file from {SAMPLE_SIZE} samples")
            gc.collect()

        logger.info(f"Estimated {chunk_count} chunks from {file_count} files")
        return file_count, chunk_count
    
    def index_repository(
        self,
        repo_path: str,
        workspace: str,
        project: str,
        branch: str,
        commit: str,
        alias_name: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        source_tree_sha256: str = "",
        source_tree=None,
        operation_id: Optional[str] = None,
        activation_guard: Optional[Callable[[], None]] = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
        project_type: Optional[str] = None,
        source_root: Optional[str] = None,
    ) -> IndexStats:
        """Index entire repository for a branch using atomic swap strategy."""
        def report_progress(
            stage: str,
            message: str,
            progress: Optional[int] = None,
            total: Optional[int] = None,
            **details,
        ) -> None:
            if progress_callback is None:
                return
            event = {"stage": stage, "message": message}
            if progress is not None:
                event["progress"] = max(0, min(100, progress))
            if total is not None:
                event["total"] = total
            event.update({key: value for key, value in details.items()
                          if value is not None})
            try:
                progress_callback(event)
            except Exception as exception:
                # Progress reporting is optional enrichment. It must never fail
                # an otherwise healthy index operation.
                logger.warning(
                    "RAG progress callback failed at stage %s: %s",
                    stage,
                    exception,
                )

        operation_id = operation_id or hashlib.sha256(
            f"{workspace}\0{project}\0{branch}\0{commit}\0{time.time_ns()}".encode()
        ).hexdigest()[:32]
        operation_started = time.perf_counter()
        logger.info(
            "Indexing repository operation_id=%s workspace=%s project=%s "
            "branch=%s repo_path=%s",
            operation_id,
            workspace,
            project,
            branch,
            repo_path,
        )
        report_progress("preparing", "Preparing the pending structural collection", 2)

        repo_path_obj = Path(repo_path)
        pending_collection_name = self.collection_manager.create_pending_collection(
            alias_name,
            operation_id=operation_id,
        )

        activation_aliases = [alias_name]
        activation_alias_targets = self.collection_manager.read_alias_targets(
            activation_aliases
        )

        direct_activation_collections = {
            name
            for name in activation_aliases
            if self.collection_manager.physical_collection_exists(name)
        }

        skipped_file_paths: set[str] = set()
        oversized_file_paths: set[str] = set()

        def record_loader_skip(diagnostic: RepositoryFileSkip) -> None:
            normalized_path = clean_archive_path(
                Path(diagnostic.path).as_posix()
            )
            skipped_file_paths.add(normalized_path)
            if diagnostic.code == REPOSITORY_FILE_SIZE_LIMIT_CODE:
                oversized_file_paths.add(normalized_path)

        # Get file list
        repository_file_list = list(
            self.loader.iter_repository_files(
                repo_path_obj,
                include_patterns,
                exclude_patterns,
                expected_file_sha256=(
                    source_tree.file_sha256_by_path if source_tree else None
                ),
                on_skip=record_loader_skip,
            )
        )
        logger.info(
            "Found %s repository files before plugin file policy for branch '%s'",
            len(repository_file_list),
            branch,
        )
        report_progress(
            "scanning",
            f"Discovered {len(repository_file_list)} repository files",
            7,
            len(repository_file_list),
            skippedFiles=len(skipped_file_paths),
            oversizedFileCount=len(oversized_file_paths),
            maxFileSizeBytes=self.config.max_file_size_bytes,
        )
        
        capabilities = None
        implementation_fingerprint = "sha256:" + "0" * 64
        repository_analysis = None
        repository_facts = None
        if self.plugin_catalog is not None and self.plugin_selector is not None:
            from codecrow_plugins import build_repository_facts

            repository_facts = build_repository_facts(
                repo_path_obj,
                commit,
                repository_file_list,
                self.plugin_catalog.registry,
                project_type=project_type,
                source_root=source_root,
            )
            capabilities = self.plugin_selector.select(repository_facts)
            implementation_fingerprint = (
                self.plugin_catalog.implementation_fingerprint(
                    capabilities.repository_plugins
                )
            )
            logger.info(
                "Selected plugins for %s: %s",
                commit,
                ", ".join(capabilities.repository_plugins) or "generic fallback",
            )
            report_progress(
                "framework",
                "Selected repository plugins: "
                + (", ".join(capabilities.repository_plugins) or "generic fallback"),
                10,
            )
        elif not repository_file_list:
            # A repository whose complete selected input was quarantined is
            # still an exact generation. Persist the neutral inventory as its
            # non-searchable member so the usual manifest, integrity checks,
            # and atomic activation replace any stale branch generation.
            from codecrow_plugins import RepositoryFacts

            repository_facts = RepositoryFacts(
                revision=commit,
                paths=(),
                project_type=project_type,
                source_root=source_root,
            )

        if not repository_file_list:
            logger.warning(
                "No repository files are eligible for structural indexing; "
                "publishing an exact empty generation (skipped=%s, oversized=%s)",
                len(skipped_file_paths),
                len(oversized_file_paths),
            )

        file_list = repository_file_list
        source_paths = {
            clean_archive_path(Path(path).as_posix()) for path in file_list
        }
        if self.plugin_runtime is not None and capabilities is not None:
            from codecrow_plugins import FileDisposition

            dispositions = {
                clean_archive_path(Path(path).as_posix()): self.plugin_runtime.file_disposition(
                    clean_archive_path(Path(path).as_posix()), capabilities
                )
                for path in repository_file_list
            }
            file_list = [
                path for path in repository_file_list
                if dispositions[clean_archive_path(Path(path).as_posix())]
                not in {
                    FileDisposition.EXCLUDED,
                    FileDisposition.GENERATED,
                }
            ]
            source_paths = {
                path
                for path, disposition in dispositions.items()
                if disposition is FileDisposition.FULL
            }
            logger.info(
                "Plugin file policy selected %s source files and %s architecture-only files",
                len(source_paths),
                len(file_list) - len(source_paths),
            )
        total_files = len(source_paths)
        report_progress(
            "scope",
            f"Selected {total_files} source files for structural indexing",
            12,
            total_files,
        )

        analysis_handle = None
        if self.plugin_runtime is not None and capabilities is not None:
            analysis_handle = self.plugin_runtime.start_repository_analysis(
                capabilities,
                commit,
                source_root=repository_facts.source_root if repository_facts else None,
            )
        
        # Validate limits
        if self.config.max_files_per_index > 0 and total_files > self.config.max_files_per_index:
            self.collection_manager.delete_collection(pending_collection_name)
            raise ValueError(
                f"Repository exceeds file limit: {total_files} files (max: {self.config.max_files_per_index})."
            )
        
        if self.config.max_chunks_per_index > 0:
            logger.info("Estimating chunk count before indexing...")
            report_progress("estimating", "Estimating repository chunk count", 14)
            _, estimated_chunks = self.estimate_repository_size(
                repo_path,
                include_patterns,
                exclude_patterns
            )
            if estimated_chunks > self.config.max_chunks_per_index * 1.2:
                self.collection_manager.delete_collection(pending_collection_name)
                raise ValueError(
                    f"Repository estimated to exceed chunk limit: ~{estimated_chunks} chunks (max: {self.config.max_chunks_per_index})."
                )

        document_count = 0
        chunk_count = 0
        successful_chunks = 0
        skipped_chunk_count = 0
        architecture_source_paths: set[str] = set()
        architecture_source_record_count = 0
        estimated_chunks = None

        # Chunk totals are an estimate rather than an expensive second exact
        # indexing pass.  Progress remains useful even when estimation fails.
        if progress_callback is not None:
            try:
                _, estimated_chunks = self.estimate_repository_size(
                    repo_path, include_patterns, exclude_patterns
                )
                report_progress(
                    "estimating",
                    f"Estimated approximately {estimated_chunks} chunks",
                    14,
                    estimatedChunks=estimated_chunks,
                )
            except Exception as exception:
                logger.warning("Could not estimate RAG progress chunks: %s", exception)

        direct_cutover_started = False
        try:
            # Stream process files in batches
            logger.info("Starting memory-efficient streaming indexing...")
            batch_num = 0
            total_batches = (len(file_list) + DOCUMENT_BATCH_SIZE - 1) // DOCUMENT_BATCH_SIZE
            report_progress(
                "indexing",
                f"Starting {total_batches} indexing batches",
                18,
                total_batches,
                totalBatches=total_batches,
                indexedChunks=0,
                estimatedChunks=estimated_chunks,
                skippedFiles=len(skipped_file_paths),
                oversizedFileCount=len(oversized_file_paths),
                maxFileSizeBytes=self.config.max_file_size_bytes,
            )
            
            # Architecture-only files still have to reach the repository
            # resolver. ``total_files`` counts source-bearing files
            # for admission control, so using it as the iteration bound would
            # silently truncate the resolver input whenever a plugin removes
            # files from source chunking.
            completed_batch_duration_ms = 0
            for i in range(0, len(file_list), DOCUMENT_BATCH_SIZE):
                batch_num += 1
                file_batch = file_list[i:i + DOCUMENT_BATCH_SIZE]
                batch_started = time.perf_counter()
                load_started = time.perf_counter()
                documents = self.loader.load_file_batch(
                    file_batch, repo_path_obj, workspace, project, branch, commit,
                    strict=False,
                    expected_file_sha256=(
                        source_tree.file_sha256_by_path if source_tree else None
                    ),
                    on_skip=record_loader_skip,
                )
                load_duration_ms = round(
                    (time.perf_counter() - load_started) * 1000
                )
                loaded_paths = {
                    document.metadata["path"] for document in documents
                }
                missing_paths = {
                    clean_archive_path(Path(path).as_posix())
                    for path in file_batch
                } - loaded_paths
                for path in sorted(missing_paths):
                    logger.warning(
                        "Skipping repository file that could not be loaded: %s",
                        path,
                    )
                skipped_file_paths.update(missing_paths)
                if (
                    documents
                    and analysis_handle is not None
                    and analysis_handle.active
                ):
                    from codecrow_plugins import FileArtifact

                    artifacts = tuple(sorted(
                        (
                            FileArtifact(
                                path=document.metadata["path"],
                                content=document.text,
                            )
                            for document in documents
                        ),
                        key=lambda artifact: artifact.path,
                    ))
                    def report_repository_ingest(event: dict) -> None:
                        status = str(event.get("status", "processing"))
                        plugin_id = str(event.get("pluginId", "unknown"))
                        logger.info(
                            "RAG repository plugin ingest operation_id=%s "
                            "batch=%s/%s plugin=%s status=%s files=%s "
                            "first_path=%s last_path=%s duration_ms=%s",
                            operation_id,
                            batch_num,
                            total_batches,
                            plugin_id,
                            status,
                            event.get("files"),
                            event.get("firstPath"),
                            event.get("lastPath"),
                            event.get("durationMs"),
                        )
                        report_progress(
                            "architecture_ingest",
                            str(event.get(
                                "message",
                                "Ingesting repository architecture inputs",
                            )),
                            18 + round(
                                67 * max(batch_num - 1, 0)
                                / max(total_batches, 1)
                            ),
                            total_batches,
                            architecturePlugin=plugin_id,
                            architectureSubstage="ingest",
                            architectureStatus=status,
                            completedBatches=max(batch_num - 1, 0),
                            totalBatches=total_batches,
                            files=event.get("files"),
                            firstPath=event.get("firstPath"),
                            lastPath=event.get("lastPath"),
                            substageDurationMs=event.get("durationMs"),
                        )

                    analysis_handle.ingest(
                        artifacts,
                        progress_callback=report_repository_ingest,
                    )

                source_documents = [
                    document
                    for document in documents
                    if document.metadata["path"] in source_paths
                ]
                architecture_only_documents = [
                    document
                    for document in documents
                    if document.metadata["path"] not in source_paths
                ]
                architecture_only_count = len(architecture_only_documents)
                split_duration_ms = 0
                point_pipeline_duration_ms = 0
                batch_chunk_count = 0
                chunks = []

                if source_documents:
                    split_started = time.perf_counter()
                    chunks, split_skipped_paths = (
                        self.splitter.split_documents_resilient(
                            source_documents,
                            capabilities=capabilities,
                        )
                    )
                    split_duration_ms = round(
                        (time.perf_counter() - split_started) * 1000
                    )
                    skipped_file_paths.update(split_skipped_paths)
                    document_count += (
                        len(source_documents) - len(split_skipped_paths)
                    )
                    identity_metadata = _plugin_identity_metadata(
                        capabilities,
                        implementation_fingerprint,
                        self.representation_fingerprint,
                    )
                    for chunk in chunks:
                        chunk.metadata.update(identity_metadata)

                architecture_source_nodes = (
                    self._architecture_only_source_nodes(
                        architecture_only_documents,
                        capabilities,
                        workspace,
                        project,
                        branch,
                        commit,
                        implementation_fingerprint,
                        self.representation_fingerprint,
                    )
                )
                chunks.extend(architecture_source_nodes)
                architecture_source_record_count += len(
                    architecture_source_nodes
                )
                architecture_source_paths.update(
                    node.metadata["path"]
                    for node in architecture_source_nodes
                )
                batch_chunk_count = len(chunks)
                chunk_count += batch_chunk_count

                # Raw architecture sources are structural records too and
                # therefore consume the same hard generation limit.
                if (
                    self.config.max_chunks_per_index > 0
                    and chunk_count > self.config.max_chunks_per_index
                ):
                    self.collection_manager.delete_collection(
                        pending_collection_name
                    )
                    raise ValueError(
                        f"Repository exceeds chunk limit: {chunk_count}+ chunks."
                    )

                if chunks:
                    point_pipeline_started = time.perf_counter()
                    success, failed = self.point_ops.process_and_store_chunks(
                        chunks,
                        pending_collection_name,
                        workspace,
                        project,
                        branch,
                        operation_id=operation_id,
                    )
                    point_pipeline_duration_ms = round(
                        (time.perf_counter() - point_pipeline_started) * 1000
                    )
                    successful_chunks += success
                    skipped_chunk_count += failed
                    if failed:
                        logger.warning(
                            "Skipped %s rejected chunks in batch %s; "
                            "continuing repository indexing",
                            failed,
                            batch_num,
                        )

                batch_duration_ms = round(
                    (time.perf_counter() - batch_started) * 1000
                )
                logger.info(
                    "RAG document batch completed operation_id=%s batch=%s/%s "
                    "source_files=%s architecture_only_files=%s "
                    "architecture_source_records=%s chunks=%s "
                    "load_duration_ms=%s "
                    "split_duration_ms=%s point_pipeline_duration_ms=%s "
                    "duration_ms=%s",
                    operation_id,
                    batch_num,
                    total_batches,
                    len(source_documents),
                    architecture_only_count,
                    len(architecture_source_nodes),
                    batch_chunk_count,
                    load_duration_ms,
                    split_duration_ms,
                    point_pipeline_duration_ms,
                    batch_duration_ms,
                )
                batch_progress = 18 + round(67 * batch_num / max(total_batches, 1))
                completed_batch_duration_ms += batch_duration_ms
                average_batch_ms = completed_batch_duration_ms / batch_num
                estimated_remaining_ms = round(
                    average_batch_ms * max(total_batches - batch_num, 0)
                )
                report_progress(
                    "indexing",
                    (
                        f"Indexed batch {batch_num}/{total_batches}: "
                        f"{document_count}/{total_files} files, "
                        f"{successful_chunks} chunks"
                    ),
                    batch_progress,
                    total_batches,
                    indexedChunks=successful_chunks,
                    estimatedChunks=estimated_chunks,
                    sourceFiles=len(source_documents),
                    architectureOnlyFiles=architecture_only_count,
                    architectureSourceRecords=len(
                        architecture_source_nodes
                    ),
                    completedBatches=batch_num,
                    totalBatches=total_batches,
                    batchDurationMs=batch_duration_ms,
                    estimatedRemainingMs=estimated_remaining_ms,
                    remainingEstimateScope="file_batches",
                    skippedFiles=len(skipped_file_paths),
                    oversizedFileCount=len(oversized_file_paths),
                    maxFileSizeBytes=self.config.max_file_size_bytes,
                )

                del documents
                del chunks

                if batch_num % 5 == 0:
                    gc.collect()

            analysis_nodes = []
            architecture_nodes = []
            context_nodes = []
            snapshot_nodes = []
            symbol_nodes = []
            if analysis_handle is not None:
                configured_architecture_timeout = getattr(
                    self.config,
                    "architecture_finalization_timeout_seconds",
                    600,
                )
                if not isinstance(configured_architecture_timeout, (int, float)):
                    configured_architecture_timeout = 600
                architecture_timeout_seconds = max(
                    1.0,
                    float(configured_architecture_timeout),
                )
                architecture_deadline = (
                    time.monotonic() + architecture_timeout_seconds
                )
                report_progress(
                    "architecture",
                    "Building deterministic architecture context",
                    88,
                    architectureTimeoutSeconds=architecture_timeout_seconds,
                    estimatedRemainingMs=0,
                    remainingEstimateScope="file_batches_complete",
                )

                def report_architecture_progress(event: dict) -> None:
                    report_progress(
                        "architecture",
                        str(event.get(
                            "message",
                            "Building deterministic architecture context",
                        )),
                        88 if event.get("status") == "started" else 89,
                        architectureTimeoutSeconds=architecture_timeout_seconds,
                        architecturePlugin=event.get("pluginId"),
                        architectureSubstage=event.get("substage"),
                        architectureStatus=event.get("status"),
                        sourceRoot=event.get("sourceRoot"),
                        substageDurationMs=event.get("durationMs"),
                    )

                repository_analysis, diagnostics = analysis_handle.finish(
                    progress_callback=report_architecture_progress,
                    deadline=architecture_deadline,
                )
                skipped_file_paths.update(
                    self.accept_recoverable_repository_diagnostics(
                        diagnostics,
                        "repository architecture analysis",
                    )
                )
                architecture_timed_out = any(
                    diagnostic.code
                    == "plugin-repository-finalization-timeout"
                    for diagnostic in diagnostics
                )
                if architecture_timed_out:
                    logger.warning(
                        "Repository architecture finalization exceeded %.1fs; "
                        "continuing operation_id=%s without deterministic "
                        "architecture context",
                        architecture_timeout_seconds,
                        operation_id,
                    )
                    report_progress(
                        "architecture",
                        (
                            "Architecture time budget exhausted; continuing "
                            "with source indexing"
                        ),
                        90,
                        architectureStatus="degraded",
                        degraded=True,
                        architectureTimeoutSeconds=architecture_timeout_seconds,
                    )
                else:
                    try:
                        report_progress(
                            "architecture",
                            "Materializing deterministic architecture records",
                            89,
                            architectureStatus="materializing",
                        )
                        architecture_nodes = self._architecture_nodes(
                            repository_analysis,
                            capabilities,
                            workspace,
                            project,
                            branch,
                            commit,
                            implementation_fingerprint,
                            self.representation_fingerprint,
                        )
                        snapshot_nodes = self._snapshot_nodes(
                            repository_analysis,
                            capabilities,
                            workspace,
                            project,
                            branch,
                            commit,
                            implementation_fingerprint,
                            self.representation_fingerprint,
                        )
                        symbol_nodes = self._symbol_nodes(
                            repository_analysis,
                            capabilities,
                            workspace,
                            project,
                            branch,
                            commit,
                            implementation_fingerprint,
                            self.representation_fingerprint,
                        )
                        context_nodes = self._repository_context_nodes(
                            repository_analysis,
                            capabilities,
                            workspace,
                            project,
                            branch,
                            commit,
                            implementation_fingerprint,
                            self.representation_fingerprint,
                            excluded_paths=architecture_source_paths,
                        )
                        analysis_nodes.extend(
                            (
                                *architecture_nodes,
                                *context_nodes,
                                *snapshot_nodes,
                                *symbol_nodes,
                            )
                        )
                    except Exception as exception:
                        architecture_nodes = []
                        snapshot_nodes = []
                        context_nodes = []
                        symbol_nodes = []
                        logger.warning(
                            "Repository architecture materialization failed; "
                            "continuing operation_id=%s with source indexing: %s",
                            operation_id,
                            exception,
                            exc_info=True,
                        )
                        report_progress(
                            "architecture",
                            (
                                "Architecture materialization failed; continuing "
                                "with source indexing"
                            ),
                            90,
                            architectureStatus="degraded",
                            degraded=True,
                            architectureFailure=type(exception).__name__,
                        )

                # Repository sessions retain the architecture-only source
                # artifacts used to build snapshots, while RepositoryAnalysis
                # retains the unsliced snapshot strings. Once bounded storage
                # nodes exist, release both before PointStruct/Qdrant payloads
                # are materialized to avoid holding three copies concurrently.
                analysis_handle = None
                del repository_analysis
                gc.collect()

            facts_nodes = []
            if repository_facts is not None:
                facts_nodes = self._repository_facts_nodes(
                    repository_facts,
                    capabilities,
                    workspace,
                    project,
                    branch,
                    commit,
                    implementation_fingerprint,
                    self.representation_fingerprint,
                )
                analysis_nodes.extend(facts_nodes)

            if analysis_nodes:
                architecture_count = len(analysis_nodes)
                chunk_count += architecture_count
                if (
                    self.config.max_chunks_per_index > 0
                    and chunk_count > self.config.max_chunks_per_index
                ):
                    raise ValueError(
                        f"Repository exceeds chunk limit after architecture analysis: {chunk_count} chunks."
                    )
                report_progress(
                    "architecture",
                    f"Persisting {architecture_count} deterministic context records",
                    90,
                    architectureStatus="persisting",
                    architectureRecords=architecture_count,
                )
                success, failed = self.point_ops.process_and_store_chunks(
                    analysis_nodes,
                    pending_collection_name,
                    workspace,
                    project,
                    branch,
                    operation_id=operation_id,
                )
                successful_chunks += success
                skipped_chunk_count += failed
                if failed:
                    logger.warning(
                        "Skipped %s rejected deterministic context points; "
                        "continuing repository indexing",
                        failed,
                    )
                logger.info(
                    "Indexed %s architecture packets, %s exact source parts, "
                    "%s repository snapshots, %s symbols, and %s repository-fact parts",
                    len(architecture_nodes),
                    architecture_source_record_count + len(context_nodes),
                    len(snapshot_nodes),
                    len(symbol_nodes),
                    len(facts_nodes),
                )
                report_progress(
                    "architecture",
                    (
                        f"Persisted {success} deterministic context records"
                    ),
                    91,
                    architectureStatus="completed",
                    architectureRecords=success,
                    skippedArchitectureRecords=failed,
                )

            report_progress(
                "sealing",
                "Sealing persisted payloads for generation integrity",
                92,
                indexedChunks=successful_chunks,
                estimatedChunks=estimated_chunks,
            )
            sealing_started = time.perf_counter()

            def report_sealing_progress(sealed_members: int) -> None:
                report_progress(
                    "sealing",
                    (
                        f"Sealed {sealed_members}/{successful_chunks} "
                        "persisted members"
                    ),
                    92 if sealed_members < successful_chunks else 93,
                    sealedMembers=sealed_members,
                    expectedMembers=successful_chunks,
                )

            members = seal_generation_members(
                self.point_ops.client,
                pending_collection_name,
                branch,
                commit,
                progress_callback=report_sealing_progress,
            )
            logger.info(
                "RAG generation members sealed operation_id=%s "
                "members=%s duration_ms=%s",
                operation_id,
                len(members),
                round((time.perf_counter() - sealing_started) * 1000),
            )
            identity_metadata = _plugin_identity_metadata(
                capabilities,
                implementation_fingerprint,
                self.representation_fingerprint,
            )
            if source_tree is not None:
                require_repository_source_tree_unchanged(
                    repo_path_obj,
                    source_tree,
                )
            if not source_tree_sha256:
                raise RuntimeError(
                    "repository source-tree identity is required to seal an index generation"
                )
            manifest = build_generation_manifest_node(
                workspace=workspace,
                project=project,
                branch=branch,
                commit=commit,
                member_count=len(members),
                members_sha256=compute_generation_members_digest(members),
                source_tree_sha256=source_tree_sha256,
                index_include_patterns=include_patterns or (),
                index_exclude_patterns=exclude_patterns or (),
                identity_metadata=identity_metadata,
            )
            manifest_success, manifest_failed = (
                self.point_ops.process_and_store_chunks(
                    [manifest],
                    pending_collection_name,
                    workspace,
                    project,
                    branch,
                    operation_id=operation_id,
                )
            )
            if manifest_success != 1 or manifest_failed:
                raise RuntimeError(
                    "repository generation manifest could not be persisted"
                )
            generation_manifest_points = 1
            generation_manifest_sha256 = manifest.metadata[
                "generation_manifest_sha256"
            ]

            logger.info(
                f"Streaming indexing complete: {document_count} files, "
                f"{successful_chunks}/{chunk_count} chunks indexed "
                f"({skipped_chunk_count} skipped across "
                f"{len(skipped_file_paths)} files)"
            )

            # Verify and perform atomic swap
            report_progress("verifying", "Verifying the pending structural collection", 94)
            pending_info = self.point_ops.client.get_collection(pending_collection_name)
            actual_point_count = int(pending_info.points_count or 0)
            expected_point_count = successful_chunks + generation_manifest_points
            if actual_point_count != expected_point_count:
                raise RuntimeError(
                    "Pending collection point count is incomplete: "
                    f"expected={expected_point_count}, actual={actual_point_count}"
                )

            target_branch_point_count = (
                self.branch_manager.get_branch_point_count(
                    pending_collection_name,
                    branch,
                )
            )
            expected_target_branch_points = (
                successful_chunks + generation_manifest_points
            )
            if target_branch_point_count != expected_target_branch_points:
                raise RuntimeError(
                    "Pending target-branch point count is incomplete: "
                    f"branch={branch}, expected={expected_target_branch_points}, "
                    f"actual={target_branch_point_count}"
                )

            if activation_guard is not None:
                activation_guard()
            observed_targets = self.collection_manager.read_alias_targets(
                activation_aliases
            )
            if observed_targets != activation_alias_targets:
                raise RuntimeError(
                    "Active RAG alias changed before pending activation"
                )

            activation_started = time.perf_counter()
            report_progress("activating", "Activating the completed structural collection", 97)
            direct_cutover_started = bool(direct_activation_collections)
            old_targets = self._perform_atomic_swap(
                alias_name,
                pending_collection_name,
                activation_aliases,
            )
            logger.info(
                "RAG pending collection activated operation_id=%s collection=%s "
                "duration_ms=%s",
                operation_id,
                pending_collection_name,
                round((time.perf_counter() - activation_started) * 1000),
            )

            try:
                self.stats_manager.store_metadata(
                    workspace,
                    project,
                    branch,
                    commit,
                    document_count,
                    successful_chunks,
                )
            except Exception:
                if not direct_cutover_started:
                    self._rollback_atomic_swap(old_targets)
                raise

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            if direct_cutover_started:
                logger.error(
                    "Retaining completed structural collection %s after a "
                    "direct-name cutover failure",
                    pending_collection_name,
                )
            else:
                self.collection_manager.delete_collection(
                    pending_collection_name
                )
            raise
        finally:
            gc.collect()

        namespace = make_namespace(workspace, project, branch)
        logger.info(
            "Structural repository index completed operation_id=%s workspace=%s "
            "project=%s branch=%s files=%s records=%s "
            "duration_ms=%s",
            operation_id,
            workspace,
            project,
            branch,
            document_count,
            successful_chunks,
            round((time.perf_counter() - operation_started) * 1000),
        )
        report_progress(
            "complete",
            f"Indexed {document_count} files into {successful_chunks} chunks",
            100,
            document_count,
            indexedChunks=successful_chunks,
            estimatedChunks=successful_chunks,
            completedBatches=total_batches,
            totalBatches=total_batches,
            estimatedRemainingMs=0,
            skippedFiles=len(skipped_file_paths),
            oversizedFileCount=len(oversized_file_paths),
            maxFileSizeBytes=self.config.max_file_size_bytes,
        )
        return IndexStats(
            namespace=namespace,
            document_count=document_count,
            chunk_count=successful_chunks,
            skipped_file_count=len(skipped_file_paths),
            skipped_chunk_count=skipped_chunk_count,
            last_updated=datetime.now(timezone.utc).isoformat(),
            workspace=workspace,
            project=project,
            branch=branch,
            generation_manifest_sha256=generation_manifest_sha256,
            source_tree_sha256=source_tree_sha256,
            collection_target=alias_name,
        )
    
    def _perform_atomic_swap(
            self,
            alias_name: str,
            pending_collection_name: str,
            activation_aliases: List[str],
    ) -> dict[str, Optional[str]]:
        """Activate a complete generation at its exact target."""
        logger.info("Performing atomic alias swap...")
        old_targets = self.collection_manager.read_alias_targets(activation_aliases)
        direct_collections = [
            alias
            for alias in activation_aliases
            if self.collection_manager.physical_collection_exists(alias)
        ]
        for collection_name in direct_collections:
            if not self.collection_manager.delete_collection(collection_name):
                raise RuntimeError(
                    "could not remove the pre-structural direct collection "
                    f"during cutover: {collection_name}"
                )
        self.collection_manager.atomic_assign_aliases(
            {alias: pending_collection_name for alias in activation_aliases}
        )
        return old_targets

    def _rollback_atomic_swap(self, old_targets: dict[str, Optional[str]]) -> None:
        """Restore every alias changed during a failed metadata publication."""
        self.collection_manager.atomic_assign_aliases(old_targets)


class PrOverlayOperations:
    """Publish sealed PR overlay records with rollback protection."""

    def __init__(self, client, point_ops: PointOperations):
        self.client = client
        self.point_ops = point_ops

    @staticmethod
    def _as_point_struct(record) -> PointStruct:
        return PointStruct(
            id=record.id,
            vector=STORAGE_MARKER_VECTOR,
            payload=record.payload,
        )

    def _delete_point_ids(self, collection_name: str, point_ids) -> None:
        point_ids = list(point_ids)
        for offset in range(0, len(point_ids), 512):
            self.client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(
                    points=point_ids[offset:offset + 512],
                ),
            )

    def _restore_old_points(
        self,
        collection_name: str,
        old_points,
        new_only_ids,
    ) -> None:
        """Restore the exact pre-mutation point set after a failed replacement."""
        rollback_failures = []
        old_structs = [self._as_point_struct(point) for point in old_points.values()]
        for offset in range(0, len(old_structs), 128):
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=old_structs[offset:offset + 128],
                    wait=True,
                )
            except Exception as exception:
                rollback_failures.append(exception)
        try:
            self._delete_point_ids(collection_name, new_only_ids)
        except Exception as exception:
            rollback_failures.append(exception)
        if rollback_failures:
            raise RuntimeError(
                "PR overlay replacement failed and rollback was incomplete"
            ) from rollback_failures[0]

    def replace_pr_overlay_generation(
        self,
        nodes,
        old_records,
        collection_name: str,
        workspace: str,
        project: str,
        point_id_branch: str,
        mutation_guard: Optional[Callable[[], None]] = None,
        *,
        pr_number: int,
        branch: str,
        base_branch: str,
        source_revision: str,
        base_revision: str,
        base_generation_manifest_sha256: str,
        generation_fingerprint: str,
        overlay_representation_fingerprint: str,
        identity_metadata,
    ) -> tuple[int, dict]:
        """Prepare all PR members, seal them, and publish one complete set."""
        from ..pr_overlay_manifest import build_pr_overlay_manifest_node

        old_records = list(old_records)
        old_points = {str(record.id): record for record in old_records}
        chunk_data = self.point_ops.prepare_chunks_for_storage(
            nodes, workspace, project, point_id_branch
        )
        new_points = self.point_ops.create_points(chunk_data)
        new_ids = [point.id for point in new_points]
        if mutation_guard is not None:
            mutation_guard()
        successful, failed = self.point_ops.upsert_points(
            collection_name, new_points
        )
        if failed or successful != len(new_points):
            self._delete_point_ids(collection_name, new_ids)
            raise RuntimeError("PR overlay member write was incomplete")
        try:
            members = self.point_ops.verify_persisted_point_digests(
                collection_name, new_points
            )
            manifest_node, receipt = build_pr_overlay_manifest_node(
                workspace=workspace,
                project=project,
                pr_number=pr_number,
                branch=branch,
                base_branch=base_branch,
                source_revision=source_revision,
                base_revision=base_revision,
                base_generation_manifest_sha256=(
                    base_generation_manifest_sha256
                ),
                generation_fingerprint=generation_fingerprint,
                overlay_representation_fingerprint=(
                    overlay_representation_fingerprint
                ),
                members=members,
                identity_metadata=identity_metadata,
            )
            manifest_data = self.point_ops.prepare_chunks_for_storage(
                [manifest_node], workspace, project, point_id_branch
            )
            manifest_points = self.point_ops.create_points(manifest_data)
            manifest_success, manifest_failed = self.point_ops.upsert_points(
                collection_name, manifest_points
            )
            if manifest_success != 1 or manifest_failed:
                raise RuntimeError("PR overlay manifest write was incomplete")
        except Exception:
            self._delete_point_ids(
                collection_name,
                [*new_ids, *(point.id for point in locals().get("manifest_points", []))],
            )
            raise

        active_ids = {str(point.id) for point in (*new_points, *manifest_points)}
        stale_ids = [
            record.id for point_id, record in old_points.items()
            if point_id not in active_ids
        ]
        try:
            if mutation_guard is not None:
                mutation_guard()
            self._delete_point_ids(collection_name, stale_ids)
        except Exception:
            self._restore_old_points(
                collection_name,
                old_points,
                [*new_ids, *(point.id for point in manifest_points)],
            )
            raise
        return successful, receipt
