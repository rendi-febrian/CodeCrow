"""PR file indexing endpoints."""
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

from ..models import PRIndexRequest
from ...core.repository_overlay import (
    build_overlay_capabilities,
    load_repository_snapshots,
)
from ...core.exact_index import ExactIndexPreconditionError
from ...core.coordination import (
    MutationCoordinationUnavailable,
    MutationLeaseUnavailable,
)
from ...core.pr_overlay_identity import (
    ZERO_FINGERPRINT,
    pr_overlay_generation_fingerprint,
)
from ...core.review_grouping import review_groups_from_architecture_payloads
from ...core.index_representation import (
    INDEX_REPRESENTATION_PAYLOAD_KEY,
    observe_branch_representation,
)
from ...core.pr_overlay_representation import (
    PR_OVERLAY_REPRESENTATION_PAYLOAD_KEY,
)
from ...core.pr_overlay_manifest import (
    read_pr_overlay_generation,
)
from ...core.revision_binding import require_repository_generation
from ...core.documents import Document
from ...core.loader import REPOSITORY_FILE_SIZE_LIMIT_CODE
from ...models.config import DEFAULT_MAX_FILE_SIZE_BYTES

logger = logging.getLogger(__name__)
router = APIRouter(tags=["pr"])


def _get_index_manager():
    from ..api import index_manager
    return index_manager


def _content_state(file_info: object) -> str:
    state = getattr(file_info, "content_state", "complete")
    return state if state in {"complete", "partial_diff"} else "complete"


def _configured_max_file_size_bytes(index_manager: object) -> int:
    config = getattr(index_manager, "config", None)
    configured = getattr(
        config,
        "max_file_size_bytes",
        DEFAULT_MAX_FILE_SIZE_BYTES,
    )
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured < 1
    ):
        return DEFAULT_MAX_FILE_SIZE_BYTES
    return configured


def _utf8_size_bytes(content: str) -> int:
    return len(content.encode("utf-8"))


def _effective_detection_evidence(
    *,
    repository_plugins: tuple[str, ...],
    stored_plugin_ids: tuple[str, ...],
    requested_evidence: dict[str, list[str]],
    target_branch: str,
    stored_fingerprint: str,
) -> dict[str, tuple[str, ...]]:
    """Bind the effective plugin set to target-index and PR evidence."""
    indexed = set(stored_plugin_ids)
    result: dict[str, tuple[str, ...]] = {}
    for plugin_id in repository_plugins:
        evidence = set(requested_evidence.get(plugin_id, ()))
        if plugin_id in indexed:
            evidence.add(
                "indexed-target:"
                f"{target_branch}:"
                f"{stored_fingerprint}:"
                f"{plugin_id}"
            )
        if not evidence:
            raise RuntimeError(
                f"effective repository plugin {plugin_id} has no "
                "revision-bound selection evidence"
            )
        result[plugin_id] = tuple(sorted(evidence))
    return result


def _capabilities_payload(capabilities, implementation_fingerprint: str):
    if capabilities is None:
        return None
    return {
        "repositoryPlugins": list(capabilities.repository_plugins),
        "filePlugins": {
            path: list(plugin_ids)
            for path, plugin_ids in capabilities.file_plugins.items()
        },
        "detectionEvidence": {
            plugin_id: list(evidence)
            for plugin_id, evidence
            in capabilities.detection_evidence.items()
        },
        "unavailableCapabilities": list(
            capabilities.unavailable_capabilities
        ),
        "fingerprint": capabilities.fingerprint,
        "descriptorFingerprint": capabilities.descriptor_fingerprint,
        "implementationFingerprint": implementation_fingerprint,
    }


def _target_architecture_payloads(
    index_manager,
    collection_name: str,
    *,
    workspace: str,
    project: str,
    branch: str,
    revision: str | None,
    changed_paths,
):
    """Read exact target-branch graph facts for the changed paths.

    Architecture lookup is auxiliary to PR indexing.  A storage failure must
    therefore remain visible without rejecting an otherwise reviewable PR.
    """
    paths = tuple(sorted({path for path in changed_paths if path}))
    if not paths:
        return (), ()

    payloads = {}
    try:
        for path_offset in range(0, len(paths), 64):
            conditions = [
                FieldCondition(
                    key="workspace",
                    match=MatchValue(value=workspace),
                ),
                FieldCondition(
                    key="project",
                    match=MatchValue(value=project),
                ),
                FieldCondition(
                    key="branch",
                    match=MatchValue(value=branch),
                ),
                FieldCondition(
                    key="architecture_context",
                    match=MatchValue(value=True),
                ),
                FieldCondition(
                    key="architecture_paths",
                    match=MatchAny(any=paths[path_offset:path_offset + 64]),
                ),
            ]
            if revision:
                conditions.append(FieldCondition(
                    key="commit",
                    match=MatchValue(value=revision),
                ))

            offset = None
            while True:
                points, offset = index_manager.qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(
                        must=conditions,
                        must_not=[FieldCondition(
                            key="pr",
                            match=MatchValue(value=True),
                        )],
                    ),
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payloads[str(point.id)] = point.payload or {}
                if offset is None:
                    break
    except Exception as exception:
        logger.warning(
            "Could not load target-branch architecture facts for PR review "
            "groups collection=%s branch=%s: %s",
            collection_name,
            branch,
            exception,
            exc_info=True,
        )
        return (), ({
            "code": "target_architecture_facts_unavailable",
            "message": (
                "Exact target-branch architecture facts could not be loaded; "
                "PR indexing continued with available overlay facts."
            ),
            "target_branch": branch,
        },)

    return tuple(payloads.values()), ()


def _normalized_repository_path(value):
    if not isinstance(value, str):
        return ""
    return value.strip().replace("\\", "/").lstrip("/")


def _target_fallback_payloads(target_payloads, changed_paths, fallback_paths):
    """Keep base facts only when every changed member lacks post-change source.

    Overlay facts are authoritative for complete and deleted artifacts. A base
    fact may fill a gap for partial files, but it must not reconnect a partial
    file through a complete/deleted path when the post-change overlay removed
    that relation.
    """
    changed = {
        normalized
        for path in changed_paths
        if (normalized := _normalized_repository_path(path))
    }
    fallback = {
        normalized
        for path in fallback_paths
        if (normalized := _normalized_repository_path(path))
    }
    if not fallback:
        return ()

    focused_payloads = []
    for payload in target_payloads:
        facts = payload.get("plugin_graph_facts")
        if not isinstance(facts, (list, tuple)):
            continue
        focused_facts = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            members = {_normalized_repository_path(fact.get("path"))}
            related_paths = fact.get("related_paths")
            if isinstance(related_paths, (list, tuple)):
                members.update(
                    _normalized_repository_path(path)
                    for path in related_paths
                )
            changed_members = (members - {""}) & changed
            if changed_members and changed_members.issubset(fallback):
                focused_facts.append(fact)
        if focused_facts:
            focused = dict(payload)
            focused["plugin_graph_facts"] = focused_facts
            focused_payloads.append(focused)
    return tuple(focused_payloads)


def _review_groups(
    changed_paths,
    overlay_payloads,
    target_payloads,
    fallback_paths,
):
    """Project groups with post-change facts taking deterministic precedence."""
    target_fallbacks = _target_fallback_payloads(
        target_payloads,
        changed_paths,
        fallback_paths,
    )
    return review_groups_from_architecture_payloads(
        (*target_fallbacks, *overlay_payloads),
        changed_paths,
    )


@router.post("/index/pr-files")
def index_pr_files(request: PRIndexRequest):
    """
    Index PR files into the main collection with PR-specific metadata.

    Files are indexed with metadata: pr=true, pr_number, pr_branch.
    This allows hybrid queries that prioritize PR data over branch data.
    An exact persisted generation is reused. A changed generation is prepared
    completely and then replaces the previous points with rollback protection.
    """
    index_manager = _get_index_manager()
    mutation_context = index_manager.pr_overlay_mutation(
        request.workspace,
        request.project,
        request.pr_number,
        "index-pr-overlay",
    )
    mutation_lease = None
    try:
        mutation_lease = mutation_context.__enter__()
        target_branch = request.base_branch or request.branch
        base_receipt = require_repository_generation(
            index_manager,
            workspace=request.workspace,
            project=request.project,
            branch=target_branch,
            revision=request.base_revision,
            generation_manifest_sha256=(
                request.base_generation_manifest_sha256
            ),
            collection_target=request.collection_target,
        )
        collection_name = base_receipt["_collection_target"]
        base_generation_receipt = {
            "base_generation_manifest_sha256": base_receipt[
                "generation_manifest_sha256"
            ],
            "plugin_fingerprint": base_receipt["plugin_fingerprint"],
            "plugin_descriptor_fingerprint": base_receipt[
                "plugin_descriptor_fingerprint"
            ],
            "plugin_implementation_fingerprint": base_receipt[
                "plugin_implementation_fingerprint"
            ],
            "index_representation_fingerprint": base_receipt[
                "index_representation_fingerprint"
            ],
        }

        # Keep the last complete PR generation until its replacement has been
        # fully parsed and validated. Mutation happens once at the end and the
        # shared replacement primitive restores these payloads with the fixed
        # storage marker if either upsert or stale-point deletion fails.
        old_pr_points = []
        offset = None
        while True:
            points, offset = index_manager.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(must=[
                    FieldCondition(
                        key="pr_number",
                        match=MatchValue(value=request.pr_number),
                    )
                ]),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            old_pr_points.extend(points)
            if offset is None:
                break

        # Recover the exact plugin-owned repository state from the PR target
        # branch.  The host never interprets the snapshots; it only validates,
        # overlays changed artifacts, and asks the selected plugins to rebuild.
        representation_fingerprint = (
            index_manager.index_representation_fingerprint
        )
        overlay_representation_fingerprint = (
            index_manager.pr_overlay_representation_fingerprint
        )
        observe_branch_representation(
            index_manager.qdrant_client,
            collection_name,
            target_branch,
            expected_fingerprint=representation_fingerprint,
        )
        (
            snapshots,
            stored_plugin_ids,
            stored_fingerprint,
            _stored_descriptor_fingerprint,
            _stored_implementation_fingerprint,
            stored_repository_facts,
        ) = load_repository_snapshots(
            index_manager.qdrant_client,
            collection_name,
            target_branch,
            include_facts=True,
        )
        requested_plugin_ids = tuple(request.repository_plugins)
        selected_plugin_ids = (*stored_plugin_ids, *requested_plugin_ids)
        if selected_plugin_ids:
            if (
                index_manager.plugin_catalog is None
                or index_manager.plugin_runtime is None
            ):
                raise RuntimeError("repository plugins are unavailable")
        # The PR request is selected from changed/enriched paths, while the
        # target-branch capability set is selected from the whole repository.
        # Therefore the PR set is normally a subset and its selection
        # fingerprint normally differs across revisions. The indexed target
        # remains authoritative for existing capabilities. A capability
        # introduced entirely by added PR files can start from empty state;
        # modified/deleted evidence still requires a target snapshot.
        missing_requested_plugins = tuple(
            plugin_id
            for plugin_id in requested_plugin_ids
            if plugin_id not in stored_plugin_ids
        )
        repository_plugins = (
            tuple(
                descriptor.id
                for descriptor in index_manager.plugin_catalog.registry.resolve(
                    selected_plugin_ids
                )
            )
            if selected_plugin_ids
            else ()
        )
        implementation_fingerprint = (
            index_manager.plugin_catalog.implementation_fingerprint(
                repository_plugins
            )
            if repository_plugins
            else "sha256:" + "0" * 64
        )
        capabilities = None
        required_snapshot_plugins: set[str] = set()
        fresh_repository_plugins: set[str] = set()
        if repository_plugins:
            effective_detection_evidence = _effective_detection_evidence(
                repository_plugins=repository_plugins,
                stored_plugin_ids=tuple(stored_plugin_ids),
                requested_evidence=request.plugin_detection_evidence,
                target_branch=target_branch,
                stored_fingerprint=stored_fingerprint,
            )
            capabilities = build_overlay_capabilities(
                index_manager.plugin_catalog.registry,
                repository_plugins,
                tuple(sorted(file_info.path for file_info in request.files)),
                revision=request.source_revision,
                detection_evidence=effective_detection_evidence,
            )
            required_snapshot_plugins = set(
                index_manager.plugin_runtime.repository_analysis_plugins(capabilities)
            )
            fresh_repository_plugins = (
                required_snapshot_plugins & set(missing_requested_plugins)
            )
            if fresh_repository_plugins:
                added_paths = {
                    file_info.path
                    for file_info in request.files
                    if file_info.change_type == "ADDED"
                }
                request_paths = {file_info.path for file_info in request.files}
                unsafe_fresh_plugins = []
                for plugin_id in sorted(fresh_repository_plugins):
                    evidence = request.plugin_detection_evidence.get(plugin_id, ())
                    evidence_paths = {
                        path
                        for path in request_paths
                        if any(
                            item == f"file:{path}"
                            or item.endswith(f":{path}")
                            or f":{path}:" in item
                            for item in evidence
                        )
                    }
                    if (
                        not evidence_paths
                        or not evidence_paths.issubset(added_paths)
                    ):
                        unsafe_fresh_plugins.append(plugin_id)
                if unsafe_fresh_plugins:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"target branch '{target_branch}' is indexed without "
                            "repository-analysis plugins "
                            f"({', '.join(unsafe_fresh_plugins)}) and the PR does "
                            "not prove they are introduced only by added files; "
                            f"reindex target branch '{target_branch}' before review"
                        ),
                    )
            available_snapshot_plugins = {
                snapshot.plugin_id for snapshot in snapshots
            }
            missing_snapshot_plugins = sorted(
                required_snapshot_plugins
                - available_snapshot_plugins
                - fresh_repository_plugins
            )
            if missing_snapshot_plugins:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"target branch '{target_branch}' is missing repository-analysis "
                        f"snapshots for {', '.join(missing_snapshot_plugins)}; reindex "
                        f"target branch '{target_branch}' before review"
                    ),
                )

        request_partial_files = tuple(sorted(
            file_info.path
            for file_info in request.files
            if (
                file_info.change_type != "DELETED"
                and _content_state(file_info) != "complete"
            )
        ))
        max_file_size_bytes = _configured_max_file_size_bytes(index_manager)
        oversized_overlay_file_sizes = {}
        for file_info in request.files:
            if (
                file_info.change_type == "DELETED"
                or _content_state(file_info) != "complete"
            ):
                continue
            content_size_bytes = _utf8_size_bytes(file_info.content)
            if content_size_bytes > max_file_size_bytes:
                oversized_overlay_file_sizes[file_info.path] = (
                    content_size_bytes
                )
        oversized_overlay_files = tuple(sorted(oversized_overlay_file_sizes))
        oversized_overlay_paths = set(oversized_overlay_files)
        fallback_overlay_files = tuple(sorted({
            *request_partial_files,
            *oversized_overlay_files,
        }))
        for oversized_path in oversized_overlay_files:
            logger.warning(
                "PR repository source exceeds the configured indexing ceiling; "
                "skipping it without truncation: code=%s pr=%s path=%s "
                "bytes=%d max_bytes=%d",
                REPOSITORY_FILE_SIZE_LIMIT_CODE,
                request.pr_number,
                oversized_path,
                oversized_overlay_file_sizes[oversized_path],
                max_file_size_bytes,
            )
        changed_paths = tuple(
            file_info.path for file_info in request.files
        )
        (
            target_architecture_payloads,
            target_architecture_diagnostics,
        ) = _target_architecture_payloads(
            index_manager,
            collection_name,
            workspace=request.workspace,
            project=request.project,
            branch=target_branch,
            revision=request.base_revision,
            changed_paths=fallback_overlay_files,
        )
        index_diagnostics = list(target_architecture_diagnostics)
        if request_partial_files:
            index_diagnostics.append({
                "code": "partial_diff_source_omitted",
                "message": (
                    "Partial diff content was not treated as complete source; "
                    "exact target-branch architecture facts were used only "
                    "as fallback for paths without complete source."
                ),
                "paths": list(request_partial_files),
            })
        if oversized_overlay_files:
            index_diagnostics.append({
                "code": REPOSITORY_FILE_SIZE_LIMIT_CODE,
                "message": (
                    "Complete PR source above the configured repository-index "
                    "file ceiling was omitted as a whole without truncation. "
                    "The PR diff remains review evidence."
                ),
                "paths": list(oversized_overlay_files),
                "file_sizes_bytes": {
                    path: oversized_overlay_file_sizes[path]
                    for path in oversized_overlay_files
                },
                "max_file_size_bytes": max_file_size_bytes,
            })
        generation_fingerprint = pr_overlay_generation_fingerprint(
            workspace=request.workspace,
            project=request.project,
            pr_number=request.pr_number,
            branch=request.branch,
            base_branch=target_branch,
            source_revision=request.source_revision,
            base_revision=request.base_revision,
            base_generation_manifest_sha256=base_receipt[
                "generation_manifest_sha256"
            ],
            files=request.files,
            requested_plugin_ids=requested_plugin_ids,
            repository_plugin_ids=repository_plugins,
            request_plugin_fingerprint=request.plugin_fingerprint,
            target_plugin_fingerprint=stored_fingerprint,
            capability_fingerprint=(
                capabilities.fingerprint
                if capabilities is not None
                else ZERO_FINGERPRINT
            ),
            descriptor_fingerprint=(
                capabilities.descriptor_fingerprint
                if capabilities is not None
                else ZERO_FINGERPRINT
            ),
            implementation_fingerprint=implementation_fingerprint,
            index_representation_fingerprint=representation_fingerprint,
            pr_overlay_representation_fingerprint=(
                overlay_representation_fingerprint
            ),
            snapshots=snapshots,
        )
        reusable_receipt = read_pr_overlay_generation(
            index_manager.qdrant_client,
            collection_name,
            workspace=request.workspace,
            project=request.project,
            pr_number=request.pr_number,
            branch=request.branch,
            base_branch=target_branch,
            source_revision=request.source_revision,
            base_revision=request.base_revision,
            base_generation_manifest_sha256=base_receipt[
                "generation_manifest_sha256"
            ],
            generation_fingerprint=generation_fingerprint,
            overlay_representation_fingerprint=(
                overlay_representation_fingerprint
            ),
        )
        if reusable_receipt is not None:
            architecture_points = sum(
                1
                for point in old_pr_points
                if (
                    (point.payload or {}).get("architecture_context")
                    or (point.payload or {}).get("architecture_source")
                )
            )
            logger.info(
                "Reused PR #%s overlay generation: %s points from %s changed files",
                request.pr_number,
                len(old_pr_points),
                len(request.files),
            )
            return {
                "status": "reused",
                **base_generation_receipt,
                "pr_number": request.pr_number,
                "files_processed": len(request.files),
                "chunks_indexed": len(old_pr_points),
                "chunks_failed": 0,
                "architecture_packets_indexed": architecture_points,
                "generation_fingerprint": generation_fingerprint,
                **reusable_receipt,
                "overlay_representation_fingerprint": (
                    overlay_representation_fingerprint
                ),
                "partial_files": list(request_partial_files),
                "skipped_files": list(oversized_overlay_files),
                "diagnostics": index_diagnostics,
                "effective_project_capabilities": _capabilities_payload(
                    capabilities,
                    implementation_fingerprint,
                ),
                "review_groups": _review_groups(
                    changed_paths,
                    tuple(
                        point.payload or {}
                        for point in old_pr_points
                        if (point.payload or {}).get("architecture_context")
                    ),
                    target_architecture_payloads,
                    fallback_overlay_files,
                ),
            }

        file_dispositions = {}
        active_overlay_files = [
            file_info
            for file_info in request.files
            if file_info.path not in oversized_overlay_paths
        ]
        if capabilities is not None:
            from codecrow_plugins import FileDisposition

            file_dispositions = {
                file_info.path: index_manager.plugin_runtime.file_disposition(
                    file_info.path,
                    capabilities,
                )
                for file_info in request.files
            }
            active_overlay_files = [
                file_info
                for file_info in active_overlay_files
                if file_dispositions[file_info.path] not in {
                    FileDisposition.EXCLUDED,
                    FileDisposition.GENERATED,
                }
            ]

        partial_overlay_files = tuple(sorted(
            file_info.path
            for file_info in active_overlay_files
            if (
                file_info.change_type != "DELETED"
                and _content_state(file_info) != "complete"
            )
        ))

        # Only complete post-change source can become an indexed source record.
        # Partial diffs remain review evidence in the inference request and are
        # represented here only by their changed-file identity and the exact
        # target-branch architecture facts loaded above.
        documents = []
        for file_info in active_overlay_files:
            if not file_info.content or not file_info.content.strip():
                continue
            if file_info.change_type == "DELETED":
                continue
            if _content_state(file_info) != "complete":
                continue
            if capabilities is not None:
                disposition = file_dispositions[file_info.path]
                if disposition is not FileDisposition.FULL:
                    continue

            doc = Document(
                text=file_info.content,
                metadata={
                    "path": file_info.path,
                    "change_type": file_info.change_type,
                    "content_state": "complete",
                }
            )
            documents.append(doc)

        if documents:
            chunks, split_skipped_paths = (
                index_manager.splitter.split_documents_resilient(
                    documents,
                    capabilities=capabilities,
                )
            )
        else:
            chunks = []
            split_skipped_paths = ()
        split_skipped_paths = tuple(sorted({
            *split_skipped_paths,
            *oversized_overlay_files,
        }))

        # Add PR metadata to all chunks
        for chunk in chunks:
            chunk.metadata["content_state"] = "complete"
            chunk.metadata[INDEX_REPRESENTATION_PAYLOAD_KEY] = (
                representation_fingerprint
            )
            chunk.metadata[PR_OVERLAY_REPRESENTATION_PAYLOAD_KEY] = (
                overlay_representation_fingerprint
            )
            if capabilities is not None:
                chunk.metadata["plugin_ids"] = list(
                    capabilities.repository_plugins
                )
                chunk.metadata["plugin_fingerprint"] = capabilities.fingerprint
                chunk.metadata["plugin_descriptor_fingerprint"] = (
                    capabilities.descriptor_fingerprint
                )
                chunk.metadata["plugin_implementation_fingerprint"] = (
                    implementation_fingerprint
                )
            chunk.metadata["pr"] = True
            chunk.metadata["pr_number"] = request.pr_number
            chunk.metadata["pr_branch"] = request.branch
            chunk.metadata["workspace"] = request.workspace
            chunk.metadata["project"] = request.project
            chunk.metadata["branch"] = request.branch
            chunk.metadata["pr_generation_fingerprint"] = generation_fingerprint
            chunk.metadata["pr_source_revision"] = request.source_revision
            chunk.metadata["pr_base_revision"] = request.base_revision
            chunk.metadata["pr_base_generation_manifest_sha256"] = (
                base_receipt["generation_manifest_sha256"]
            )
            chunk.metadata["pr_overlay_base_branch"] = target_branch
            chunk.metadata["indexed_at"] = datetime.now(timezone.utc).isoformat()

        point_id_branch = f"__pr__/{request.pr_number}/{request.branch}"
        analysis_revision = request.source_revision or f"pr-{request.pr_number}"

        architecture_nodes = []
        symbol_nodes = []
        overlay_artifact_files = tuple(
            file_info
            for file_info in active_overlay_files
            if (
                file_info.change_type == "DELETED"
                or _content_state(file_info) == "complete"
            )
        )
        if (
            capabilities is not None
            and overlay_artifact_files
            and (snapshots or fresh_repository_plugins)
        ):
            from codecrow_plugins import (
                FileArtifact,
                RepositoryAnalysis,
                RepositoryAnalysisMode,
            )

            handle = index_manager.plugin_runtime.start_repository_analysis(
                capabilities,
                analysis_revision,
                snapshots=snapshots,
                mode=RepositoryAnalysisMode.PR_OVERLAY,
                source_root=(
                    stored_repository_facts.source_root
                    if stored_repository_facts is not None
                    else None
                ),
            )
            artifacts = tuple(sorted(
                (
                    FileArtifact(
                        path=file_info.path,
                        content=(
                            "" if file_info.change_type == "DELETED"
                            else file_info.content
                        ),
                        deleted=file_info.change_type == "DELETED",
                    )
                    for file_info in overlay_artifact_files
                ),
                key=lambda artifact: artifact.path,
            ))
            handle.ingest(artifacts)
            analysis, diagnostics = handle.finish()
            repository_skipped_paths = (
                index_manager._indexer
                .accept_recoverable_repository_diagnostics(
                    diagnostics,
                    "PR repository overlay",
                )
            )
            split_skipped_paths = tuple(sorted({
                *split_skipped_paths,
                *repository_skipped_paths,
            }))

            overlay_artifact_paths = {
                file_info.path for file_info in overlay_artifact_files
            }
            affected_packets = tuple(
                packet for packet in analysis.packets
                if overlay_artifact_paths.intersection(packet.paths)
            )
            affected_related_paths = {
                path for packet in affected_packets for path in packet.paths
            }
            affected_analysis = RepositoryAnalysis(
                symbols=tuple(
                    symbol for symbol in analysis.symbols
                    if symbol.path in {
                        *overlay_artifact_paths,
                        *affected_related_paths,
                    }
                ),
                packets=affected_packets,
                contexts=tuple(
                    context for context in analysis.contexts
                    if context.path in affected_related_paths
                ),
            )
            architecture_nodes = index_manager._indexer._architecture_nodes(
                affected_analysis,
                capabilities,
                request.workspace,
                request.project,
                request.branch,
                analysis_revision,
                implementation_fingerprint,
                representation_fingerprint,
            )
            architecture_nodes.extend(
                index_manager._indexer._repository_context_nodes(
                    affected_analysis,
                    capabilities,
                    request.workspace,
                    request.project,
                    request.branch,
                    analysis_revision,
                    implementation_fingerprint,
                    representation_fingerprint,
                )
            )
            symbol_nodes = index_manager._indexer._symbol_nodes(
                affected_analysis,
                capabilities,
                request.workspace,
                request.project,
                request.branch,
                analysis_revision,
                implementation_fingerprint,
                representation_fingerprint,
            )
            for node in (*architecture_nodes, *symbol_nodes):
                node.metadata["pr"] = True
                node.metadata["pr_number"] = request.pr_number
                node.metadata["pr_branch"] = request.branch
                node.metadata[PR_OVERLAY_REPRESENTATION_PAYLOAD_KEY] = (
                    overlay_representation_fingerprint
                )
                node.metadata["pr_generation_fingerprint"] = generation_fingerprint
                node.metadata["pr_source_revision"] = request.source_revision
                node.metadata["pr_base_revision"] = request.base_revision
                node.metadata["pr_base_generation_manifest_sha256"] = (
                    base_receipt["generation_manifest_sha256"]
                )
                node.metadata["pr_overlay_base_branch"] = target_branch
                node.metadata["indexed_at"] = datetime.now(timezone.utc).isoformat()
        identity_metadata = {
            "plugin_ids": list(
                capabilities.repository_plugins
                if capabilities is not None else stored_plugin_ids
            ),
            "plugin_fingerprint": (
                capabilities.fingerprint
                if capabilities is not None else stored_fingerprint
            ),
            "plugin_descriptor_fingerprint": (
                capabilities.descriptor_fingerprint
                if capabilities is not None
                else _stored_descriptor_fingerprint
            ),
            "plugin_implementation_fingerprint": implementation_fingerprint,
            INDEX_REPRESENTATION_PAYLOAD_KEY: representation_fingerprint,
            PR_OVERLAY_REPRESENTATION_PAYLOAD_KEY: (
                overlay_representation_fingerprint
            ),
        }

        successful, overlay_receipt = (
            index_manager._pr_overlay_ops.replace_pr_overlay_generation(
                [*chunks, *architecture_nodes, *symbol_nodes],
                old_pr_points,
                collection_name,
                request.workspace,
                request.project,
                point_id_branch,
                mutation_lease.assert_owned,
                pr_number=request.pr_number,
                branch=request.branch,
                base_branch=target_branch,
                source_revision=request.source_revision,
                base_revision=request.base_revision,
                base_generation_manifest_sha256=base_receipt[
                    "generation_manifest_sha256"
                ],
                generation_fingerprint=generation_fingerprint,
                overlay_representation_fingerprint=(
                    overlay_representation_fingerprint
                ),
                identity_metadata=identity_metadata,
            )
        )
        skipped_points = (
            len(chunks) + len(architecture_nodes) + len(symbol_nodes) - successful
        )

        logger.info(
            "Indexed PR #%s: %s structural points from %s changed files "
            "(%s architecture packets, %s symbols, %s partial files, "
            "%s oversized files)",
            request.pr_number,
            successful,
            len(request.files),
            len(architecture_nodes),
            len(symbol_nodes),
            len(partial_overlay_files),
            len(oversized_overlay_files),
        )

        return {
            "status": "indexed",
            **base_generation_receipt,
            "pr_number": request.pr_number,
            "files_processed": len(request.files),
            "chunks_indexed": successful,
            "chunks_failed": 0,
            "chunks_skipped": skipped_points,
            "skipped_files": list(split_skipped_paths),
            "architecture_packets_indexed": len(architecture_nodes),
            "symbols_indexed": len(symbol_nodes),
            "generation_fingerprint": generation_fingerprint,
            **overlay_receipt,
            "overlay_representation_fingerprint": (
                overlay_representation_fingerprint
            ),
            "partial_files": list(request_partial_files),
            "diagnostics": index_diagnostics,
            "effective_project_capabilities": _capabilities_payload(
                capabilities,
                implementation_fingerprint,
            ),
            "review_groups": _review_groups(
                changed_paths,
                tuple(
                    node.metadata
                    for node in architecture_nodes
                    if node.metadata.get("architecture_context")
                ),
                target_architecture_payloads,
                fallback_overlay_files,
            ),
        }

    except HTTPException:
        raise
    except ExactIndexPreconditionError as e:
        logger.info("Rejected PR indexing against invalid repository state: %s", e)
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        logger.info(f"Invalid request for PR indexing: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except MutationLeaseUnavailable as e:
        raise HTTPException(status_code=409, detail=str(e))
    except MutationCoordinationUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Internal error indexing PR files: {e}")
        raise HTTPException(status_code=500, detail="Internal indexing error")
    finally:
        if mutation_lease is not None:
            mutation_context.__exit__(None, None, None)


@router.delete("/index/pr-files/{workspace}/{project}/{pr_number}")
def delete_pr_files(
    workspace: str,
    project: str,
    pr_number: int,
    collection_target: str = Query(min_length=1),
):
    """Delete all indexed points for a specific PR."""
    index_manager = _get_index_manager()
    try:
        with index_manager.pr_overlay_mutation(
            workspace,
            project,
            pr_number,
            "delete-pr-overlay",
        ) as lease:
            physical_collection = (
                index_manager._collection_manager.resolve_collection_target(
                    collection_target
                )
            )
            if physical_collection is None:
                return {"status": "skipped", "message": "Collection does not exist"}
            physical_collection = (
                index_manager._collection_manager.require_structural_collection(
                    physical_collection
                )
            )

            # Existing exact generations may predate the PR filter indexes.
            # Repair them before the acknowledged filter delete. Keeping
            # wait=True is correctness-critical: releasing the same-PR lease
            # before Qdrant applies the delete could erase a subsequent rerun.
            index_manager._collection_manager.ensure_payload_indexes(
                physical_collection
            )

            lease.assert_owned()
            index_manager.qdrant_client.delete(
                collection_name=physical_collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="workspace", match=MatchValue(value=workspace)),
                        FieldCondition(key="project", match=MatchValue(value=project)),
                        FieldCondition(key="pr", match=MatchValue(value=True)),
                        FieldCondition(key="pr_number", match=MatchValue(value=pr_number)),
                    ]
                ),
                wait=True,
            )

            logger.info(
                "Deleted PR #%s points from %s",
                pr_number,
                physical_collection,
            )

            return {
                "status": "deleted",
                "pr_number": pr_number,
                "collection": physical_collection
            }

    except MutationLeaseUnavailable as e:
        raise HTTPException(status_code=409, detail=str(e))
    except MutationCoordinationUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.debug("PR file deletion failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
