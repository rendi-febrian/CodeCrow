"""Revision-bound structural retrieval for Stage 1 review batches.

This module owns transport, retrieval-state admission, and PR freshness handling.
The structural selector is injected by the Stage 1 prompt module so retrieval
does not depend on prompt rendering or language/framework implementations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from model.dtos import ReviewRequestDto
from utils.path_identity import (
    normalize_repository_path,
    repository_paths_match,
)


logger = logging.getLogger(__name__)


@dataclass
class Stage1RagState:
    """Per-review exact-retrieval state shared across Stage 1 batches."""

    context_disabled: bool = False
    context_disable_reason: str = ""
    exact_evidence_by_id: Dict[str, tuple[Dict[str, Any], ...]] = field(
        default_factory=dict
    )
    deterministic_retrieval_states: List[str] = field(default_factory=list)


def unwrap_rag_context(
    response: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    context = response.get("context")
    if isinstance(context, dict):
        return context
    return response


def rag_response_error(
    response: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Return the sanitized failure carried by a RAG client response, if any."""

    if not isinstance(response, dict):
        return None
    if str(response.get("status", "")).strip().casefold() != "error":
        return None
    detail = str(response.get("error") or "RAG request failed").strip()
    return detail or "RAG request failed"


def deterministic_retrieval_state(
    deterministic_response: Optional[Dict[str, Any]],
) -> str:
    context = unwrap_rag_context(deterministic_response)
    metadata = context.get("_metadata") if isinstance(context, dict) else None
    if isinstance(metadata, dict) and metadata.get("retrieval_state"):
        return str(metadata["retrieval_state"]).strip().casefold()
    return "unknown"


def capture_deterministic_retrieval_state(
    deterministic_response: Optional[Dict[str, Any]],
    rag_state: Optional[Stage1RagState],
) -> None:
    """Retain exact-retrieval state for review diagnostics and evidence gates."""

    if rag_state is None:
        return
    if rag_response_error(deterministic_response):
        rag_state.deterministic_retrieval_states.append("failed")
        return
    rag_state.deterministic_retrieval_states.append(
        deterministic_retrieval_state(deterministic_response)
    )


def is_exact_revision_bound(
    request: ReviewRequestDto,
    pr_indexed: bool,
) -> bool:
    pr_number = getattr(request, "pullRequestId", None)
    return bool(
        pr_indexed
        and isinstance(pr_number, int)
        and pr_number > 0
        and all(
            isinstance(value, str) and bool(value.strip())
            for value in (
                getattr(request, "currentCommitHash", None)
                or getattr(request, "commitHash", None),
                request.get_target_head_commit_hash(),
                getattr(request, "ragCollectionTarget", None),
                getattr(request, "ragBaseGenerationManifestSha256", None),
                getattr(request, "ragPrGenerationFingerprint", None),
                getattr(
                    request,
                    "ragPrOverlayGenerationManifestSha256",
                    None,
                ),
            )
        )
    )


def has_exact_base_binding(request: ReviewRequestDto) -> bool:
    return all(
        isinstance(value, str) and bool(value.strip())
        for value in (
            request.get_target_head_commit_hash(),
            getattr(request, "ragCollectionTarget", None),
            getattr(request, "ragBaseGenerationManifestSha256", None),
        )
    )


def disable_rag_context(
    rag_state: Optional[Stage1RagState],
    reason: str,
) -> bool:
    """Open the optional-context circuit once and report whether it changed."""

    if rag_state is None:
        return True
    if rag_state.context_disabled:
        return False
    rag_state.context_disabled = True
    rag_state.context_disable_reason = reason
    return True


def deduplicate_pr_stale_chunks(
    chunks: List[Dict[str, Any]],
    pr_changed_files: List[str],
    batch_file_paths: List[str],
) -> List[Dict[str, Any]]:
    if not chunks or not pr_changed_files:
        return chunks

    pr_changed_set = {
        normalize_repository_path(path)
        for path in pr_changed_files
        if normalize_repository_path(path)
    }
    batch_set = {
        normalize_repository_path(path)
        for path in batch_file_paths
        if normalize_repository_path(path)
    }

    by_path: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        path = normalize_repository_path(
            metadata.get("path")
            or chunk.get("path")
            or chunk.get("file_path", "")
        )
        if not path:
            path = "__unknown__"
        by_path.setdefault(path, []).append(chunk)

    result = []
    for path, path_chunks in by_path.items():
        is_pr_file = any(
            repository_paths_match(path, changed_path)
            for changed_path in pr_changed_set
        )
        is_batch_file = any(
            repository_paths_match(path, batch_path)
            for batch_path in batch_set
        )

        if not is_pr_file or is_batch_file:
            result.extend(path_chunks)
            continue

        pr_chunks = [
            chunk
            for chunk in path_chunks
            if chunk.get("_source") == "pr_indexed"
        ]
        non_pr_chunks = [
            chunk
            for chunk in path_chunks
            if chunk.get("_source") != "pr_indexed"
        ]

        if pr_chunks and non_pr_chunks:
            result.extend(pr_chunks)
            logger.info(
                "Dedup: replaced %d stale branch chunk(s) with %d "
                "PR-indexed chunk(s) for %s",
                len(non_pr_chunks),
                len(pr_chunks),
                path,
            )
        elif pr_chunks:
            result.extend(pr_chunks)
        else:
            for chunk in non_pr_chunks:
                chunk["_potentially_stale"] = True
            result.extend(non_pr_chunks)

    return result


async def fetch_structural_context(
    rag_client: Any,
    request: ReviewRequestDto,
    batch_file_paths: List[str],
    *,
    flatten_context: Callable[
        [Optional[Dict[str, Any]], int, Optional[Sequence[str]]],
        List[Dict[str, Any]],
    ],
    max_chunks: int,
    pr_indexed: bool = False,
    enrichment_identifiers: Optional[List[str]] = None,
    rag_state: Optional[Stage1RagState] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch and admit revision-bound structural context for one batch."""

    exact_revision_bound = is_exact_revision_bound(request, pr_indexed)
    exact_base_bound = has_exact_base_binding(request)
    exact_context_bound = exact_revision_bound or exact_base_bound
    if rag_state and rag_state.context_disabled:
        logger.debug(
            "Per-batch RAG context skipped after an earlier optional-context "
            "failure: %s",
            rag_state.context_disable_reason,
        )
        return None
    if not rag_client:
        if exact_context_bound and disable_rag_context(
            rag_state,
            "revision-bound Stage 1 retrieval has no RAG client",
        ):
            logger.info(
                "Optional revision-bound RAG context is unavailable; "
                "continuing with local review evidence"
            )
        return None

    try:
        rag_branch = request.get_rag_branch()
        base_branch = request.get_rag_base_branch()
        if not rag_branch:
            message = "Missing authoritative target branch for Stage 1 RAG retrieval"
            capture_deterministic_retrieval_state(
                {"status": "error", "error": message},
                rag_state,
            )
            if disable_rag_context(rag_state, message):
                (logger.info if exact_context_bound else logger.warning)(
                    "%s; disabling optional RAG context for the remaining "
                    "Stage 1 batches",
                    message,
                )
            return None

        logger.info(
            "Fetching structural context for %d Stage 1 file(s)",
            len(batch_file_paths),
        )

        pr_number = request.pullRequestId if exact_revision_bound else None
        all_pr_files = request.changedFiles if exact_revision_bound else None
        source_revision = (
            request.currentCommitHash or request.commitHash
            if exact_revision_bound
            else None
        )
        base_revision = (
            request.get_target_head_commit_hash()
            if exact_context_bound
            else None
        )
        base_generation_receipt = (
            request.ragBaseGenerationManifestSha256
            if exact_context_bound
            else None
        )
        pr_generation_fingerprint = (
            request.ragPrGenerationFingerprint
            if exact_revision_bound
            else None
        )
        pr_overlay_generation_manifest_sha256 = (
            request.ragPrOverlayGenerationManifestSha256
            if exact_revision_bound
            else None
        )
        collection_target = (
            request.ragCollectionTarget if exact_context_bound else None
        )

        try:
            deterministic_response = await rag_client.get_deterministic_context(
                workspace=request.projectWorkspace,
                project=request.projectNamespace,
                branches=(
                    [rag_branch]
                    if exact_context_bound
                    else list(dict.fromkeys(
                        branch
                        for branch in (rag_branch, base_branch)
                        if branch
                    ))
                ),
                file_paths=batch_file_paths,
                pr_number=pr_number,
                pr_changed_files=all_pr_files,
                additional_identifiers=enrichment_identifiers,
                source_revision=source_revision,
                base_revision=base_revision,
                base_generation_manifest_sha256=base_generation_receipt,
                pr_generation_fingerprint=pr_generation_fingerprint,
                pr_overlay_generation_manifest_sha256=(
                    pr_overlay_generation_manifest_sha256
                ),
                collection_target=collection_target,
            )
        except Exception as error:
            deterministic_response = {
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }

        deterministic_error = rag_response_error(deterministic_response)
        deterministic_chunks = flatten_context(
            deterministic_response,
            max_chunks,
            batch_file_paths,
        )
        capture_deterministic_retrieval_state(
            deterministic_response,
            rag_state,
        )
        retrieval_state = deterministic_retrieval_state(deterministic_response)
        deterministic_context = unwrap_rag_context(deterministic_response)
        deterministic_metadata = deterministic_context.get("_metadata")
        if not isinstance(deterministic_metadata, dict):
            deterministic_metadata = {}

        explicitly_unusable = (
            deterministic_metadata.get("context_usable") is False
        )
        terminal_retrieval_state = retrieval_state in {
            "error",
            "failed",
            "unavailable",
        }
        context_error = (
            deterministic_error
            or (
                f"deterministic retrieval state is {retrieval_state}"
                if terminal_retrieval_state
                else None
            )
            or (
                "deterministic retrieval marked returned context unusable"
                if deterministic_chunks and explicitly_unusable
                else None
            )
        )
        if context_error:
            if disable_rag_context(rag_state, context_error):
                (logger.info if exact_context_bound else logger.warning)(
                    "Optional %sstructural context is unavailable; disabling "
                    "it for the remaining Stage 1 batches and continuing with "
                    "local review evidence: %s",
                    "revision-bound " if exact_context_bound else "",
                    context_error,
                )
            return None
        if not deterministic_chunks:
            return None

        # Completeness qualifies coverage; it does not erase exact evidence.
        # A safety-cap partial result remains useful for positive claims, while
        # absence from that result is explicitly not negative evidence.
        context = {
            "relevant_code": deterministic_chunks,
            "_metadata": {
                key: deterministic_metadata[key]
                for key in (
                    "retrieval_state",
                    "retrieval_scope",
                    "coverage_state",
                    "context_usable",
                    "bounded_files",
                    "partial_reasons",
                )
                if key in deterministic_metadata
            },
        }
        if pr_indexed and all_pr_files:
            context["relevant_code"] = deduplicate_pr_stale_chunks(
                context["relevant_code"],
                pr_changed_files=all_pr_files,
                batch_file_paths=batch_file_paths,
            )
        logger.info(
            "Structural context included %d exact chunk(s) for %s "
            "(retrieval_state=%s, coverage_state=%s)",
            len(context["relevant_code"]),
            batch_file_paths,
            retrieval_state,
            deterministic_metadata.get("coverage_state") or "unknown",
        )
        return context if context["relevant_code"] else None

    except Exception as error:
        if disable_rag_context(rag_state, str(error)):
            logger.warning(
                "Failed to fetch optional structural context; disabling it "
                "for the remaining Stage 1 batches: %s",
                error,
            )
        return None
