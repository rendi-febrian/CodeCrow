"""Revision-bound structural query endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from ..models import CodeSearchRequest, DeterministicContextRequest
from ...core.pr_overlay_manifest import read_pr_overlay_generation
from ...core.exact_index import ExactIndexPreconditionError
from ...core.revision_binding import (
    require_repository_generation,
    require_same_repository_generation,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


def _get_singletons():
    from ..api import index_manager, query_service
    return index_manager, query_service


def _optional_string(value) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _require_complete_pr_overlay_binding(
    *,
    pr_number: Optional[int],
    target_branch: Optional[str],
    source_revision: Optional[str],
    base_revision: Optional[str],
    base_generation_manifest: Optional[str],
    pr_generation_fingerprint: Optional[str],
    pr_overlay_manifest: Optional[str],
) -> bool:
    overlay_binding_requested = bool(
        pr_generation_fingerprint or pr_overlay_manifest
    )
    if not overlay_binding_requested:
        return False
    if not all((
        pr_number,
        target_branch,
        source_revision,
        base_revision,
        base_generation_manifest,
        pr_generation_fingerprint,
        pr_overlay_manifest,
    )):
        raise ExactIndexPreconditionError(
            "revision-bound PR overlay requires PR number, one authoritative "
            "branch, source/base revisions, and both generation receipts"
        )
    return True


@router.post("/query/code-search")
def code_search(request: CodeSearchRequest):
    """Search exact indexed tokens in one immutable repository generation."""
    index_manager, query_service = _get_singletons()
    try:
        receipt = require_repository_generation(
            index_manager=index_manager,
            workspace=request.workspace,
            project=request.project,
            branch=request.branch,
            revision=request.repository_revision,
            generation_manifest_sha256=(
                request.repository_generation_manifest_sha256
            ),
            collection_target=request.collection_target,
        )
        search_result = query_service.search_code(
            query=request.query,
            workspace=request.workspace,
            project=request.project,
            branch=request.branch,
            repository_revision=request.repository_revision,
            collection_target=receipt["_collection_target"],
            limit=request.limit,
        )
        require_same_repository_generation(
            index_manager,
            workspace=request.workspace,
            project=request.project,
            branch=request.branch,
            revision=request.repository_revision,
            receipt=receipt,
        )
        if isinstance(search_result, dict):
            return search_result
        # Compatibility for injected/custom query services during rollout.
        return {
            "results": search_result,
            "coverage": {"complete": False, "partial_reasons": ["unknown"]},
        }
    except ExactIndexPreconditionError as exception:
        raise HTTPException(status_code=409, detail=str(exception))
    except Exception as exception:
        logger.error("Structural code search failed: %s", exception)
        raise HTTPException(status_code=500, detail=str(exception))


@router.post("/query/deterministic")
def get_deterministic_context(request: DeterministicContextRequest):
    """Retrieve revision-bound source and relations by exact metadata."""
    index_manager, query_service = _get_singletons()
    try:
        target_branch = request.branches[0] if request.branches else None
        source_revision = _optional_string(request.source_revision)
        base_revision = _optional_string(request.base_revision)
        base_generation_manifest = _optional_string(
            request.base_generation_manifest_sha256
        )
        pr_generation_fingerprint = _optional_string(
            request.pr_generation_fingerprint
        )
        pr_overlay_manifest = _optional_string(
            request.pr_overlay_generation_manifest_sha256
        )
        collection_target = _optional_string(request.collection_target)
        receipt = None
        overlay_receipt = None
        exact_overlay_binding = _require_complete_pr_overlay_binding(
            pr_number=request.pr_number,
            target_branch=target_branch,
            source_revision=source_revision,
            base_revision=base_revision,
            base_generation_manifest=base_generation_manifest,
            pr_generation_fingerprint=pr_generation_fingerprint,
            pr_overlay_manifest=pr_overlay_manifest,
        )
        receipt = require_repository_generation(
            index_manager,
            workspace=request.workspace,
            project=request.project,
            branch=target_branch,
            revision=base_revision,
            generation_manifest_sha256=base_generation_manifest,
            collection_target=collection_target,
        )
        if exact_overlay_binding:
            overlay_receipt = read_pr_overlay_generation(
                index_manager.qdrant_client,
                receipt["_collection_target"],
                workspace=request.workspace,
                project=request.project,
                pr_number=request.pr_number,
                branch=target_branch,
                base_branch=target_branch,
                source_revision=source_revision,
                base_revision=base_revision,
                base_generation_manifest_sha256=base_generation_manifest,
                generation_fingerprint=pr_generation_fingerprint,
                overlay_representation_fingerprint=(
                    index_manager.pr_overlay_representation_fingerprint
                ),
                expected_manifest_sha256=pr_overlay_manifest,
            )
            if overlay_receipt is None:
                raise ExactIndexPreconditionError(
                    "requested PR overlay generation is unavailable"
                )
        context = query_service.get_deterministic_context(
            workspace=request.workspace,
            project=request.project,
            branches=request.branches,
            file_paths=request.file_paths,
            limit_per_file=request.limit_per_file,
            pr_number=request.pr_number,
            pr_changed_files=request.pr_changed_files,
            additional_identifiers=request.additional_identifiers,
            expected_revisions=(
                {target_branch: base_revision}
                if base_revision and target_branch else None
            ),
            pr_source_revision=source_revision,
            pr_base_revision=base_revision,
            pr_base_generation_manifest_sha256=base_generation_manifest,
            pr_generation_fingerprint=pr_generation_fingerprint,
            collection_target=receipt["_collection_target"],
        )
        require_same_repository_generation(
            index_manager,
            workspace=request.workspace,
            project=request.project,
            branch=target_branch,
            revision=base_revision,
            receipt=receipt,
        )
        if overlay_receipt:
            second_overlay = read_pr_overlay_generation(
                index_manager.qdrant_client,
                receipt["_collection_target"],
                workspace=request.workspace,
                project=request.project,
                pr_number=request.pr_number,
                branch=target_branch,
                base_branch=target_branch,
                source_revision=source_revision,
                base_revision=base_revision,
                base_generation_manifest_sha256=base_generation_manifest,
                generation_fingerprint=pr_generation_fingerprint,
                overlay_representation_fingerprint=(
                    index_manager.pr_overlay_representation_fingerprint
                ),
            )
            if second_overlay != overlay_receipt:
                raise ExactIndexPreconditionError(
                    "PR overlay generation changed while context was retrieved"
                )
        return {"context": context}
    except ExactIndexPreconditionError as exception:
        raise HTTPException(status_code=409, detail=str(exception))
    except Exception as exception:
        logger.error("Deterministic context retrieval failed: %s", exception)
        raise HTTPException(status_code=500, detail=str(exception))
