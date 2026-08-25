from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from rag_pipeline.api.models import CodeSearchRequest, DeterministicContextRequest
from rag_pipeline.api.routers.query import code_search, get_deterministic_context
from rag_pipeline.core.exact_index import ExactIndexPreconditionError
from rag_pipeline.services.deterministic_context import DeterministicContextMixin


BASE_REVISION = "a" * 40
SOURCE_REVISION = "b" * 40
BASE_GENERATION = "c" * 64
PR_GENERATION = "sha256:" + "d" * 64
PR_OVERLAY_MANIFEST = "e" * 64
OVERLAY_REPRESENTATION = "sha256:" + "f" * 64


def _overlay_receipt(manifest=PR_OVERLAY_MANIFEST):
    return {
        "overlay_generation_member_count": 1,
        "overlay_generation_members_sha256": "1" * 64,
        "overlay_generation_manifest_sha256": manifest,
    }


def _manager():
    manager = MagicMock()
    manager._get_project_collection_name.return_value = "code_index"
    manager._collection_manager.resolve_collection_target.return_value = (
        "code_index_generation"
    )
    manager._collection_manager.require_structural_collection.return_value = (
        "code_index_generation"
    )
    manager.pr_overlay_representation_fingerprint = OVERLAY_REPRESENTATION
    return manager


def test_code_search_detects_generation_swap_during_request():
    manager = _manager()
    manager.get_revision_preflight.side_effect = [
        {"generation_manifest_sha256": BASE_GENERATION},
        {"generation_manifest_sha256": "f" * 64},
    ]
    service = MagicMock()
    service.search_code.return_value = []
    request = CodeSearchRequest(
        query="dependency lookup",
        workspace="ws",
        project="project",
        branch="main",
        repository_revision=BASE_REVISION,
        repository_generation_manifest_sha256=BASE_GENERATION,
        collection_target="generation-target",
    )

    with patch(
        "rag_pipeline.api.routers.query._get_singletons",
        return_value=(manager, service),
    ):
        with pytest.raises(HTTPException) as exception:
            code_search(request)

    assert exception.value.status_code == 409
    assert "generation changed" in exception.value.detail
    service.search_code.assert_called_once()


def test_deterministic_context_detects_overlay_generation_swap():
    manager = _manager()
    manager.get_revision_preflight.return_value = {
        "generation_manifest_sha256": BASE_GENERATION,
    }
    service = MagicMock()
    service.get_deterministic_context.return_value = {"chunks": []}
    request = DeterministicContextRequest(
        workspace="ws",
        project="project",
        branches=["main"],
        file_paths=["src/Foo.php"],
        pr_number=42,
        source_revision=SOURCE_REVISION,
        base_revision=BASE_REVISION,
        base_generation_manifest_sha256=BASE_GENERATION,
        pr_generation_fingerprint=PR_GENERATION,
        pr_overlay_generation_manifest_sha256=PR_OVERLAY_MANIFEST,
        collection_target="generation-target",
    )

    with (
        patch(
            "rag_pipeline.api.routers.query._get_singletons",
            return_value=(manager, service),
        ),
        patch(
            "rag_pipeline.api.routers.query.read_pr_overlay_generation",
            side_effect=[
                _overlay_receipt(),
                _overlay_receipt("2" * 64),
            ],
        ),
    ):
        with pytest.raises(HTTPException) as exception:
            get_deterministic_context(request)

    assert exception.value.status_code == 409
    assert "overlay generation changed" in exception.value.detail


def test_revision_bound_deterministic_context_rejects_extra_branch():
    with pytest.raises(ValueError, match="at most 1 item"):
        DeterministicContextRequest(
            workspace="ws",
            project="project",
            branches=["main", "feature"],
            file_paths=["src/Foo.php"],
            source_revision=SOURCE_REVISION,
            base_revision=BASE_REVISION,
            base_generation_manifest_sha256=BASE_GENERATION,
            collection_target="generation-target",
        )


def test_deterministic_context_rejects_partial_overlay_identity():
    manager = _manager()
    service = MagicMock()
    request = DeterministicContextRequest(
        workspace="ws",
        project="project",
        branches=["main"],
        file_paths=["src/Foo.php"],
        pr_number=42,
        base_revision=BASE_REVISION,
        base_generation_manifest_sha256=BASE_GENERATION,
        collection_target="generation-target",
        pr_generation_fingerprint=PR_GENERATION,
    )

    with patch(
        "rag_pipeline.api.routers.query._get_singletons",
        return_value=(manager, service),
    ):
        with pytest.raises(HTTPException) as exception:
            get_deterministic_context(request)

    assert exception.value.status_code == 409
    service.get_deterministic_context.assert_not_called()


def test_deterministic_service_rejects_fingerprint_without_revisions():
    with pytest.raises(
        ExactIndexPreconditionError,
        match="complete source/base generation identity",
    ):
        DeterministicContextMixin.get_deterministic_context(
            MagicMock(),
            workspace="ws",
            project="project",
            branches=["main"],
            file_paths=["src/Foo.php"],
            collection_target="generation-target",
            pr_number=42,
            pr_generation_fingerprint=PR_GENERATION,
        )
