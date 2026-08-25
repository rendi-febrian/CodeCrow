"""Focused tests for the exact-generation repository index API."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

from rag_pipeline.api.models import IndexRequest
from rag_pipeline.api.routers.index import (
    _index_collection_target,
    delete_branch,
    get_limits,
    get_revision_preflight,
    index_repository,
)
from rag_pipeline.core.exact_index import ExactIndexPreconditionError
from rag_pipeline.core.source_tree import RepositorySourceTreeError
from rag_pipeline.models.config import IndexStats


def _stats() -> IndexStats:
    return IndexStats(
        namespace="ws__project__main",
        document_count=2,
        chunk_count=4,
        last_updated="now",
        workspace="ws",
        project="project",
        branch="main",
        generation_manifest_sha256="b" * 64,
        source_tree_sha256="a" * 64,
        collection_target="generation-target",
    )


def _preflight_receipt():
    return {
        "workspace": "ws",
        "project": "project",
        "branch": "main",
        "commit": "commit",
        "point_count": 5,
        "repository_revision": "commit",
        "repository_facts_sha256": "1" * 64,
        "plugin_ids": ["python"],
        "plugin_fingerprint": "sha256:selection",
        "plugin_descriptor_fingerprint": "sha256:descriptor",
        "plugin_implementation_fingerprint": "sha256:implementation",
        "index_representation_fingerprint": "sha256:representation",
        "current_index_representation_fingerprint": "sha256:representation",
        "generation_schema": "codecrow.repository-generation.v2",
        "generation_member_count": 4,
        "generation_members_sha256": "2" * 64,
        "generation_manifest_sha256": "3" * 64,
        "source_tree_sha256": "4" * 64,
        "index_include_patterns": ["src/**"],
        "index_exclude_patterns": ["vendor/**"],
        "index_selection_policy_sha256": "5" * 64,
    }


def test_limits_are_reported_from_config():
    config = MagicMock(
        max_file_size_bytes=512 * 1024,
        max_files_per_index=100,
        max_chunks_per_index=1_000_000,
        chunk_size=8000,
        chunk_overlap=200,
    )
    with patch(
        "rag_pipeline.api.routers.index._get_singletons",
        return_value=(config, MagicMock()),
    ):
        assert get_limits() == {
            "max_chunks_per_index": 1_000_000,
            "max_file_size_bytes": 512 * 1024,
            "max_files_per_index": 100,
            "chunk_size": 8000,
            "chunk_overlap": 200,
        }


@patch.dict("os.environ", {"ALLOWED_REPO_ROOT": "/tmp"})
def test_full_index_forwards_only_exact_generation_identity():
    manager = MagicMock()
    manager.index_repository.return_value = _stats()
    request = IndexRequest(
        repo_path="/tmp/repository",
        workspace="ws",
        project="project",
        branch="main",
        commit="commit",
        source_tree_sha256="a" * 64,
        collection_target="generation-target",
    )

    with patch(
        "rag_pipeline.api.routers.index._get_singletons",
        return_value=(MagicMock(), manager),
    ):
        assert index_repository(request, BackgroundTasks()) == _stats()

    assert manager.index_repository.call_args.kwargs == {
        "repo_path": "/tmp/repository",
        "workspace": "ws",
        "project": "project",
        "branch": "main",
        "commit": "commit",
        "include_patterns": None,
        "exclude_patterns": None,
        "project_type": None,
        "source_root": None,
        "source_tree_sha256": "a" * 64,
        "collection_target": "generation-target",
    }


@patch.dict("os.environ", {"ALLOWED_REPO_ROOT": "/tmp"})
def test_direct_index_allocates_fresh_opaque_target_and_delegates_attestation():
    manager = MagicMock()
    manager.index_repository.return_value = _stats()
    request = IndexRequest(
        repo_path="/tmp/repository",
        workspace="ws",
        project="project",
        branch="main",
        commit="commit",
    )

    first_target = _index_collection_target(request)
    second_target = _index_collection_target(request)
    assert first_target.startswith("cc_http_g_")
    assert second_target.startswith("cc_http_g_")
    assert first_target != second_target

    with patch(
        "rag_pipeline.api.routers.index._get_singletons",
        return_value=(MagicMock(), manager),
    ):
        assert index_repository(request, BackgroundTasks()) == _stats()

    forwarded = manager.index_repository.call_args.kwargs
    assert forwarded["source_tree_sha256"] is None
    assert forwarded["collection_target"].startswith("cc_http_g_")


@patch.dict("os.environ", {"ALLOWED_REPO_ROOT": "/tmp"})
def test_source_tree_revision_mismatch_maps_to_conflict():
    manager = MagicMock()
    manager.index_repository.side_effect = RepositorySourceTreeError(
        "repository Git HEAD does not match the supplied commit"
    )
    request = IndexRequest(
        repo_path="/tmp/repository",
        workspace="ws",
        project="project",
        branch="main",
        commit="commit",
    )

    with patch(
        "rag_pipeline.api.routers.index._get_singletons",
        return_value=(MagicMock(), manager),
    ), pytest.raises(HTTPException) as exception:
        index_repository(request, BackgroundTasks())

    assert exception.value.status_code == 409


def test_exact_revision_preflight_requires_and_forwards_target():
    manager = MagicMock()
    manager.get_revision_preflight.return_value = _preflight_receipt()
    with patch(
        "rag_pipeline.api.routers.index._get_singletons",
        return_value=(MagicMock(), manager),
    ):
        result = get_revision_preflight(
            "ws",
            "project",
            branch="main",
            commit="commit",
            collection_target="cc_http_g_exact",
        )

    assert result["generation_manifest_sha256"] == "3" * 64
    manager.get_revision_preflight.assert_called_once_with(
        "ws",
        "project",
        "main",
        "commit",
        collection_target="cc_http_g_exact",
    )


def test_exact_revision_preflight_returns_not_found_for_absent_target():
    manager = MagicMock()
    manager.get_revision_preflight.return_value = None
    with patch(
        "rag_pipeline.api.routers.index._get_singletons",
        return_value=(MagicMock(), manager),
    ), pytest.raises(HTTPException) as exception:
        get_revision_preflight(
            "ws",
            "project",
            branch="main",
            commit="commit",
            collection_target="missing-target",
        )

    assert exception.value.status_code == 404


def test_exact_revision_preflight_maps_integrity_mismatch_to_conflict():
    manager = MagicMock()
    manager.get_revision_preflight.side_effect = ExactIndexPreconditionError(
        "repository generation membership failed integrity validation"
    )
    with patch(
        "rag_pipeline.api.routers.index._get_singletons",
        return_value=(MagicMock(), manager),
    ), pytest.raises(HTTPException) as exception:
        get_revision_preflight(
            "ws",
            "project",
            branch="main",
            commit="commit",
            collection_target="cc_http_g_exact",
        )

    assert exception.value.status_code == 409


def test_exact_generation_delete_requires_and_forwards_receipt():
    manager = MagicMock()
    manager.delete_branch.return_value = True
    with patch(
        "rag_pipeline.api.routers.index._get_singletons",
        return_value=(MagicMock(), manager),
    ):
        result = delete_branch(
            "ws",
            "project",
            "main",
            collection_target="generation-target",
            generation_revision="commit",
            generation_manifest_sha256="b" * 64,
        )

    assert result["status"] == "success"
    manager.delete_branch.assert_called_once_with(
        "ws",
        "project",
        "main",
        collection_target="generation-target",
        generation_revision="commit",
        generation_manifest_sha256="b" * 64,
    )


def test_exact_generation_delete_maps_receipt_mismatch_to_conflict():
    manager = MagicMock()
    manager.delete_branch.side_effect = ExactIndexPreconditionError(
        "receipt mismatch"
    )
    with patch(
        "rag_pipeline.api.routers.index._get_singletons",
        return_value=(MagicMock(), manager),
    ), pytest.raises(HTTPException) as exception:
        delete_branch(
            "ws",
            "project",
            "main",
            collection_target="generation-target",
            generation_revision="commit",
            generation_manifest_sha256="b" * 64,
        )

    assert exception.value.status_code == 409
