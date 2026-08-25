"""
Unit tests for rag_pipeline.api.models — Pydantic request/response models.
"""
import os
import pytest
from unittest.mock import patch

from rag_pipeline.api.models import (
    IndexRequest,
    CodeSearchRequest,
    DeterministicContextRequest,
    ParseFileRequest,
    ParseBatchRequest,
    ParsedFileMetadata,
    PRFileInfo,
    PRIndexRequest,
    EstimateRequest,
    EstimateResponse,
    RepositoryIndexGraphRequest,
    RepositoryIndexNodeRequest,
)


class TestIndexRequest:

    @patch.dict(os.environ, {"ALLOWED_REPO_ROOT": "/tmp"})
    def test_valid_path(self):
        req = IndexRequest(
            repo_path="/tmp/repo",
            workspace="ws",
            project="proj",
            branch="main",
            commit="abc123",
            source_tree_sha256="a" * 64,
            collection_target="generation-target",
        )
        assert req.workspace == "ws"
        assert req.transfer_repo_ownership is False

    @patch.dict(os.environ, {"ALLOWED_REPO_ROOT": "/tmp"})
    def test_direct_api_may_delegate_exact_identity_to_server(self):
        req = IndexRequest(
            repo_path="/tmp/repo",
            workspace="ws",
            project="proj",
            branch="main",
            commit="abc123",
        )

        assert req.source_tree_sha256 is None
        assert req.collection_target is None

    @patch.dict(os.environ, {"ALLOWED_REPO_ROOT": "/tmp"})
    def test_malformed_explicit_source_identity_is_rejected(self):
        with pytest.raises(ValueError, match="source_tree_sha256"):
            IndexRequest(
                repo_path="/tmp/repo",
                workspace="ws",
                project="proj",
                branch="main",
                commit="abc123",
                source_tree_sha256="not-a-sha256",
            )

    @patch.dict(os.environ, {"ALLOWED_REPO_ROOT": "/tmp"})
    def test_stream_repository_ownership_requires_explicit_opt_in(self):
        req = IndexRequest(
            repo_path="/tmp/codecrow-rag-branch-generation-owned",
            workspace="ws",
            project="proj",
            branch="main",
            commit="abc123",
            source_tree_sha256="a" * 64,
            collection_target="generation-target",
            transfer_repo_ownership=True,
        )
        assert req.transfer_repo_ownership is True

    @patch.dict(os.environ, {"ALLOWED_REPO_ROOT": "/tmp"})
    def test_path_traversal_rejected(self):
        with pytest.raises(ValueError, match="Path must be under"):
            IndexRequest(
                repo_path="/etc/passwd",
                workspace="ws",
                project="proj",
                branch="main",
                commit="abc123",
                source_tree_sha256="a" * 64,
                collection_target="generation-target",
            )

    @patch.dict(os.environ, {"ALLOWED_REPO_ROOT": "/tmp"})
    def test_manual_project_profile_accepts_arbitrary_nested_source_root(self):
        req = IndexRequest(
            repo_path="/tmp/repo",
            workspace="ws",
            project="proj",
            branch="main",
            commit="abc123",
            source_tree_sha256="a" * 64,
            collection_target="generation-target",
            project_type="magento",
            source_root=r"magento\src\etc",
        )

        assert req.project_type == "magento"
        assert req.source_root == "magento/src/etc"

    @patch.dict(os.environ, {"ALLOWED_REPO_ROOT": "/tmp"})
    def test_auto_project_profile_normalizes_to_marker_detection(self):
        req = IndexRequest(
            repo_path="/tmp/repo",
            workspace="ws",
            project="proj",
            branch="main",
            commit="abc123",
            source_tree_sha256="a" * 64,
            collection_target="generation-target",
            project_type=" AUTO ",
        )

        assert req.project_type is None

    @patch.dict(os.environ, {"ALLOWED_REPO_ROOT": "/tmp"})
    @pytest.mark.parametrize("source_root", ["/magento", "magento/", "../magento", "magento//src"])
    def test_source_root_must_be_a_repository_relative_directory(self, source_root):
        with pytest.raises(ValueError, match="source_root"):
            IndexRequest(
                repo_path="/tmp/repo",
                workspace="ws",
                project="proj",
                branch="main",
                commit="abc123",
                source_tree_sha256="a" * 64,
                collection_target="generation-target",
                project_type="magento",
                source_root=source_root,
            )


class TestCodeSearchRequest:

    def test_exact_generation_binding_is_required(self):
        request = CodeSearchRequest(
            query="UserService",
            workspace="ws",
            project="proj",
            branch="main",
            repository_revision="abc123",
            repository_generation_manifest_sha256="a" * 64,
            collection_target="generation-target",
        )
        assert request.limit is None
        assert request.repository_revision == "abc123"

    def test_limit_is_bounded(self):
        with pytest.raises(ValueError):
            CodeSearchRequest(
                query="UserService",
                workspace="ws",
                project="proj",
                branch="main",
                repository_revision="abc123",
                repository_generation_manifest_sha256="a" * 64,
                collection_target="generation-target",
                limit=5001,
            )


class TestDeterministicContextRequest:

    def test_basic_construction(self):
        req = DeterministicContextRequest(
            workspace="ws",
            project="proj",
            branches=["main"],
            file_paths=["src/main.py"],
            base_revision="abc123",
            base_generation_manifest_sha256="a" * 64,
            collection_target="generation-target",
        )
        assert req.limit_per_file is None
        assert req.additional_identifiers is None

    def test_with_additional_identifiers(self):
        req = DeterministicContextRequest(
            workspace="ws",
            project="proj",
            branches=["main"],
            file_paths=["a.py"],
            base_revision="abc123",
            base_generation_manifest_sha256="a" * 64,
            collection_target="generation-target",
            additional_identifiers=["UserService", "OrderRepository"],
        )
        assert len(req.additional_identifiers) == 2


class TestParseModels:

    def test_parse_file_request(self):
        req = ParseFileRequest(path="main.py", content="print('hello')")
        assert req.language is None

    def test_parsed_file_metadata_defaults(self):
        meta = ParsedFileMetadata(path="main.py")
        assert meta.imports == []
        assert meta.extends == []
        assert meta.success is True
        assert meta.error is None

    def test_parse_batch_request(self):
        req = ParseBatchRequest(files=[
            ParseFileRequest(path="a.py", content="x = 1"),
            ParseFileRequest(path="b.py", content="y = 2"),
        ])
        assert len(req.files) == 2


class TestPRIndexRequest:

    def test_construction(self):
        req = PRIndexRequest(
            workspace="ws",
            project="proj",
            pr_number=42,
            branch="feature",
            source_revision="head-commit",
            base_revision="base-commit",
            base_generation_manifest_sha256="a" * 64,
            collection_target="generation-target",
            files=[
                PRFileInfo(path="src/main.py", content="x = 1", change_type="MODIFIED"),
            ],
        )
        assert req.pr_number == 42
        assert req.source_revision == "head-commit"
        assert req.base_revision == "base-commit"
        assert len(req.files) == 1
        assert req.files[0].change_type == "MODIFIED"
        assert req.files[0].content_state == "complete"

    def test_partial_diff_state_is_explicit_and_validated(self):
        partial = PRFileInfo(
            path="src/main.py",
            content="@@ -1 +1 @@\n-old\n+new",
            change_type="MODIFIED",
            content_state="partial_diff",
        )

        assert partial.content_state == "partial_diff"
        assert PRFileInfo(
            path="src/main.py",
            content="x = 1",
            change_type="modified",
        ).change_type == "MODIFIED"
        with pytest.raises(ValueError):
            PRFileInfo(
                path="src/main.py",
                content="x = 1",
                change_type="MODIFIED",
                content_state="unknown",
            )
        with pytest.raises(ValueError):
            PRFileInfo(
                path="src/main.py",
                content="x = 1",
                change_type="UNKNOWN",
            )


class TestEstimateResponse:

    def test_round_trip(self):
        resp = EstimateResponse(
            file_count=100,
            estimated_chunks=500,
            max_files_allowed=50000,
            max_chunks_allowed=1000000,
            within_limits=True,
            message="OK",
        )
        data = resp.model_dump()
        restored = EstimateResponse(**data)
        assert restored.within_limits is True


class TestRepositoryIndexInspectionModels:

    def test_graph_request_defaults(self):
        req = RepositoryIndexGraphRequest(collection_target="generation-target")
        assert req.limit == 160
        assert req.scan_limit == 2500
        assert req.filters.include_pr is True

    def test_graph_limits_are_bounded(self):
        with pytest.raises(ValueError):
            RepositoryIndexGraphRequest(collection_target="generation-target", limit=5001)

        with pytest.raises(ValueError):
            RepositoryIndexGraphRequest(collection_target="generation-target", scan_limit=99)

    def test_node_neighbor_limit_is_bounded(self):
        req = RepositoryIndexNodeRequest(collection_target="generation-target", neighbor_limit=40)
        assert req.neighbor_limit == 40

        with pytest.raises(ValueError):
            RepositoryIndexNodeRequest(collection_target="generation-target", neighbor_limit=500)
