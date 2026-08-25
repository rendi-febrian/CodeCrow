"""Focused unit coverage for production API routers."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestSystemRouter:
    def test_root(self):
        from rag_pipeline.api.routers.system import root

        assert root()["message"] == "CodeCrow Repository Index API"

    def test_health(self):
        from rag_pipeline.api.routers.system import health

        assert asyncio.run(health())["status"] == "healthy"


class TestParseRouter:
    @patch("rag_pipeline.core.splitter.ASTCodeSplitter")
    def test_parse_file_projects_ast_chunk_metadata(self, splitter_class):
        from rag_pipeline.api.models import ParseFileRequest
        from rag_pipeline.api.routers.parse import parse_file

        splitter_class.return_value.split_documents.return_value = [
            SimpleNamespace(metadata={
                "imports": ["os"],
                "symbol_names": ["hello"],
                "calls": ["print"],
            })
        ]

        result = parse_file(ParseFileRequest(
            path="test.py",
            content="import os\ndef hello():\n    pass\n",
            language="python",
        ))
        assert result.path == "test.py"
        assert result.success is True
        assert result.language == "python"
        assert result.imports == ["os"]
        assert result.symbol_names == ["hello"]
        assert result.calls == ["print"]
        parsed_documents = (
            splitter_class.return_value.split_documents.call_args.args[0]
        )
        assert parsed_documents[0].metadata == {"path": "test.py"}


class TestIndexRouter:
    @patch("rag_pipeline.api.routers.index._get_singletons")
    def test_get_limits(self, mock_singletons):
        from rag_pipeline.api.routers.index import get_limits

        config = MagicMock(
            max_file_size_bytes=512 * 1024,
            max_files_per_index=5000,
            max_chunks_per_index=1_000_000,
            chunk_size=8000,
            chunk_overlap=200,
        )
        mock_singletons.return_value = (config, MagicMock())
        assert get_limits() == {
            "max_chunks_per_index": 1_000_000,
            "max_file_size_bytes": 512 * 1024,
            "max_files_per_index": 5000,
            "chunk_size": 8000,
            "chunk_overlap": 200,
        }


class TestQueryRouter:
    @patch("rag_pipeline.api.routers.query._get_singletons")
    def test_code_search_is_revision_bound(self, mock_singletons):
        from rag_pipeline.api.models import CodeSearchRequest
        from rag_pipeline.api.routers.query import code_search

        manager = MagicMock()
        manager._get_project_collection_name.return_value = "index"
        manager._collection_manager.require_structural_collection.return_value = (
            "index-generation"
        )
        manager.get_revision_preflight.return_value = {
            "generation_manifest_sha256": "a" * 64,
        }
        service = MagicMock()
        service.search_code.return_value = [{
            "path": "src/main.py",
            "match_reasons": ["exact indexed token: main"],
        }]
        mock_singletons.return_value = (manager, service)

        result = code_search(CodeSearchRequest(
            query="main",
            workspace="ws",
            project="proj",
            branch="main",
            repository_revision="abc123",
            repository_generation_manifest_sha256="a" * 64,
            collection_target="generation-target",
        ))

        assert result["results"][0]["match_reasons"] == [
            "exact indexed token: main"
        ]
        assert service.search_code.call_args.kwargs["collection_target"] == (
            "index-generation"
        )

    @patch("rag_pipeline.api.routers.query._get_singletons")
    def test_deterministic_context(self, mock_singletons):
        from rag_pipeline.api.models import DeterministicContextRequest
        from rag_pipeline.api.routers.query import get_deterministic_context

        manager = MagicMock()
        manager._collection_manager.require_structural_collection.return_value = (
            "index-generation"
        )
        manager.get_revision_preflight.return_value = {
            "generation_manifest_sha256": "a" * 64,
        }
        service = MagicMock()
        service.get_deterministic_context.return_value = {"chunks": []}
        mock_singletons.return_value = (manager, service)
        result = get_deterministic_context(DeterministicContextRequest(
            workspace="ws",
            project="proj",
            branches=["main"],
            file_paths=["src/main.py"],
            base_revision="abc123",
            base_generation_manifest_sha256="a" * 64,
            collection_target="generation-target",
        ))
        assert result == {"context": {"chunks": []}}
        assert service.get_deterministic_context.call_args.kwargs[
            "limit_per_file"
        ] is None
