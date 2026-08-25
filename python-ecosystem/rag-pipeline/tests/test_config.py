"""Unit tests for structural repository-index configuration."""

import os
from unittest.mock import patch

import pytest

from rag_pipeline.models.config import IndexStats, RAGConfig


class TestRAGConfig:
    def test_default_values(self):
        config = RAGConfig()
        assert config.qdrant_timeout_seconds == 30
        assert config.qdrant_upsert_max_payload_bytes == 8 * 1024 * 1024
        assert config.full_index_concurrency == 1
        assert config.architecture_finalization_timeout_seconds == 600
        assert config.max_file_size_bytes == 512 * 1024
        assert config.max_files_per_index == 50000
        assert config.chunk_size == 8000
        assert config.chunk_overlap == 200
        assert config.max_chunks_per_index == 1_000_000

    def test_qdrant_timeout_is_configurable(self):
        with patch.dict(os.environ, {"QDRANT_TIMEOUT_SECONDS": "45"}):
            assert RAGConfig().qdrant_timeout_seconds == 45

    def test_write_limits_are_configurable(self):
        with patch.dict(os.environ, {
            "QDRANT_UPSERT_MAX_PAYLOAD_BYTES": "4194304",
            "RAG_FULL_INDEX_CONCURRENCY": "2",
        }):
            config = RAGConfig()
            assert config.qdrant_upsert_max_payload_bytes == 4194304
            assert config.full_index_concurrency == 2

    def test_max_file_size_is_configurable(self):
        with patch.dict(os.environ, {"RAG_MAX_FILE_SIZE_BYTES": "262144"}):
            assert RAGConfig().max_file_size_bytes == 262144

    @pytest.mark.parametrize("configured", [0, -1])
    def test_max_file_size_must_be_positive(self, configured):
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            RAGConfig(max_file_size_bytes=configured)

    def test_excluded_patterns_defaults(self):
        config = RAGConfig()
        assert "node_modules/**" in config.excluded_patterns
        assert ".git/**" in config.excluded_patterns
        assert "*.min.js" in config.excluded_patterns


class TestIndexStats:
    def test_round_trip(self):
        stats = IndexStats(
            namespace="ws__proj__main",
            document_count=100,
            chunk_count=500,
            last_updated="2026-01-01T00:00:00",
            workspace="ws",
            project="proj",
            branch="main",
        )
        restored = IndexStats(**stats.model_dump())
        assert restored.namespace == "ws__proj__main"
        assert restored.chunk_count == 500
