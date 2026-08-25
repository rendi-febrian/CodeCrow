"""Focused unit coverage for structural index manager components."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client.models import Distance, VectorParams


def _collection_info(*, size=1, distance=Distance.DOT):
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=VectorParams(size=size, distance=distance),
            )
        ),
        payload_schema={},
    )


class TestCollectionManager:
    def _make(self):
        from rag_pipeline.core.index_manager.collection_manager import (
            CollectionManager,
        )

        client = MagicMock()
        client.get_aliases.return_value = SimpleNamespace(aliases=[])
        client.get_collections.return_value = SimpleNamespace(collections=[])
        client.get_collection.return_value = _collection_info()
        return CollectionManager(client)

    def test_creates_one_dimensional_dot_payload_store(self):
        manager = self._make()
        manager._ensure_payload_indexes = MagicMock(return_value=True)

        manager.create_pending_collection("structural", operation_id="a")

        config = manager.client.create_collection.call_args.kwargs[
            "vectors_config"
        ]
        assert config.size == 1
        assert config.distance is Distance.DOT
        assert config.on_disk is True

    def test_existing_vector_collection_is_rejected(self):
        from rag_pipeline.core.exact_index import ExactIndexPreconditionError

        manager = self._make()
        manager.client.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name="pre_structural")]
        )
        manager.client.get_collection.return_value = _collection_info(
            size=768,
            distance=Distance.COSINE,
        )

        with pytest.raises(
            ExactIndexPreconditionError,
            match="predates structural payload storage",
        ):
            manager.require_structural_collection("pre_structural")

        manager.client.create_collection.assert_not_called()
        manager.client.create_payload_index.assert_not_called()

    def test_alias_target_schema_is_checked_before_repair(self):
        from rag_pipeline.core.exact_index import ExactIndexPreconditionError

        manager = self._make()
        manager.client.get_aliases.return_value = SimpleNamespace(aliases=[
            SimpleNamespace(alias_name="active", collection_name="pre_structural"),
        ])
        manager.client.get_collection.return_value = _collection_info(size=384)

        with pytest.raises(ExactIndexPreconditionError):
            manager.require_structural_collection("active")

        manager.client.create_payload_index.assert_not_called()

    def test_payload_indexes_cover_structural_lookup_fields(self):
        manager = self._make()

        manager._ensure_payload_indexes("structural")

        fields = {
            call.kwargs["field_name"]
            for call in manager.client.create_payload_index.call_args_list
        }
        assert {
            "primary_name",
            "search_terms",
            "structural_record_type",
            "architecture_paths",
        } <= fields

    def test_pending_collections_are_unique(self):
        manager = self._make()
        manager._ensure_payload_indexes = MagicMock(return_value=True)

        first = manager.create_pending_collection("repository")
        second = manager.create_pending_collection("repository")

        assert first != second
        assert first.startswith("repository_pending_")
        assert second.startswith("repository_pending_")

    def test_atomic_assignment_moves_requested_aliases_together(self):
        manager = self._make()
        manager.client.get_aliases.return_value = SimpleNamespace(aliases=[
            SimpleNamespace(alias_name="project", collection_name="old"),
        ])

        manager.atomic_assign_aliases({
            "project": "new",
            "project__main": "new",
        })

        operations = manager.client.update_collection_aliases.call_args.kwargs[
            "change_aliases_operations"
        ]
        assert len(operations) == 3


class TestBranchManager:
    def test_count_uses_payload_filter(self):
        from rag_pipeline.core.index_manager.branch_manager import BranchManager

        client = MagicMock()
        client.count.return_value = SimpleNamespace(count=42)
        manager = BranchManager(client)

        assert manager.get_branch_point_count("structural", "feature") == 42
        client.count.assert_called_once()


class TestRAGIndexManager:
    def test_missing_caller_attestation_is_derived_before_indexing(
        self,
        tmp_path,
    ):
        from rag_pipeline.core.index_manager.manager import RAGIndexManager
        from rag_pipeline.core.source_tree import (
            compute_repository_source_tree_sha256,
        )

        (tmp_path / "Example.php").write_text("<?php\n", encoding="utf-8")
        expected_tree_sha256 = compute_repository_source_tree_sha256(tmp_path)
        manager = object.__new__(RAGIndexManager)
        manager._mutation_coordinator = MagicMock()
        lease = SimpleNamespace(
            token="operation-token",
            assert_owned=MagicMock(),
        )
        manager._mutation_coordinator.acquire.return_value.__enter__.return_value = (
            lease
        )
        manager._indexer = MagicMock()
        expected_result = MagicMock()
        manager._indexer.index_repository.return_value = expected_result

        result = manager._index_repository_admitted(
            repo_path=str(tmp_path),
            workspace="ws",
            project="project",
            branch="main",
            commit="a" * 40,
            source_tree_sha256=None,
            collection_target="cc_http_g_exact",
        )

        assert result is expected_result
        forwarded = manager._indexer.index_repository.call_args.kwargs
        assert forwarded["source_tree_sha256"] == expected_tree_sha256
        assert forwarded["source_tree"].tree_sha256 == expected_tree_sha256
        assert forwarded["source_tree"].git_commit_verified is False

    def test_close_releases_coordinator_and_qdrant_client(self):
        from rag_pipeline.core.index_manager.manager import RAGIndexManager

        manager = object.__new__(RAGIndexManager)
        manager._mutation_coordinator = MagicMock()
        manager.qdrant_client = MagicMock()

        manager.close()

        manager._mutation_coordinator.close.assert_called_once_with()
        manager.qdrant_client.close.assert_called_once_with()
