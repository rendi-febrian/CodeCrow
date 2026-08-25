"""
Tests for rag_pipeline.core.index_manager.indexer — RepositoryIndexer.

Covers:
- estimate_repository_size (small repo, sampled large repo)
- index_repository (full flow, limits, atomic swap, errors)
- _perform_atomic_swap (normal, migration from direct collection)
"""
import gc
import pytest
from unittest.mock import patch, MagicMock, PropertyMock, call
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from rag_pipeline.core.index_manager.indexer import (
    ARCHITECTURE_SOURCE_PART_SIZE,
    RepositoryIndexer,
    DOCUMENT_BATCH_SIZE,
    INSERT_BATCH_SIZE,
)
from rag_pipeline.core.loader import (
    REPOSITORY_FILE_SIZE_LIMIT_CODE,
    RepositoryFileSkip,
)
def _mock_config(**overrides):
    cfg = MagicMock()
    cfg.max_file_size_bytes = 512 * 1024
    cfg.max_files_per_index = 0
    cfg.max_chunks_per_index = 1_000_000
    cfg.architecture_finalization_timeout_seconds = 600
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _mock_components():
    """Return mocked sub-components for RepositoryIndexer."""
    coll_mgr = MagicMock()
    coll_mgr.physical_collection_exists.return_value = False
    branch_mgr = MagicMock()
    point_ops = MagicMock()
    stats_mgr = MagicMock()
    splitter = MagicMock()
    loader = MagicMock()

    # Default: point_ops returns success
    point_ops.process_and_store_chunks.side_effect = (
        lambda nodes, *_args, **_kwargs: (len(nodes), 0)
    )
    point_ops.client = MagicMock()
    point_ops.client.scroll.return_value = ([SimpleNamespace(
        id="stored-member",
        payload={"generation_member_sha256": "a" * 64},
    )], None)
    branch_mgr.get_branch_point_count.return_value = 1
    branch_mgr.stream_copy_points_to_collection.return_value = 0
    splitter.split_documents.return_value = []
    splitter.split_documents_resilient.side_effect = (
        lambda documents, capabilities=None: (
            splitter.split_documents(
                documents,
                capabilities=capabilities,
            ),
            (),
        )
    )

    return coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader


# ─────────────────────────────────────────────────────────────
# estimate_repository_size
# ─────────────────────────────────────────────────────────────
class TestEstimateRepositorySize:

    def test_empty_repo(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()
        loader.iter_repository_files.return_value = iter([])

        indexer = RepositoryIndexer(config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader)
        fc, cc = indexer.estimate_repository_size("/repo")
        assert fc == 0
        assert cc == 0

    def test_small_repo_exact_count(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()

        fake_files = [f"file{i}.py" for i in range(10)]
        loader.iter_repository_files.return_value = iter(fake_files)

        # Simulate loader returning 2 docs per batch, splitter returning 3 chunks per doc batch
        from rag_pipeline.core.documents import Document as LlamaDoc
        mock_docs = [MagicMock() for _ in range(2)]
        loader.load_file_batch.return_value = mock_docs
        mock_chunks = [MagicMock() for _ in range(3)]
        splitter.split_documents.return_value = mock_chunks

        indexer = RepositoryIndexer(config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader)
        fc, cc = indexer.estimate_repository_size("/repo")
        assert fc == 10
        assert cc > 0

    def test_large_repo_sampling(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()

        # More than SAMPLE_SIZE=100 files
        fake_files = [f"file{i}.py" for i in range(200)]
        loader.iter_repository_files.return_value = iter(fake_files)

        loader.load_file_batch.return_value = [
            SimpleNamespace(text="a = 1", metadata={"path": "a.py"})
        ]
        splitter.split_documents.return_value = [MagicMock(), MagicMock()]

        indexer = RepositoryIndexer(config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader)
        fc, cc = indexer.estimate_repository_size("/repo")
        assert fc == 200
        assert cc > 0  # Estimated from sampling


# ─────────────────────────────────────────────────────────────
# index_repository
# ─────────────────────────────────────────────────────────────
class TestIndexRepository:

    def test_empty_repo_publishes_an_exact_empty_generation(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()
        loader.iter_repository_files.return_value = iter([])

        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.alias_exists.return_value = False
        coll_mgr.collection_exists.return_value = False
        coll_mgr.resolve_alias.return_value = None

        point_ops.client.get_collection.return_value = SimpleNamespace(
            points_count=2
        )
        branch_mgr.get_branch_point_count.return_value = 2

        indexer = RepositoryIndexer(config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader)
        progress_events = []
        result = indexer.index_repository(
            "/repo", "ws", "proj", "main", "abc123", "alias1",
            source_tree_sha256="a" * 64,
            progress_callback=progress_events.append,
        )

        assert result.document_count == 0
        assert result.chunk_count == 1
        assert len(result.generation_manifest_sha256) == 64
        assert progress_events[-1]["stage"] == "complete"
        assert progress_events[-1]["total"] == 0
        coll_mgr.atomic_assign_aliases.assert_called_once_with({
            "alias1": "pending"
        })
        coll_mgr.delete_collection.assert_not_called()
        stats_mgr.get_branch_stats.assert_not_called()
        stats_mgr.store_metadata.assert_called_once()

    def test_progress_callback_failure_does_not_fail_indexing(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()
        loader.iter_repository_files.return_value = iter([])
        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.alias_exists.return_value = False
        coll_mgr.collection_exists.return_value = False
        coll_mgr.resolve_alias.return_value = None
        point_ops.client.get_collection.return_value = SimpleNamespace(
            points_count=2
        )
        branch_mgr.get_branch_point_count.return_value = 2
        indexer = RepositoryIndexer(
            config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader,
        )

        result = indexer.index_repository(
            "/repo", "ws", "proj", "main", "abc123", "alias1",
            source_tree_sha256="a" * 64,
            progress_callback=lambda _event: (_ for _ in ()).throw(
                RuntimeError("event sink unavailable")
            ),
        )

        assert result.document_count == 0

    def test_all_oversized_files_return_an_observable_skip_count(self):
        config = _mock_config(max_file_size_bytes=100)
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()

        def scan_files(*_args, **kwargs):
            on_skip = kwargs.get("on_skip")
            if on_skip is not None:
                on_skip(RepositoryFileSkip(
                    code=REPOSITORY_FILE_SIZE_LIMIT_CODE,
                    path="large.js",
                    message="omitted whole",
                    size_bytes=101,
                    max_file_size_bytes=100,
                ))
            return iter([])

        loader.iter_repository_files.side_effect = scan_files
        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.resolve_collection_target.return_value = "active"
        point_ops.client.get_collection.return_value = SimpleNamespace(
            points_count=2
        )
        branch_mgr.get_branch_point_count.return_value = 2
        progress_events = []
        indexer = RepositoryIndexer(
            config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader,
        )

        result = indexer.index_repository(
            "/repo",
            "ws",
            "proj",
            "main",
            "abc123",
            "alias1",
            source_tree_sha256="a" * 64,
            progress_callback=progress_events.append,
        )

        assert result.document_count == 0
        assert result.chunk_count == 1
        assert result.skipped_file_count == 1
        assert result.source_tree_sha256 == "a" * 64
        assert len(result.generation_manifest_sha256) == 64
        scan_event = next(
            event for event in progress_events if event["stage"] == "scanning"
        )
        assert scan_event["oversizedFileCount"] == 1
        assert scan_event["skippedFiles"] == 1
        assert progress_events[-1]["stage"] == "complete"
        assert progress_events[-1]["oversizedFileCount"] == 1
        assert progress_events[-1]["skippedFiles"] == 1
        splitter.split_documents.assert_not_called()
        loader.load_file_batch.assert_not_called()
        stats_mgr.get_branch_stats.assert_not_called()
        coll_mgr.atomic_assign_aliases.assert_called_once_with({
            "alias1": "pending"
        })
        coll_mgr.delete_collection.assert_not_called()
        manifest = next(
            invocation.args[0][0]
            for invocation in point_ops.process_and_store_chunks.call_args_list
            if invocation.args[0]
            and invocation.args[0][0].metadata.get(
                "repository_generation_manifest"
            )
        )
        assert manifest.metadata["generation_member_count"] == 1
        assert manifest.metadata["generation_manifest_sha256"] == (
            result.generation_manifest_sha256
        )

    def test_architecture_only_batch_is_ingested_and_reports_completion(
        self,
        tmp_path,
        monkeypatch,
    ):
        from codecrow_plugins import (
            FileDisposition,
            PluginDiagnostic,
            ProjectCapabilities,
            RepositoryAnalysis,
        )
        from rag_pipeline.core.index_manager import indexer as indexer_module

        monkeypatch.setattr(indexer_module, "DOCUMENT_BATCH_SIZE", 1)

        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()
        paths = [
            Path("source.php"),
            Path("etc/di.xml"),
            Path("generated/code/Proxy.php"),
        ]
        loader.iter_repository_files.return_value = iter(paths)
        documents_by_path = {
            "source.php": SimpleNamespace(
                text="<?php class Source {}",
                metadata={"path": "source.php"},
            ),
            "etc/di.xml": SimpleNamespace(
                text="<config/>",
                metadata={"path": "etc/di.xml"},
            ),
        }
        loader.load_file_batch.side_effect = lambda batch, *_args, **_kwargs: [
            documents_by_path[Path(path).as_posix()]
            for path in batch
        ]
        splitter.split_documents.return_value = [MagicMock()]
        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.alias_exists.return_value = False
        coll_mgr.collection_exists.return_value = False
        coll_mgr.resolve_alias.return_value = None
        # One bounded source chunk, one architecture-source node, one plugin
        # symbol node, and the generation manifest seal the collection.
        point_ops.client.get_collection.return_value = SimpleNamespace(points_count=4)
        branch_mgr.get_branch_point_count.return_value = 4
        stats_mgr.store_metadata.return_value = None
        selector = MagicMock()
        selector.select.return_value = ProjectCapabilities(
            repository_plugins=("php", "magento"),
            file_plugins={}, detection_evidence={}, unavailable_capabilities=(),
            fingerprint="sha256:" + "0" * 64,
        )
        handle = MagicMock(active=True)
        handle.finish.return_value = (
            RepositoryAnalysis(),
            (PluginDiagnostic(
                code="plugin-repository-finalization-timeout",
                message="Magento architecture exceeded its time budget",
                plugin_id="magento",
                recoverable=True,
            ),),
        )
        runtime = MagicMock()
        runtime.start_repository_analysis.return_value = handle
        runtime.file_disposition.side_effect = lambda path, _capabilities: {
            "source.php": FileDisposition.FULL,
            "etc/di.xml": FileDisposition.ARCHITECTURE_ONLY,
            "generated/code/Proxy.php": FileDisposition.GENERATED,
        }[path]
        indexer = RepositoryIndexer(
            config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader,
            plugin_catalog=MagicMock(), plugin_runtime=runtime, plugin_selector=selector,
        )

        progress_events = []
        indexer.index_repository(
            str(tmp_path), "ws", "proj", "main", "abc", "alias",
            source_tree_sha256="a" * 64,
            progress_callback=progress_events.append,
        )

        ingested_paths = [
            artifact.path
            for invocation in handle.ingest.call_args_list
            for artifact in invocation.args[0]
        ]
        assert ingested_paths == ["source.php", "etc/di.xml"]
        loaded_paths = [
            invocation.args[0]
            for invocation in loader.load_file_batch.call_args_list
        ]
        assert loaded_paths == [[Path("source.php")], [Path("etc/di.xml")]]
        structural_documents = [
            document
            for invocation in splitter.split_documents.call_args_list[-2:]
            for document in invocation.args[0]
        ]
        assert [
            document.metadata["path"] for document in structural_documents
        ] == ["source.php"]
        stored_batches = [
            invocation.args[0]
            for invocation in point_ops.process_and_store_chunks.call_args_list
        ]
        architecture_sources = [
            node
            for batch in stored_batches
            for node in batch
            if getattr(node, "metadata", {}).get("path") == "etc/di.xml"
            and node.metadata.get("architecture_source") is True
        ]
        assert [node.text for node in architecture_sources] == ["<config/>"]
        batch_events = [
            event for event in progress_events
            if event["stage"] == "indexing" and "completedBatches" in event
        ]
        assert [event["completedBatches"] for event in batch_events] == [1, 2]
        assert batch_events[-1]["architectureOnlyFiles"] == 1
        assert batch_events[-1]["architectureSourceRecords"] == 1
        assert batch_events[-1]["estimatedRemainingMs"] == 0
        assert batch_events[-1]["remainingEstimateScope"] == "file_batches"
        assert any(
            event.get("architectureStatus") == "degraded"
            and event.get("degraded") is True
            for event in progress_events
        )

    def test_exceeds_file_limit(self):
        config = _mock_config(max_files_per_index=5)
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()

        fake_files = [f"f{i}.py" for i in range(10)]
        loader.iter_repository_files.return_value = iter(fake_files)

        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.alias_exists.return_value = False
        coll_mgr.collection_exists.return_value = False
        coll_mgr.resolve_alias.return_value = None

        indexer = RepositoryIndexer(config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader)

        with pytest.raises(ValueError, match="exceeds file limit"):
            indexer.index_repository("/repo", "ws", "proj", "main", "abc123", "alias1")

        coll_mgr.delete_collection.assert_called_with("pending")

    def test_successful_indexing(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()

        fake_files = ["a.py", "b.py"]
        loader.iter_repository_files.return_value = iter(fake_files)

        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.alias_exists.return_value = False
        coll_mgr.collection_exists.return_value = False
        coll_mgr.resolve_alias.return_value = None

        loader.load_file_batch.return_value = [
            SimpleNamespace(text="a = 1", metadata={"path": "a.py"})
        ]
        splitter.split_documents.return_value = [MagicMock(), MagicMock()]
        # Two bounded source chunks plus the generation manifest.
        temp_info = MagicMock()
        temp_info.points_count = 3
        point_ops.client.get_collection.return_value = temp_info
        branch_mgr.get_branch_point_count.return_value = 3

        indexer = RepositoryIndexer(config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader)
        result = indexer.index_repository(
            "/repo", "ws", "proj", "main", "abc123", "alias1",
            source_tree_sha256="a" * 64,
        )

        assert result.workspace == "ws"
        assert result.project == "proj"
        assert result.branch == "main"
        stats_mgr.store_metadata.assert_called_once()

    def test_oversized_scan_skip_is_counted_and_never_reaches_parsers(self):
        config = _mock_config(max_file_size_bytes=100)
        (
            coll_mgr,
            branch_mgr,
            point_ops,
            stats_mgr,
            splitter,
            loader,
        ) = _mock_components()

        def scan_files(*_args, **kwargs):
            on_skip = kwargs.get("on_skip")
            if on_skip is not None:
                on_skip(RepositoryFileSkip(
                    code=REPOSITORY_FILE_SIZE_LIMIT_CODE,
                    path="large.js",
                    message="omitted whole",
                    size_bytes=101,
                    max_file_size_bytes=100,
                ))
            return iter([Path("small.py")])

        loader.iter_repository_files.side_effect = scan_files
        loader.load_file_batch.return_value = [SimpleNamespace(
            text="value = 1\n",
            metadata={"path": "small.py"},
        )]
        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.physical_collection_exists.return_value = False
        point_ops.client.get_collection.return_value = SimpleNamespace(
            points_count=1
        )
        branch_mgr.get_branch_point_count.return_value = 1
        progress_events = []

        indexer = RepositoryIndexer(
            config,
            coll_mgr,
            branch_mgr,
            point_ops,
            stats_mgr,
            splitter,
            loader,
        )
        result = indexer.index_repository(
            "/repo",
            "ws",
            "proj",
            "main",
            "abc123",
            "alias1",
            source_tree_sha256="a" * 64,
            progress_callback=progress_events.append,
        )

        assert result.document_count == 1
        assert result.skipped_file_count == 1
        parsed_documents = splitter.split_documents.call_args.args[0]
        assert [document.metadata["path"] for document in parsed_documents] == [
            "small.py"
        ]
        scan_event = next(
            event for event in progress_events if event["stage"] == "scanning"
        )
        assert scan_event["skippedFiles"] == 1
        assert scan_event["oversizedFileCount"] == 1
        assert scan_event["maxFileSizeBytes"] == 100
        complete_event = progress_events[-1]
        assert complete_event["stage"] == "complete"
        assert complete_event["skippedFiles"] == 1

    def test_exact_generation_does_not_read_or_copy_an_existing_target(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()

        loader.iter_repository_files.return_value = iter(["a.py"])
        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.alias_exists.return_value = True
        coll_mgr.collection_exists.return_value = True
        coll_mgr.resolve_alias.return_value = "active"
        loader.load_file_batch.return_value = [
            SimpleNamespace(text="a = 1", metadata={"path": "a.py"})
        ]
        splitter.split_documents.return_value = [MagicMock()]
        temp_info = MagicMock()
        temp_info.points_count = 2
        point_ops.client.get_collection.return_value = temp_info
        branch_mgr.get_branch_point_count.return_value = 2

        indexer = RepositoryIndexer(
            config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader
        )
        indexer.index_repository(
            "/repo", "ws", "proj", "main", "abc123", "alias1",
            source_tree_sha256="a" * 64,
        )

        coll_mgr.resolve_alias.assert_not_called()

    def test_point_count_mismatch_never_swaps_alias(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()
        loader.iter_repository_files.return_value = iter(["a.py"])
        loader.load_file_batch.return_value = [
            SimpleNamespace(text="a = 1", metadata={"path": "a.py"})
        ]
        splitter.split_documents.return_value = [MagicMock(), MagicMock()]
        point_ops.client.get_collection.return_value = SimpleNamespace(
            points_count=1
        )
        branch_mgr.get_branch_point_count.return_value = 5
        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.alias_exists.return_value = True
        coll_mgr.collection_exists.return_value = True
        coll_mgr.resolve_alias.return_value = "active"

        indexer = RepositoryIndexer(
            config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader
        )
        with pytest.raises(
            RuntimeError,
            match="Pending collection point count is incomplete",
        ):
            indexer.index_repository(
                "/repo", "ws", "proj", "main", "abc123", "alias1",
                source_tree_sha256="a" * 64,
            )

        coll_mgr.atomic_assign_aliases.assert_not_called()
        stats_mgr.store_metadata.assert_not_called()
        coll_mgr.delete_collection.assert_called_with("pending")

    def test_target_branch_point_count_mismatch_never_swaps_alias(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()
        loader.iter_repository_files.return_value = iter(["a.py"])
        loader.load_file_batch.return_value = [
            SimpleNamespace(text="a = 1", metadata={"path": "a.py"})
        ]
        splitter.split_documents.return_value = [MagicMock()]
        point_ops.client.get_collection.return_value = SimpleNamespace(
            points_count=2
        )
        branch_mgr.get_branch_point_count.return_value = 0
        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.alias_exists.return_value = True
        coll_mgr.collection_exists.return_value = True
        coll_mgr.resolve_alias.return_value = "active"

        indexer = RepositoryIndexer(
            config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader
        )
        with pytest.raises(
            RuntimeError,
            match="Pending target-branch point count is incomplete",
        ):
            indexer.index_repository(
                "/repo", "ws", "proj", "main", "abc123", "alias1",
                source_tree_sha256="a" * 64,
            )

        coll_mgr.atomic_assign_aliases.assert_not_called()
        stats_mgr.store_metadata.assert_not_called()
        coll_mgr.delete_collection.assert_called_with("pending")

    def test_indexing_failure_cleans_up(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()

        loader.iter_repository_files.return_value = iter(["a.py"])

        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.alias_exists.return_value = False
        coll_mgr.collection_exists.return_value = False
        coll_mgr.resolve_alias.return_value = None

        loader.load_file_batch.side_effect = RuntimeError("disk error")

        indexer = RepositoryIndexer(config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader)

        with pytest.raises(RuntimeError, match="disk error"):
            indexer.index_repository("/repo", "ws", "proj", "main", "abc123", "alias1")

        coll_mgr.delete_collection.assert_called_with("pending")

    def test_rejected_vector_point_is_skipped_and_valid_index_is_published(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()
        loader.iter_repository_files.return_value = iter(["a.php"])
        loader.load_file_batch.return_value = [
            SimpleNamespace(text="<?php", metadata={"path": "a.php"})
        ]
        splitter.split_documents.return_value = [MagicMock(), MagicMock()]
        def persist_with_one_rejected_structural_record(nodes, *_args, **_kwargs):
            metadata = getattr(nodes[0], "metadata", None) if nodes else None
            if isinstance(metadata, dict) and metadata.get("generation_manifest"):
                return 1, 0
            return 1, max(0, len(nodes) - 1)

        point_ops.process_and_store_chunks.side_effect = (
            persist_with_one_rejected_structural_record
        )
        point_ops.client.get_collection.return_value = SimpleNamespace(
            points_count=2
        )
        branch_mgr.get_branch_point_count.return_value = 2
        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.alias_exists.return_value = True
        coll_mgr.collection_exists.return_value = True
        coll_mgr.resolve_alias.return_value = "active"

        indexer = RepositoryIndexer(config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader)

        result = indexer.index_repository(
            "/repo", "ws", "proj", "main", "abc123", "alias1",
            source_tree_sha256="a" * 64,
        )

        assert result.chunk_count == 1
        assert result.skipped_chunk_count == 1
        coll_mgr.atomic_assign_aliases.assert_called_once_with({"alias1": "pending"})
        stats_mgr.store_metadata.assert_called_once()

    def test_repository_architecture_is_streamed_and_indexed_as_context(self, tmp_path):
        from codecrow_plugins import (
            ArchitecturePacket,
            FileDisposition,
            GraphFact,
            PluginDiagnostic,
            RepositoryAnalysis,
        )

        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()
        source = tmp_path / "app" / "code" / "Acme" / "Module.php"
        source.parent.mkdir(parents=True)
        source.write_text("<?php class Module {}", encoding="utf-8")
        loader.iter_repository_files.return_value = iter(["app/code/Acme/Module.php"])
        document = SimpleNamespace(
            text="<?php class Module {}",
            metadata={"path": "app/code/Acme/Module.php"},
        )
        loader.load_file_batch.return_value = [document]
        splitter.split_documents.return_value = [MagicMock()]
        point_ops.client.get_collection.return_value = SimpleNamespace(points_count=4)
        branch_mgr.get_branch_point_count.return_value = 4
        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.alias_exists.return_value = False
        coll_mgr.collection_exists.return_value = False
        coll_mgr.resolve_alias.return_value = None

        capabilities = SimpleNamespace(
            repository_plugins=("php", "magento"),
            fingerprint="sha256:" + "0" * 64,
            descriptor_fingerprint="sha256:" + "1" * 64,
        )
        selector = MagicMock()
        selector.select.return_value = capabilities
        handle = MagicMock()
        handle.active = True
        handle.finish.return_value = (
            RepositoryAnalysis(packets=(ArchitecturePacket(
                plugin_id="magento",
                kind="magento-object-graph",
                key="global:Acme\\Module",
                paths=("app/code/Acme/Module.php", "app/code/Acme/etc/di.xml"),
                facts=(GraphFact(
                    "magento-object-resolution",
                    "Acme\\Api\\Contract",
                    "resolves-to",
                    "Acme\\Model\\Implementation",
                    "app/code/Acme/etc/di.xml",
                    related_paths=("app/code/Acme/Module.php",),
                ),),
            ),)),
            (PluginDiagnostic(
                code="magento-invalid-xml",
                message="Cannot parse etc/invalid.xml",
                plugin_id="magento",
                path="etc/invalid.xml",
                recoverable=True,
            ),),
        )
        runtime = MagicMock()
        runtime.file_disposition.return_value = FileDisposition.FULL
        runtime.start_repository_analysis.return_value = handle

        indexer = RepositoryIndexer(
            config,
            coll_mgr,
            branch_mgr,
            point_ops,
            stats_mgr,
            splitter,
            loader,
            plugin_catalog=MagicMock(),
            plugin_runtime=runtime,
            plugin_selector=selector,
        )
        result = indexer.index_repository(
            str(tmp_path), "ws", "proj", "main", "abc123", "alias1",
            source_tree_sha256="a" * 64,
        )

        ingested = handle.ingest.call_args.args[0]
        assert [(artifact.path, artifact.content) for artifact in ingested] == [
            ("app/code/Acme/Module.php", "<?php class Module {}"),
        ]
        architecture_nodes = next(
            invocation.args[0]
            for invocation in point_ops.process_and_store_chunks.call_args_list
            if invocation.args[0]
            and isinstance(invocation.args[0][0].metadata, dict)
            and invocation.args[0][0].metadata.get("architecture_context")
        )
        assert architecture_nodes[0].metadata["architecture_context"] is True
        assert architecture_nodes[0].metadata["architecture_paths"] == [
            "app/code/Acme/Module.php",
            "app/code/Acme/etc/di.xml",
        ]
        assert result.chunk_count == 3
        assert result.skipped_file_count == 1


# ─────────────────────────────────────────────────────────────
# architecture storage
# ─────────────────────────────────────────────────────────────
class TestArchitectureStorage:

    def test_large_plugin_context_is_split_into_bounded_source_nodes(self):
        from codecrow_plugins import RepositoryAnalysis, RepositoryContext

        capabilities = SimpleNamespace(
            repository_plugins=("generic",),
            fingerprint="sha256:" + "0" * 64,
            descriptor_fingerprint="sha256:" + "1" * 64,
        )
        path = "schema/api.graphqls"
        content = "a\n" * (ARCHITECTURE_SOURCE_PART_SIZE // 2) + "tail"

        nodes = RepositoryIndexer._repository_context_nodes(
            RepositoryAnalysis(contexts=(RepositoryContext(
                "generic", "schema", path, content,
            ),)),
            capabilities,
            "ws",
            "project",
            "main",
            "commit",
        )

        assert len(nodes) == 2
        assert all(node.metadata["path"] == path for node in nodes)
        assert all(len(node.text) <= ARCHITECTURE_SOURCE_PART_SIZE for node in nodes)
        assert [node.metadata["architecture_source_part"] for node in nodes] == [
            0, 1,
        ]
        assert all(
            node.metadata["architecture_source_parts"] == 2 for node in nodes
        )
        assert "".join(node.text for node in nodes) == content
        assert not any(node.metadata.get("source_manifest") for node in nodes)

    def test_plugin_context_suppression_matches_the_exact_stored_path(self):
        from codecrow_plugins import RepositoryAnalysis, RepositoryContext

        capabilities = SimpleNamespace(
            repository_plugins=("generic",),
            fingerprint="sha256:" + "0" * 64,
            descriptor_fingerprint="sha256:" + "1" * 64,
        )
        analysis = RepositoryAnalysis(contexts=tuple(sorted((
            RepositoryContext(
                "generic", "config", "etc/di.xml", "<config/>",
            ),
            RepositoryContext(
                "generic", "config", "vendor/etc/di.xml", "<vendor/>",
            ),
        ))))

        nodes = RepositoryIndexer._repository_context_nodes(
            analysis,
            capabilities,
            "ws",
            "project",
            "main",
            "commit",
            excluded_paths={"etc/di.xml"},
        )

        assert {node.metadata["path"] for node in nodes} == {
            "vendor/etc/di.xml"
        }
        assert [node.text for node in nodes] == ["<vendor/>"]
        assert nodes[0].metadata["architecture_source"] is True

    def test_packs_plugin_facts_into_bounded_architecture_nodes(self):
        from codecrow_plugins import ArchitecturePacket, GraphFact, RepositoryAnalysis

        packets = []
        for index in range(30):
            source_path = "app/code/Acme/Module/etc/di.xml"
            packets.append(ArchitecturePacket(
                plugin_id="magento",
                kind="magento-di",
                key=f"preference:{index:02d}",
                paths=(source_path,),
                facts=(GraphFact(
                    kind="magento-preference",
                    source=f"Acme\\Api\\Contract{index:02d}",
                    relation="resolves-to",
                    target=f"Acme\\Model\\Implementation{index:02d}",
                    path=source_path,
                ),),
            ))
        other_path = "app/code/Other/Module/etc/di.xml"
        packets.append(ArchitecturePacket(
            plugin_id="magento",
            kind="magento-di",
            key="preference:other",
            paths=(other_path,),
            facts=(GraphFact(
                kind="magento-preference",
                source="Other\\Api\\Contract",
                relation="resolves-to",
                target="Other\\Model\\Implementation",
                path=other_path,
            ),),
        ))
        capabilities = SimpleNamespace(
            repository_plugins=("php", "magento"),
            fingerprint="sha256:" + "0" * 64,
            descriptor_fingerprint="sha256:" + "1" * 64,
        )

        nodes = RepositoryIndexer._architecture_nodes(
            RepositoryAnalysis(packets=tuple(sorted(packets))),
            capabilities,
            "ws",
            "project",
            "main",
            "commit",
        )

        assert len(nodes) == 3
        acme_nodes = [
            node for node in nodes
            if node.metadata["architecture_source_path"]
            == "app/code/Acme/Module/etc/di.xml"
        ]
        assert len(acme_nodes) == 2
        assert [
            len(node.metadata["plugin_graph_facts"])
            for node in acme_nodes
        ] == [25, 5]
        assert all(
            node.metadata["architecture_context"] is True
            and "structural_relation" not in node.metadata
            for node in acme_nodes
        )
        assert len({
            node.metadata["architecture_key"] for node in acme_nodes
        }) == 2
        assert nodes[-1].metadata["architecture_source_path"] == other_path
# ─────────────────────────────────────────────────────────────
# _perform_atomic_swap
# ─────────────────────────────────────────────────────────────
class TestPerformAtomicSwap:

    def test_normal_swap(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()

        coll_mgr.read_alias_targets.return_value = {"alias1": "active"}

        indexer = RepositoryIndexer(config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader)
        old_targets = indexer._perform_atomic_swap(
            "alias1", "pending", ["alias1"]
        )

        coll_mgr.atomic_assign_aliases.assert_called_once_with({"alias1": "pending"})
        assert old_targets == {"alias1": "active"}
        coll_mgr.delete_collection.assert_not_called()

    def test_first_activation_has_no_rollback_target(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()

        coll_mgr.read_alias_targets.return_value = {"alias1": None}

        indexer = RepositoryIndexer(config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader)
        old_targets = indexer._perform_atomic_swap(
            "alias1", "pending", ["alias1"]
        )

        assert old_targets == {"alias1": None}
        coll_mgr.atomic_assign_aliases.assert_called_once_with({"alias1": "pending"})

    def test_metadata_failure_rolls_back_before_pending_collection_is_deleted(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()
        loader.iter_repository_files.return_value = iter(["a.php"])
        loader.load_file_batch.return_value = [
            SimpleNamespace(text="<?php", metadata={"path": "a.php"})
        ]
        splitter.split_documents.return_value = [MagicMock()]
        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.alias_exists.return_value = True
        coll_mgr.resolve_alias.return_value = "active"
        coll_mgr.read_alias_targets.return_value = {"alias1": "active"}
        point_ops.client.get_collection.return_value = SimpleNamespace(points_count=2)
        branch_mgr.get_branch_point_count.return_value = 2
        stats_mgr.store_metadata.side_effect = RuntimeError("metadata unavailable")

        indexer = RepositoryIndexer(config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader)
        with pytest.raises(RuntimeError, match="metadata unavailable"):
            indexer.index_repository(
                "/repo", "ws", "proj", "main", "abc123", "alias1",
                source_tree_sha256="a" * 64,
            )

        assert coll_mgr.atomic_assign_aliases.call_args_list == [
            call({"alias1": "pending"}),
            call({"alias1": "active"}),
        ]
        coll_mgr.delete_collection.assert_called_with("pending")

    def test_direct_collection_is_replaced_only_after_completed_build(self):
        config = _mock_config()
        coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader = _mock_components()
        loader.iter_repository_files.return_value = iter(["a.php"])
        loader.load_file_batch.return_value = [
            SimpleNamespace(text="<?php", metadata={"path": "a.php"})
        ]
        splitter.split_documents.return_value = [MagicMock()]
        coll_mgr.create_pending_collection.return_value = "pending"
        coll_mgr.alias_exists.return_value = False
        coll_mgr.collection_exists.return_value = True
        coll_mgr.physical_collection_exists.side_effect = (
            lambda name: name == "alias1"
        )
        coll_mgr.is_structural_collection.return_value = False
        coll_mgr.read_alias_targets.return_value = {"alias1": None}
        point_ops.client.get_collection.return_value = SimpleNamespace(
            points_count=2
        )
        branch_mgr.get_branch_point_count.return_value = 2

        indexer = RepositoryIndexer(config, coll_mgr, branch_mgr, point_ops, stats_mgr, splitter, loader)
        indexer.index_repository(
            "/repo", "ws", "proj", "main", "abc123", "alias1",
            source_tree_sha256="a" * 64,
        )

        coll_mgr.delete_collection.assert_called_once_with("alias1")
        coll_mgr.atomic_assign_aliases.assert_called_once_with({
            "alias1": "pending"
        })
