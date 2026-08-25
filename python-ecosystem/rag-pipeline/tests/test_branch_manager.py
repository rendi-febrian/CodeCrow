"""Tests for exact-generation branch record counting."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_pipeline.core.index_manager.branch_manager import BranchManager


def test_branch_point_count_uses_exact_branch_filter():
    client = MagicMock()
    client.count.return_value = SimpleNamespace(count=42)

    assert BranchManager(client).get_branch_point_count("generation", "main") == 42

    call = client.count.call_args
    assert call.kwargs["collection_name"] == "generation"
    condition = call.kwargs["count_filter"].must[0]
    assert condition.key == "branch"
    assert condition.match.value == "main"


def test_branch_point_count_fails_open_to_zero():
    client = MagicMock()
    client.count.side_effect = RuntimeError("unavailable")

    assert BranchManager(client).get_branch_point_count("generation", "main") == 0
