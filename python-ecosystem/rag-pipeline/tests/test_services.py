"""Unit coverage for deterministic structural code search."""

from unittest.mock import MagicMock

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, Record, VectorParams

import rag_pipeline.services.code_search as code_search_module
from rag_pipeline.services.code_search import CodeSearchMixin


def _service_with_points():
    client = QdrantClient(":memory:")
    client.create_collection(
        "generation",
        vectors_config=VectorParams(size=1, distance=Distance.DOT),
    )
    client.upsert("generation", points=[
        PointStruct(id=1, vector=[1.0], payload={
            "workspace": "ws",
            "project": "project",
            "branch": "main",
            "commit": "revision",
            "path": "src/UserService.py",
            "text": "class UserService: pass",
            "primary_name": "UserService",
            "symbol_names": ["UserService"],
            "search_terms": ["class", "service", "user", "userservice"],
            "structural_record_type": "structural_unit",
        }),
        PointStruct(id=2, vector=[1.0], payload={
            "workspace": "other-tenant",
            "project": "project",
            "branch": "main",
            "commit": "revision",
            "path": "src/UserService.py",
            "text": "class UserService: pass",
            "search_terms": ["userservice"],
            "structural_record_type": "structural_unit",
        }),
        PointStruct(id=3, vector=[1.0], payload={
            "workspace": "ws",
            "project": "project",
            "branch": "main",
            "commit": "revision",
            "path": "__analysis_state__/repository-facts/000000.state",
            "text": "userservice",
            "search_terms": ["userservice"],
            "structural_record_type": "repository_facts",
        }),
    ], wait=True)
    service = CodeSearchMixin()
    service.qdrant_client = client
    service._collection_or_alias_exists = MagicMock(return_value=True)
    service._observe_branches = MagicMock()
    return service


def _service_with_many_matching_points(count: int):
    client = QdrantClient(":memory:")
    client.create_collection(
        "generation",
        vectors_config=VectorParams(size=1, distance=Distance.DOT),
    )
    client.upsert(
        "generation",
        points=[
            PointStruct(id=index + 1, vector=[1.0], payload={
                "workspace": "ws",
                "project": "project",
                "branch": "main",
                "commit": "revision",
                "path": f"src/match_{index:03d}.py",
                "text": f"def shared_identifier_{index}(): pass",
                "primary_name": f"shared_identifier_{index}",
                "search_terms": ["shared", "identifier"],
                "structural_record_type": "structural_unit",
                "start_line": index + 1,
            })
            for index in range(count)
        ],
        wait=True,
    )
    service = CodeSearchMixin()
    service.qdrant_client = client
    service._collection_or_alias_exists = MagicMock(return_value=True)
    service._observe_branches = MagicMock()
    return service


def test_code_search_is_tenant_revision_bound_and_excludes_opaque_state():
    service = _service_with_points()
    response = service.search_code(
        query="UserService",
        workspace="ws",
        project="project",
        branch="main",
        repository_revision="revision",
        collection_target="generation",
        limit=None,
    )
    results = response["results"]

    assert [result["path"] for result in results] == ["src/UserService.py"]
    assert response["coverage"]["complete"] is True
    assert "score" not in results[0]
    assert "max_score" not in results[0]
    assert "_ordering_key" not in results[0]
    assert results[0]["match_reasons"] == [
        "exact indexed token: service, user, userservice",
        "identifier token: service, user, userservice",
        "path token: service, user, userservice",
        "exact identifier",
        "exact query substring in path",
        "exact query substring in source",
    ]


def test_code_search_returns_no_probability_for_missing_exact_token():
    service = _service_with_points()
    response = service.search_code(
        query="OrderRepository",
        workspace="ws",
        project="project",
        branch="main",
        repository_revision="revision",
        collection_target="generation",
        limit=None,
    )
    assert response["results"] == []
    assert response["coverage"]["complete"] is True


def test_code_search_omitted_limit_returns_every_matching_record_once():
    service = _service_with_many_matching_points(150)

    response = service.search_code(
        query="shared identifier",
        workspace="ws",
        project="project",
        branch="main",
        repository_revision="revision",
        collection_target="generation",
        limit=None,
    )

    assert response["coverage"]["complete"] is True
    assert response["coverage"]["matching_results"] == 150
    assert response["coverage"]["returned_results"] == 150
    assert len(response["results"]) == 150
    assert len({result["id"] for result in response["results"]}) == 150
    assert {result["path"] for result in response["results"]} == {
        f"src/match_{index:03d}.py" for index in range(150)
    }


def test_code_search_explicit_limit_is_observable_as_partial():
    service = _service_with_many_matching_points(20)

    response = service.search_code(
        query="shared identifier",
        workspace="ws",
        project="project",
        branch="main",
        repository_revision="revision",
        collection_target="generation",
        limit=8,
    )

    assert len(response["results"]) == 8
    assert response["coverage"]["complete"] is False
    assert response["coverage"]["matching_results"] == 20
    assert response["coverage"]["returned_results"] == 8
    assert response["coverage"]["partial_reasons"] == [
        "explicit_result_limit"
    ]


def test_code_search_deduplicates_point_identity_across_pages(monkeypatch):
    monkeypatch.setattr(
        code_search_module,
        "DETERMINISTIC_MAX_MATCHING_POINTS",
        5,
    )
    shared_payload = {
        "workspace": "ws",
        "project": "project",
        "branch": "main",
        "commit": "revision",
        "search_terms": ["shared"],
        "structural_record_type": "source_chunk",
    }
    first = Record(
        id=1,
        payload={**shared_payload, "path": "src/first.py", "text": "shared"},
    )
    second = Record(
        id=2,
        payload={**shared_payload, "path": "src/second.py", "text": "shared"},
    )
    third = Record(
        id=3,
        payload={**shared_payload, "path": "src/third.py", "text": "shared"},
    )
    service = CodeSearchMixin()
    service.qdrant_client = MagicMock()
    service.qdrant_client.scroll.side_effect = [
        ([first, second], "next"),
        ([second, third], None),
    ]
    service._collection_or_alias_exists = MagicMock(return_value=True)
    service._observe_branches = MagicMock()

    response = service.search_code(
        query="shared",
        workspace="ws",
        project="project",
        branch="main",
        repository_revision="revision",
        collection_target="generation",
        limit=None,
    )

    assert [result["id"] for result in response["results"]] == ["1", "2", "3"]
    assert response["coverage"]["complete"] is True
    assert response["coverage"]["matching_points_scanned"] == 4
    assert response["coverage"]["unique_matching_points"] == 3


def test_code_search_never_admits_more_than_global_point_budget(monkeypatch):
    monkeypatch.setattr(
        code_search_module,
        "DETERMINISTIC_MAX_MATCHING_POINTS",
        3,
    )
    payload = {
        "workspace": "ws",
        "project": "project",
        "branch": "main",
        "commit": "revision",
        "search_terms": ["shared"],
        "structural_record_type": "source_chunk",
        "text": "shared",
    }
    records = [
        Record(id=index, payload={**payload, "path": f"src/{index}.py"})
        for index in range(1, 5)
    ]
    service = CodeSearchMixin()
    service.qdrant_client = MagicMock()
    # The second page deliberately violates its requested one-record page size.
    # Admission still stops at the global matching-point budget.
    service.qdrant_client.scroll.side_effect = [
        (records[:2], "next"),
        (records[2:], "more"),
    ]
    service._collection_or_alias_exists = MagicMock(return_value=True)
    service._observe_branches = MagicMock()

    response = service.search_code(
        query="shared",
        workspace="ws",
        project="project",
        branch="main",
        repository_revision="revision",
        collection_target="generation",
        limit=None,
    )

    assert len(response["results"]) == 3
    assert response["coverage"]["matching_points_scanned"] == 3
    assert response["coverage"]["unique_matching_points"] == 3
    assert response["coverage"]["complete"] is False
    assert response["coverage"]["partial_reasons"] == [
        "global_matching_point_limit"
    ]
    assert [call.kwargs["limit"] for call in service.qdrant_client.scroll.call_args_list] == [
        3,
        1,
    ]


def test_code_search_batches_and_processes_every_query_term_without_duplication():
    terms = [f"q{index:03d}" for index in range(150)]
    client = QdrantClient(":memory:")
    client.create_collection(
        "generation",
        vectors_config=VectorParams(size=1, distance=Distance.DOT),
    )
    base_payload = {
        "workspace": "ws",
        "project": "project",
        "branch": "main",
        "commit": "revision",
        "structural_record_type": "source_chunk",
    }
    client.upsert("generation", points=[
        PointStruct(id=1, vector=[1.0], payload={
            **base_payload,
            "path": "src/重复_界.py",
            "text": "q000 q149 完整证据",
            # This point is returned by both disjoint term batches and must be
            # admitted only once by point identity.
            "search_terms": [terms[0], terms[-1]],
        }),
        PointStruct(id=2, vector=[1.0], payload={
            **base_payload,
            "path": "src/末尾_界.py",
            "text": "q149 尾部证据",
            "search_terms": [terms[-1]],
        }),
    ], wait=True)
    service = CodeSearchMixin()
    service.qdrant_client = client
    service.qdrant_client.scroll = MagicMock(wraps=client.scroll)
    service._collection_or_alias_exists = MagicMock(return_value=True)
    service._observe_branches = MagicMock()

    response = service.search_code(
        query=" ".join(terms),
        workspace="ws",
        project="project",
        branch="main",
        repository_revision="revision",
        collection_target="generation",
        limit=None,
    )

    assert response["coverage"] == {
        "complete": True,
        "partial_reasons": [],
        "matching_points_scanned": 3,
        "unique_matching_points": 2,
        "global_matching_point_limit": min(
            5000, code_search_module.DETERMINISTIC_MAX_MATCHING_POINTS
        ),
        "query_term_count": 150,
        "query_term_batch_size": 128,
        "query_term_batches": 2,
        "processed_query_term_batches": 2,
        "completed_query_term_batches": 2,
        "matching_results": 2,
        "returned_results": 2,
    }
    assert [result["id"] for result in response["results"]] == ["1", "2"]

    filtered_batches = []
    for call in service.qdrant_client.scroll.call_args_list:
        search_filter = call.kwargs["scroll_filter"]
        term_condition = next(
            condition
            for condition in search_filter.must
            if condition.key == "search_terms"
        )
        filtered_batches.append(list(term_condition.match.any))
    assert [len(batch) for batch in filtered_batches] == [128, 22]
    assert [term for batch in filtered_batches for term in batch] == terms


def test_code_search_reports_global_safety_before_unprocessed_term_batches(
    monkeypatch,
):
    monkeypatch.setattr(
        code_search_module,
        "DETERMINISTIC_MAX_MATCHING_POINTS",
        1,
    )
    terms = [f"q{index:03d}" for index in range(150)]
    record = Record(id=1, payload={
        "workspace": "ws",
        "project": "project",
        "branch": "main",
        "commit": "revision",
        "path": "src/完整_界.py",
        "text": terms[0],
        "search_terms": [terms[0]],
        "structural_record_type": "source_chunk",
    })
    service = CodeSearchMixin()
    service.qdrant_client = MagicMock()
    service.qdrant_client.scroll.return_value = ([record], None)
    service._collection_or_alias_exists = MagicMock(return_value=True)
    service._observe_branches = MagicMock()

    response = service.search_code(
        query=" ".join(terms),
        workspace="ws",
        project="project",
        branch="main",
        repository_revision="revision",
        collection_target="generation",
        limit=None,
    )

    assert response["coverage"]["complete"] is False
    assert response["coverage"]["partial_reasons"] == [
        "global_matching_point_limit"
    ]
    assert response["coverage"]["query_term_batches"] == 2
    assert response["coverage"]["processed_query_term_batches"] == 1
    assert response["coverage"]["completed_query_term_batches"] == 1
    service.qdrant_client.scroll.assert_called_once()
