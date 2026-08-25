"""Tests for structural payload preparation and marker storage."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rag_pipeline.core.documents import TextNode
from rag_pipeline.core.generation_manifest import (
    GENERATION_MEMBER_DIGEST_PAYLOAD_KEY,
    verified_generation_member,
)
from rag_pipeline.core.index_manager.point_operations import (
    STORAGE_MARKER_VECTOR,
    PointOperations,
    normalize_search_terms,
)


def _client():
    client = QdrantClient(":memory:")
    client.create_collection(
        "records",
        vectors_config=VectorParams(size=1, distance=Distance.DOT),
    )
    return client


def test_search_terms_split_identifiers_paths_and_single_character_names():
    assert normalize_search_terms("src/UserService.py x") == [
        "py", "service", "src", "user", "userservice", "x"
    ]


def test_search_terms_bound_identifier_length():
    identifier = "identifier_" + ("x" * 180)

    terms = normalize_search_terms(identifier)

    assert identifier not in terms
    assert terms
    assert all(len(term) <= 128 for term in terms)


def test_search_terms_apply_the_payload_cap_deterministically():
    records = [
        f"identifier_{index:04d}_unicode界"
        for index in range(2_105)
    ]
    terms = normalize_search_terms(
        records
    )

    assert len(terms) == 2_000
    assert terms[0] == "identifier_0000_unicode"
    assert terms[-1] == "identifier_1999_unicode"

    node = TextNode(text="", metadata={"symbol_names": records})
    assert PointOperations._search_terms(node) == terms


def test_create_points_stores_raw_text_metadata_and_fixed_marker():
    client = _client()
    operations = PointOperations(client)
    node = TextNode(
        text="class UserService: pass",
        metadata={
            "path": "src/UserService.py",
            "structural_unit": True,
            "primary_name": "UserService",
            "workspace": "ws",
            "project": "project",
            "branch": "main",
            "commit": "revision",
        },
    )
    prepared = operations.prepare_chunks_for_storage(
        [node], "ws", "project", "main"
    )
    point = operations.create_points(prepared)[0]

    assert point.vector == STORAGE_MARKER_VECTOR
    assert point.payload["text"] == "class UserService: pass"
    assert point.payload["structural_record_type"] == "source_chunk"
    assert "userservice" in point.payload["search_terms"]
    assert len(point.payload[GENERATION_MEMBER_DIGEST_PAYLOAD_KEY]) == 64


def test_opaque_state_is_not_a_code_search_candidate():
    operations = PointOperations(_client())
    point = operations.create_points(operations.prepare_chunks_for_storage(
        [TextNode(text='{"paths": []}', metadata={
            "path": "__analysis_state__/repository-facts/000000.state",
            "repository_facts_state": True,
        })],
        "ws",
        "project",
        "main",
    ))[0]

    assert point.payload["structural_record_type"] == "repository_facts"
    assert "search_terms" not in point.payload


def test_legacy_source_part_metadata_is_stored_as_an_ordinary_source_chunk():
    operations = PointOperations(_client())
    point = operations.create_points(operations.prepare_chunks_for_storage(
        [TextNode(text="secretLiteral UserService", metadata={
            "path": "src/UserService.py",
            "source_part": True,
            "source_content_sha256": "a" * 64,
            "source_part_index": 0,
            "source_part_count": 1,
            "storage_identity": "source-part:src/UserService.py:0",
        })],
        "ws",
        "project",
        "main",
    ))[0]

    assert point.payload["structural_record_type"] == "source_chunk"
    assert {"secretliteral", "userservice"} <= set(
        point.payload["search_terms"]
    )


def test_process_and_store_verifies_payload_only_digest():
    client = _client()
    operations = PointOperations(client, batch_size=1)
    nodes = [
        TextNode(text="function first", metadata={
            "path": "a.py", "structural_unit": True,
        }),
        TextNode(text="function second", metadata={
            "path": "b.py", "structural_unit": True,
        }),
    ]
    successful, failed = operations.process_and_store_chunks(
        nodes, "records", "ws", "project", "main"
    )

    assert (successful, failed) == (2, 0)
    records, _ = client.scroll(
        "records", with_payload=True, with_vectors=False, limit=10
    )
    assert len(records) == 2
    assert all(verified_generation_member(record) for record in records)


def test_storage_identity_keeps_symbol_record_distinct_from_source_path():
    operations = PointOperations(_client())
    source = TextNode(text="class User", metadata={
        "path": "src/user.py", "structural_unit": True,
    })
    symbol = TextNode(text="class User", metadata={
        "path": "src/user.py",
        "symbol_definition": True,
        "storage_identity": "symbol\\0User\\0class\\0src/user.py\\01",
    })
    prepared = operations.prepare_chunks_for_storage(
        [source, symbol], "ws", "project", "main"
    )
    assert len({point_id for point_id, _ in prepared}) == 2
