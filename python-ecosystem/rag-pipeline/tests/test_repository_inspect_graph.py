"""Repository-inspector coverage for packed plugin graph facts."""

from types import SimpleNamespace

from qdrant_client.models import FieldCondition

from rag_pipeline.api.models import RepositoryIndexFilters
from rag_pipeline.api.routers.inspect import (
    _architecture_paths,
    _build_graph,
    _build_qdrant_filter,
    _matches_post_filter,
    _to_graph_node,
)


def _point(point_id: str, payload: dict) -> SimpleNamespace:
    return SimpleNamespace(id=point_id, payload=payload)


def _architecture_payload() -> dict:
    return {
        "workspace": "acme",
        "project": "shop",
        "branch": "main",
        "path": "__analysis_architecture__/magento/packet.context",
        "text": "bounded packet text",
        "architecture_context": True,
        "architecture_plugin": "magento",
        "architecture_kind": "magento-di",
        "architecture_source_path": "app/code/Acme/etc/di.xml",
        "architecture_paths": [
            "app/code/Acme/etc/di.xml",
            "app/code/Acme/Model/Cart.php",
        ],
        "plugin_graph_facts": [{
            "kind": "magento-preference",
            "source": "Acme\\Api\\CartInterface",
            "relation": "resolves-to",
            "target": "Acme\\Model\\Cart",
            "path": "app/code/Acme/etc/di.xml",
            "line": 12,
            "related_paths": ["app/code/Acme/Model/Cart.php"],
            "attributes": {"area": "global"},
            "packetKey": "global:cart",
        }],
    }


def test_graph_node_preserves_the_packed_fact_payload():
    payload = _architecture_payload()

    node = _to_graph_node(_point("architecture", payload), detail=False)

    assert node["kind"] == "architecture_context"
    assert node["metadata"]["plugin_graph_facts"] == (
        payload["plugin_graph_facts"]
    )
    assert _architecture_paths(node) == [
        "app/code/Acme/etc/di.xml",
        "app/code/Acme/Model/Cart.php",
    ]
    assert "structural_relation" not in node["metadata"]


def test_packed_fact_links_to_matching_source_chunks_and_evidence_paths():
    architecture = _to_graph_node(
        _point("architecture", _architecture_payload())
    )
    implementation = _to_graph_node(_point("implementation", {
        "branch": "main",
        "path": "app/code/Acme/Model/Cart.php",
        "text": "class Cart implements CartInterface {}",
        "content_type": "functions_classes",
        "primary_name": "Cart",
        "symbol_names": ["Cart", "Acme\\Model\\Cart"],
        "start_line": 1,
        "end_line": 3,
    }))

    nodes, edges = _build_graph([architecture, implementation])

    assert any(
        edge["source"] == "architecture"
        and edge["target"] == "implementation"
        and edge["kind"] == "metadata_reference"
        for edge in edges
    )
    assert any(
        node.get("kind") == "file"
        and node.get("path") == "app/code/Acme/etc/di.xml"
        for node in nodes
    )
    assert all(node.get("kind") != "structural_relation" for node in nodes)


def test_inspection_filter_remains_tenant_project_and_branch_bounded():
    filters = RepositoryIndexFilters(
        branches=["main"],
        include_pr=False,
    )

    qdrant_filter = _build_qdrant_filter(filters, "acme", "shop")

    must_by_key = {
        condition.key: condition
        for condition in qdrant_filter.must
        if isinstance(condition, FieldCondition)
    }
    assert must_by_key["workspace"].match.value == "acme"
    assert must_by_key["project"].match.value == "shop"
    assert must_by_key["branch"].match.value == "main"
    assert any(
        condition.key == "pr" and condition.match.value is True
        for condition in qdrant_filter.must_not
    )


def test_post_filter_matches_packed_packet_text_and_paths():
    payload = _architecture_payload()

    assert _matches_post_filter(
        payload,
        RepositoryIndexFilters(
            file_query="__analysis_architecture__/magento",
            text_query="bounded packet",
        ),
    )
    assert not _matches_post_filter(
        payload,
        RepositoryIndexFilters(text_query="unrelated token"),
    )
