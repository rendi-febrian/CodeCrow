"""Integration coverage for structural query endpoints."""

import pytest


SEARCH_REQUEST = {
    "query": "authentication handler",
    "workspace": "ws1",
    "project": "proj1",
    "branch": "main",
    "repository_revision": "revision-1",
    "repository_generation_manifest_sha256": "a" * 64,
    "collection_target": "generation-target",
    "limit": 5,
}


@pytest.mark.asyncio
async def test_code_search(client, auth_headers):
    response = await client.post(
        "/query/code-search",
        json=SEARCH_REQUEST,
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["score"] == 180


@pytest.mark.asyncio
async def test_code_search_no_auth(client):
    response = await client.post("/query/code-search", json=SEARCH_REQUEST)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_code_search_requires_revision(client, auth_headers):
    request = dict(SEARCH_REQUEST)
    request.pop("repository_revision")
    response = await client.post(
        "/query/code-search", json=request, headers=auth_headers
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_deterministic_context(client, auth_headers):
    response = await client.post("/query/deterministic", json={
        "workspace": "ws1",
        "project": "proj1",
        "branches": ["main"],
        "file_paths": ["src/module.py"],
        "limit_per_file": 5,
        "base_revision": "revision-1",
        "base_generation_manifest_sha256": "a" * 64,
        "collection_target": "generation-target",
    }, headers=auth_headers)
    assert response.status_code == 200
    assert "context" in response.json()


@pytest.mark.asyncio
async def test_deterministic_context_no_auth(client):
    response = await client.post("/query/deterministic", json={
        "workspace": "ws1",
        "project": "proj1",
        "branches": ["main"],
        "file_paths": ["x.py"],
        "base_revision": "revision-1",
        "base_generation_manifest_sha256": "a" * 64,
        "collection_target": "generation-target",
    })
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_code_search_service_error(client, auth_headers):
    import rag_pipeline.api.api as api_module

    original = api_module.query_service.search_code.side_effect
    api_module.query_service.search_code.side_effect = RuntimeError("oops")
    try:
        response = await client.post(
            "/query/code-search",
            json=SEARCH_REQUEST,
            headers=auth_headers,
        )
        assert response.status_code == 500
    finally:
        api_module.query_service.search_code.side_effect = original
