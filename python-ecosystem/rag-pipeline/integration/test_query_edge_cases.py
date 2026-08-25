"""Edge cases for deterministic structural context retrieval."""

import pytest


@pytest.mark.asyncio
async def test_deterministic_with_pr_number(client, auth_headers):
    response = await client.post("/query/deterministic", json={
        "workspace": "ws1",
        "project": "proj1",
        "branches": ["main"],
        "file_paths": ["src/auth.py", "src/login.py"],
        "limit_per_file": 10,
        "pr_number": 42,
        "pr_changed_files": ["src/auth.py"],
        "additional_identifiers": ["verify_token", "User"],
        "base_revision": "revision-1",
        "base_generation_manifest_sha256": "a" * 64,
        "collection_target": "generation-target",
    }, headers=auth_headers)
    assert response.status_code == 200
    assert "context" in response.json()


@pytest.mark.asyncio
async def test_deterministic_service_error(client, auth_headers):
    import rag_pipeline.api.api as api_module

    original = api_module.query_service.get_deterministic_context.side_effect
    api_module.query_service.get_deterministic_context.side_effect = RuntimeError(
        "fail"
    )
    try:
        response = await client.post("/query/deterministic", json={
            "workspace": "ws1",
            "project": "proj1",
            "branches": ["main"],
            "file_paths": ["a.py"],
            "base_revision": "revision-1",
            "base_generation_manifest_sha256": "a" * 64,
            "collection_target": "generation-target",
        }, headers=auth_headers)
        assert response.status_code == 500
    finally:
        api_module.query_service.get_deterministic_context.side_effect = original


@pytest.mark.asyncio
async def test_deterministic_multiple_branches(client, auth_headers):
    response = await client.post("/query/deterministic", json={
        "workspace": "ws1",
        "project": "proj1",
        "branches": ["main", "develop", "release/1.0"],
        "file_paths": ["src/core.py"],
    }, headers=auth_headers)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_deterministic_empty_file_paths(client, auth_headers):
    response = await client.post("/query/deterministic", json={
        "workspace": "ws1",
        "project": "proj1",
        "branches": ["main"],
        "file_paths": [],
        "base_revision": "revision-1",
        "base_generation_manifest_sha256": "a" * 64,
        "collection_target": "generation-target",
    }, headers=auth_headers)
    assert response.status_code == 200
