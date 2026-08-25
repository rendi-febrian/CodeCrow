"""Additional structural RagClient HTTP integration tests via respx."""
import os
import json
import pytest
import respx
import httpx

os.environ.setdefault("RAG_ENABLED", "true")
os.environ.setdefault("RAG_API_URL", "http://rag-pipeline:8001")
os.environ.setdefault("SERVICE_SECRET", "test-secret-token")

from service.rag.rag_client import RagClient


@pytest.fixture
def rag_client():
    client = RagClient(base_url="http://rag-pipeline:8001", enabled=True)
    yield client


# ── get_deterministic_context ────────────────────────────────

@pytest.mark.asyncio(loop_scope="function")
@respx.mock
async def test_get_deterministic_context_success(rag_client):
    """Basic deterministic context retrieval."""
    route = respx.post("http://rag-pipeline:8001/query/deterministic").mock(
        return_value=httpx.Response(200, json={
            "context": {
                "chunks": [{"path": "a.py", "text": "class A:"}],
                "changed_files": {"a.py": [{"text": "class A:"}]},
                "related_definitions": {},
            }
        })
    )
    result = await rag_client.get_deterministic_context(
        workspace="ws", project="proj",
        branches=["main"],
        file_paths=["src/a.py"],
    )
    assert route.called
    payload = json.loads(route.calls[0].request.read())
    assert payload["branches"] == ["main"]
    assert payload["file_paths"] == ["src/a.py"]
    assert "context" in result
    await rag_client.close()


@pytest.mark.asyncio(loop_scope="function")
@respx.mock
async def test_get_deterministic_context_with_pr(rag_client):
    """Deterministic context with PR-specific parameters."""
    route = respx.post("http://rag-pipeline:8001/query/deterministic").mock(
        return_value=httpx.Response(200, json={
            "context": {"chunks": [], "changed_files": {}, "related_definitions": {}}
        })
    )
    result = await rag_client.get_deterministic_context(
        workspace="ws", project="proj",
        branches=["main", "develop"],
        file_paths=["auth.py"],
        pr_number=42,
        pr_changed_files=["auth.py", "login.py"],
        additional_identifiers=["verify_token"],
    )
    assert route.called
    payload = json.loads(route.calls[0].request.read())
    assert payload["pr_number"] == 42
    assert "verify_token" in payload["additional_identifiers"]
    await rag_client.close()


@pytest.mark.asyncio(loop_scope="function")
@respx.mock
async def test_get_deterministic_context_timeout(rag_client):
    """Timeout remains distinguishable from a valid empty context."""
    respx.post("http://rag-pipeline:8001/query/deterministic").mock(
        side_effect=httpx.ReadTimeout("timed out")
    )
    result = await rag_client.get_deterministic_context(
        workspace="ws", project="proj",
        branches=["main"],
        file_paths=["a.py"],
    )
    assert result["status"] == "error"
    assert result["status_code"] is None
    assert result["error"] == "timed out"
    assert "context" not in result
    await rag_client.close()


@pytest.mark.asyncio(loop_scope="function")
async def test_get_deterministic_context_disabled():
    """Disabled client returns empty deterministic context."""
    client = RagClient(enabled=False)
    result = await client.get_deterministic_context(
        workspace="ws", project="proj",
        branches=["main"],
        file_paths=["a.py"],
    )
    assert result["context"]["chunks"] == []
    await client.close()


# ── index_pr_files ───────────────────────────────────────────

@pytest.mark.asyncio(loop_scope="function")
@respx.mock
async def test_index_pr_files_success(rag_client):
    """Successful PR file indexing."""
    route = respx.post("http://rag-pipeline:8001/index/pr-files").mock(
        return_value=httpx.Response(200, json={
            "status": "indexed",
            "pr_number": 42,
            "files_processed": 2,
            "chunks_indexed": 10,
            "chunks_failed": 0,
        })
    )
    result = await rag_client.index_pr_files(
        workspace="ws", project="proj", pr_number=42, branch="feat",
        files=[
            {"path": "a.py", "content": "x=1", "change_type": "ADDED"},
            {"path": "b.py", "content": "y=2", "change_type": "MODIFIED"},
        ],
    )
    assert route.called
    assert result["status"] == "indexed"
    assert result["chunks_indexed"] == 10
    await rag_client.close()


@pytest.mark.asyncio(loop_scope="function")
@respx.mock
async def test_index_pr_files_empty_files(rag_client):
    """Empty files list → skipped without HTTP call."""
    result = await rag_client.index_pr_files(
        workspace="ws", project="proj", pr_number=1, branch="b",
        files=[],
    )
    assert result["status"] == "skipped"
    await rag_client.close()


@pytest.mark.asyncio(loop_scope="function")
async def test_index_pr_files_disabled():
    """Disabled client → skipped."""
    client = RagClient(enabled=False)
    result = await client.index_pr_files(
        workspace="ws", project="proj", pr_number=1, branch="b",
        files=[{"path": "a.py", "content": "x=1", "change_type": "ADDED"}],
    )
    assert result["status"] == "skipped"
    await client.close()


@pytest.mark.asyncio(loop_scope="function")
@respx.mock
async def test_index_pr_files_server_error(rag_client):
    """Server error → graceful error status."""
    respx.post("http://rag-pipeline:8001/index/pr-files").mock(
        return_value=httpx.Response(500, json={"detail": "boom"})
    )
    result = await rag_client.index_pr_files(
        workspace="ws", project="proj", pr_number=1, branch="b",
        files=[{"path": "a.py", "content": "x=1", "change_type": "ADDED"}],
    )
    assert result["status"] == "error"
    await rag_client.close()


@pytest.mark.asyncio(loop_scope="function")
@respx.mock
async def test_index_pr_files_connection_error(rag_client):
    """Connection error → graceful error status."""
    respx.post("http://rag-pipeline:8001/index/pr-files").mock(
        side_effect=httpx.ConnectError("refused")
    )
    result = await rag_client.index_pr_files(
        workspace="ws", project="proj", pr_number=1, branch="b",
        files=[{"path": "a.py", "content": "x=1", "change_type": "ADDED"}],
    )
    assert result["status"] == "error"
    await rag_client.close()


# ── delete_pr_files ──────────────────────────────────────────

@pytest.mark.asyncio(loop_scope="function")
@respx.mock
async def test_delete_pr_files_success(rag_client):
    """Successful PR file deletion."""
    route = respx.delete("http://rag-pipeline:8001/index/pr-files/ws/proj/42").mock(
        return_value=httpx.Response(200, json={"status": "deleted", "pr_number": 42})
    )
    result = await rag_client.delete_pr_files(
        workspace="ws", project="proj", pr_number=42,
    )
    assert route.called
    assert result is True
    await rag_client.close()


@pytest.mark.asyncio(loop_scope="function")
@respx.mock
async def test_delete_pr_files_not_found(rag_client):
    """An already-absent overlay is an idempotent cleanup success."""
    respx.delete("http://rag-pipeline:8001/index/pr-files/ws/proj/99").mock(
        return_value=httpx.Response(200, json={"status": "skipped"})
    )
    result = await rag_client.delete_pr_files(
        workspace="ws", project="proj", pr_number=99,
    )
    assert result is True
    await rag_client.close()


@pytest.mark.asyncio(loop_scope="function")
@respx.mock
async def test_delete_pr_files_server_error(rag_client):
    """Server error → False."""
    respx.delete("http://rag-pipeline:8001/index/pr-files/ws/proj/1").mock(
        return_value=httpx.Response(500, json={"detail": "error"})
    )
    result = await rag_client.delete_pr_files(
        workspace="ws", project="proj", pr_number=1,
    )
    assert result is False
    await rag_client.close()


@pytest.mark.asyncio(loop_scope="function")
async def test_delete_pr_files_disabled():
    """Disabled client → True (no-op)."""
    client = RagClient(enabled=False)
    result = await client.delete_pr_files(
        workspace="ws", project="proj", pr_number=1,
    )
    assert result is True
    await client.close()
