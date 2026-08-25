import pytest

from service.rag import rag_mcp_server


class _FakeRagClient:
    def __init__(self):
        self.calls = []
        self.closed = False

    async def search_code(self, **kwargs):
        self.calls.append(kwargs)
        return {"results": [{"path": "src/example.py", "score": 1.0}]}

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_search_tool_uses_request_scoped_repository_binding(monkeypatch):
    fake = _FakeRagClient()
    monkeypatch.setattr(rag_mcp_server, "RagClient", lambda: fake)
    monkeypatch.setenv("CODECROW_RAG_MCP_WORKSPACE", "tenant")
    monkeypatch.setenv("CODECROW_RAG_MCP_PROJECT", "project")
    monkeypatch.setenv("CODECROW_RAG_MCP_BRANCH", "main")
    monkeypatch.setenv("CODECROW_RAG_MCP_REVISION", "abc123")
    monkeypatch.setenv("CODECROW_RAG_MCP_MANIFEST", "manifest")
    monkeypatch.setenv("CODECROW_RAG_MCP_COLLECTION_TARGET", "collection")

    result = await rag_mcp_server.search_repository_code("payment policy", 100)

    assert result["results"][0]["path"] == "src/example.py"
    assert fake.calls == [{
        "query": "payment policy",
        "workspace": "tenant",
        "project": "project",
        "branch": "main",
        "top_k": 20,
        "repository_revision": "abc123",
        "repository_generation_manifest_sha256": "manifest",
        "collection_target": "collection",
    }]
    assert fake.closed is True
