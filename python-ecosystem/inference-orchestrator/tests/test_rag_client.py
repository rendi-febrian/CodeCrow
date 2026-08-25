"""
Unit tests for service.rag.rag_client — RagClient (all async methods).
"""
import logging

import pytest
import httpx
import respx
from unittest.mock import AsyncMock, patch, MagicMock
from service.rag.rag_client import RagClient


@pytest.fixture
def disabled_client():
    return RagClient(base_url="http://rag:8001", enabled=False)


@pytest.fixture
def enabled_client():
    return RagClient(base_url="http://rag:8001", enabled=True)


# ── Disabled client short-circuits ───────────────────────────

class TestRagClientDisabled:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_code_search_disabled(self, disabled_client):
        r = await disabled_client.search_code("q", "ws", "proj", "main")
        assert r == {"results": []}

    @pytest.mark.asyncio(loop_scope="function")
    async def test_is_healthy_disabled(self, disabled_client):
        assert await disabled_client.is_healthy() is False

    @pytest.mark.asyncio(loop_scope="function")
    async def test_deterministic_context_disabled(self, disabled_client):
        r = await disabled_client.get_deterministic_context("ws", "proj", ["main"], ["a.py"])
        assert "context" in r

    @pytest.mark.asyncio(loop_scope="function")
    async def test_index_pr_files_disabled(self, disabled_client):
        r = await disabled_client.index_pr_files("ws", "proj", 1, "main", [])
        assert r["status"] == "skipped"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_delete_pr_files_disabled(self, disabled_client):
        r = await disabled_client.delete_pr_files("ws", "proj", 1)
        assert r is True


# ── No-branch short-circuit ──────────────────────────────────

# ── Successful HTTP calls (mocked with respx) ───────────────

class TestRagClientSuccess:
    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_code_search_ok(self):
        route = respx.post("http://rag:8001/query/code-search").mock(
            return_value=httpx.Response(200, json={
                "results": [{
                    "path": "src/a.py",
                    "score": 12,
                    "match_reasons": ["symbol:authenticate"],
                }]
            })
        )
        c = RagClient(base_url="http://rag:8001", enabled=True)
        r = await c.search_code(
            "query",
            "ws",
            "proj",
            "main",
            repository_revision="abc123",
            repository_generation_manifest_sha256="receipt",
            collection_target="generation-collection",
        )
        assert len(r["results"]) == 1
        payload = route.calls.last.request.content.decode()
        assert '"repository_revision":"abc123"' in payload
        assert '"repository_generation_manifest_sha256":"receipt"' in payload
        assert '"collection_target":"generation-collection"' in payload
        assert '"limit":8' in payload
        assert r["results"][0]["match_reasons"] == ["symbol:authenticate"]
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_code_search_forwards_an_explicit_result_limit(self):
        route = respx.post("http://rag:8001/query/code-search").mock(
            return_value=httpx.Response(200, json={
                "results": [],
                "coverage": {
                    "complete": False,
                    "partial_reasons": ["explicit_result_limit"],
                },
            })
        )
        c = RagClient(base_url="http://rag:8001", enabled=True)

        response = await c.search_code(
            "query",
            "ws",
            "proj",
            "main",
            top_k=17,
            repository_revision="abc123",
            repository_generation_manifest_sha256="receipt",
            collection_target="generation-collection",
        )

        assert route.calls.last.request.url.path == "/query/code-search"
        assert route.calls.last.request.read()
        assert b'"limit":17' in route.calls.last.request.content
        assert response["coverage"]["complete"] is False
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_is_healthy_ok(self):
        respx.get("http://rag:8001/health").mock(return_value=httpx.Response(200))
        c = RagClient(base_url="http://rag:8001", enabled=True)
        assert await c.is_healthy() is True
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_get_deterministic_context_ok(self):
        route = respx.post("http://rag:8001/query/deterministic").mock(
            return_value=httpx.Response(200, json={
                "context": {"chunks": [{"text": "c"}], "changed_files": {}, "related_definitions": {}}
            })
        )
        c = RagClient(base_url="http://rag:8001", enabled=True)
        r = await c.get_deterministic_context("ws", "proj", ["main"], ["a.py"], pr_number=42)
        assert len(r["context"]["chunks"]) == 1
        assert b'"limit_per_file"' not in route.calls.last.request.content
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_get_deterministic_context_preserves_explicit_per_file_bound(self):
        route = respx.post("http://rag:8001/query/deterministic").mock(
            return_value=httpx.Response(200, json={
                "context": {"chunks": [], "changed_files": {}, "related_definitions": {}}
            })
        )
        c = RagClient(base_url="http://rag:8001", enabled=True)

        await c.get_deterministic_context(
            "ws",
            "proj",
            ["main"],
            ["a.py"],
            limit_per_file=7,
        )

        assert b'"limit_per_file":7' in route.calls.last.request.content
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_index_pr_files_ok(self):
        route = respx.post("http://rag:8001/index/pr-files").mock(
            return_value=httpx.Response(200, json={"status": "ok", "chunks_indexed": 5, "files_processed": 2})
        )
        c = RagClient(base_url="http://rag:8001", enabled=True)
        files = [{"path": "a.py", "content": "code", "change_type": "MODIFIED"}]
        r = await c.index_pr_files(
            "ws",
            "proj",
            1,
            "main",
            files,
            source_revision="head-commit",
            base_revision="base-commit",
            repository_plugins=["python", "fastapi"],
            plugin_detection_evidence={
                "python": ["extension:a.py"],
                "fastapi": ["file:requirements.txt"],
            },
        )
        assert r["chunks_indexed"] == 5
        assert route.calls.last.request.read()
        payload = route.calls.last.request.content.decode()
        assert '"plugin_detection_evidence"' in payload
        assert '"extension:a.py"' in payload
        assert '"source_revision":"head-commit"' in payload
        assert '"base_revision":"base-commit"' in payload
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_delete_pr_files_ok(self):
        respx.delete("http://rag:8001/index/pr-files/ws/proj/1").mock(
            return_value=httpx.Response(200, json={"status": "deleted"})
        )
        c = RagClient(base_url="http://rag:8001", enabled=True)
        assert await c.delete_pr_files("ws", "proj", 1) is True
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_delete_pr_files_uses_exact_generation_target(self):
        route = respx.delete("http://rag:8001/index/pr-files/ws/proj/1").mock(
            return_value=httpx.Response(200, json={"status": "deleted"})
        )
        c = RagClient(base_url="http://rag:8001", enabled=True)

        assert await c.delete_pr_files(
            "ws", "proj", 1, collection_target="cc_w1_p2_branch_generation"
        ) is True
        assert route.calls.last.request.url.params[
            "collection_target"
        ] == "cc_w1_p2_branch_generation"
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_delete_pr_files_treats_missing_collection_as_idempotent(self):
        respx.delete("http://rag:8001/index/pr-files/ws/proj/1").mock(
            return_value=httpx.Response(
                200,
                json={"status": "skipped", "message": "Collection does not exist"},
            )
        )
        c = RagClient(base_url="http://rag:8001", enabled=True)
        assert await c.delete_pr_files("ws", "proj", 1) is True
        await c.close()


# ── Error handling ───────────────────────────────────────────

class TestRagClientErrors:
    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_code_search_error(self):
        respx.post("http://rag:8001/query/code-search").mock(
            side_effect=httpx.ConnectError("fail")
        )
        c = RagClient(base_url="http://rag:8001", enabled=True)
        r = await c.search_code("q", "ws", "proj", "main")
        assert r["status"] == "error"
        assert r["status_code"] is None
        assert r["results"] == []
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_is_healthy_error(self):
        respx.get("http://rag:8001/health").mock(side_effect=Exception("down"))
        c = RagClient(base_url="http://rag:8001", enabled=True)
        assert await c.is_healthy() is False
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_deterministic_context_error(self):
        respx.post("http://rag:8001/query/deterministic").mock(
            return_value=httpx.Response(503)
        )
        c = RagClient(base_url="http://rag:8001", enabled=True)
        r = await c.get_deterministic_context("ws", "proj", ["main"], ["a.py"])
        assert r["status"] == "error"
        assert r["status_code"] == 503
        assert "context" not in r
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_deterministic_context_409_is_returned_without_client_warning(
        self,
        caplog,
    ):
        respx.post("http://rag:8001/query/deterministic").mock(
            return_value=httpx.Response(
                409,
                json={"detail": "requested PR overlay generation is unavailable"},
            )
        )
        c = RagClient(base_url="http://rag:8001", enabled=True)
        with caplog.at_level(logging.DEBUG, logger="service.rag.rag_client"):
            r = await c.get_deterministic_context(
                "ws",
                "proj",
                ["main"],
                ["a.py"],
            )

        assert r == {
            "status": "error",
            "status_code": 409,
            "error": "requested PR overlay generation is unavailable",
        }
        assert not any(
            record.levelno >= logging.WARNING
            for record in caplog.records
            if record.name == "service.rag.rag_client"
        )
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_index_pr_files_error(self, caplog):
        respx.post("http://rag:8001/index/pr-files").mock(
            return_value=httpx.Response(
                409,
                json={"detail": "target branch is missing plugin snapshots"},
            )
        )
        c = RagClient(base_url="http://rag:8001", enabled=True)
        with caplog.at_level(logging.DEBUG, logger="service.rag.rag_client"):
            r = await c.index_pr_files("ws", "proj", 1, "main", [{"path": "a.py", "content": "x", "change_type": "M"}])
        assert r["status"] == "error"
        assert r["status_code"] == 409
        assert r["error"] == "target branch is missing plugin snapshots"
        assert not any(
            record.levelno >= logging.WARNING
            for record in caplog.records
            if record.name == "service.rag.rag_client"
        )
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_delete_pr_files_error(self, caplog):
        respx.delete("http://rag:8001/index/pr-files/ws/proj/1").mock(
            return_value=httpx.Response(
                500,
                json={"detail": "cleanup backend timed out"},
            )
        )
        c = RagClient(base_url="http://rag:8001", enabled=True)
        with caplog.at_level(logging.WARNING, logger="service.rag.rag_client"):
            assert await c.delete_pr_files("ws", "proj", 1) is False
        assert "status=500 detail=cleanup backend timed out" in caplog.text
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_delete_pr_failure_logs_only_outage_transitions(self, caplog):
        route = respx.delete(
            "http://rag:8001/index/pr-files/ws/proj/1"
        ).mock(side_effect=[
            httpx.Response(500, json={"detail": "timed out"}),
            httpx.Response(500, json={"detail": "timed out"}),
            httpx.Response(200, json={"status": "deleted"}),
        ])
        c = RagClient(base_url="http://rag:8001", enabled=True)

        with caplog.at_level(logging.DEBUG, logger="service.rag.rag_client"):
            assert await c.delete_pr_files("ws", "proj", 1) is False
            assert await c.delete_pr_files("ws", "proj", 1) is False
            assert await c.delete_pr_files("ws", "proj", 1) is True

        assert route.call_count == 3
        warnings = [
            record for record in caplog.records
            if record.levelno == logging.WARNING
            and "RAG PR cleanup is degraded" in record.getMessage()
        ]
        assert len(warnings) == 1
        assert "RAG PR cleanup recovered" in caplog.text
        await c.close()


# ── Client lifecycle ─────────────────────────────────────────

class TestRagClientLifecycle:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_close_noop_when_no_client(self, enabled_client):
        await enabled_client.close()  # Should not raise

    @pytest.mark.asyncio(loop_scope="function")
    @respx.mock
    async def test_get_client_reuses(self):
        respx.get("http://rag:8001/health").mock(return_value=httpx.Response(200))
        c = RagClient(base_url="http://rag:8001", enabled=True)
        await c.is_healthy()
        client1 = c._client
        await c.is_healthy()
        client2 = c._client
        assert client1 is client2
        await c.close()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_query_and_mutation_connection_pools_are_isolated(self):
        c = RagClient(base_url="http://rag:8001", enabled=True)

        query_client = await c._get_client()
        mutation_client = await c._get_mutation_client()

        assert query_client is not mutation_client
        await c.close()
        assert query_client.is_closed
        assert mutation_client.is_closed

    @pytest.mark.asyncio(loop_scope="function")
    async def test_empty_files_index(self, enabled_client):
        r = await enabled_client.index_pr_files("ws", "proj", 1, "main", [])
        assert r["status"] == "skipped"
