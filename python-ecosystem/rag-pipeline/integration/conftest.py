"""
Shared fixtures for rag-pipeline **integration** tests.

These tests exercise the full FastAPI stack (routers → middleware → services)
with Qdrant and Redis mocked at the boundary.
"""
import os
import sys
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ── Ensure src/ is on sys.path ────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

# ── Environment variables ─────────────────────────────────────
os.environ.setdefault("SERVICE_SECRET", "test-secret-token")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")


async def _run_in_threadpool_inline(function, *args, **kwargs):
    """Execute mocked sync endpoints without an environment-owned worker pool."""
    return function(*args, **kwargs)


@pytest.fixture(scope="session")
def _mock_qdrant():
    """Mock qdrant_client so no real Qdrant connection is needed."""
    mock_qclient = MagicMock()
    mock_qclient.get_collections.return_value = MagicMock(collections=[])
    mock_qclient.collection_exists.return_value = False
    mock_qclient.create_collection.return_value = True
    mock_qclient.upsert.return_value = None
    mock_qclient.search.return_value = []
    mock_qclient.scroll.return_value = ([], None)
    mock_qclient.count.return_value = MagicMock(count=0)
    mock_qclient.delete_collection.return_value = True
    return mock_qclient


@pytest.fixture(scope="session")
def rag_app(_mock_qdrant):
    """
    Create the RAG FastAPI app with mocked services.

    The RAG app is a module-level singleton (not a factory), so we:
    1. Patch service constructors at source (RAGConfig, RAGIndexManager, etc.)
    2. Set the module-level globals that routers read via _get_singletons()
    3. Patch thread-pool execution so mocked endpoints stay deterministic
    """
    with patch("rag_pipeline.models.config.RAGConfig") as MockConfig, \
         patch("rag_pipeline.core.index_manager.RAGIndexManager") as MockIM, \
         patch("rag_pipeline.services.query_service.RAGQueryService") as MockQS, \
         patch(
             "fastapi.routing.run_in_threadpool",
             new=_run_in_threadpool_inline,
         ):

        mock_config = MagicMock()
        mock_config.qdrant_url = "http://localhost:6333"
        mock_config.max_file_size_bytes = 512 * 1024
        mock_config.max_files_per_index = 5000
        mock_config.max_chunks_per_index = 1_000_000
        mock_config.chunk_size = 8000
        mock_config.chunk_overlap = 200
        MockConfig.return_value = mock_config

        mock_im = MagicMock()
        mock_im._get_project_collection_name.return_value = "code_index_ws__project"
        mock_im._collection_manager.resolve_collection_target.return_value = (
            "code_index_ws__project_generation"
        )
        mock_im._collection_manager.require_structural_collection.return_value = (
            "code_index_ws__project_generation"
        )
        mock_im.get_revision_preflight.return_value = {
            "workspace": "ws1",
            "project": "proj1",
            "branch": "main",
            "commit": "revision-1",
            "generation_manifest_sha256": "a" * 64,
        }
        mutation_context = MagicMock()
        mutation_context.__enter__.return_value = SimpleNamespace(
            assert_owned=MagicMock()
        )
        mock_im.project_mutation.return_value = mutation_context
        mock_im.pr_overlay_mutation.return_value = mutation_context
        mock_im.qdrant_client = _mock_qdrant
        mock_im.splitter.split_documents.return_value = []
        mock_im.splitter.split_documents_resilient.side_effect = (
            lambda documents, capabilities=None: (
                mock_im.splitter.split_documents(
                    documents,
                    capabilities=capabilities,
                ),
                (),
            )
        )
        MockIM.return_value = mock_im

        mock_qs = MagicMock()
        mock_qs.search_code.return_value = [
            {
                "path": "a.py",
                "text": "class A: pass",
                "score": 180,
                "match_reasons": ["exact identifier"],
            }
        ]
        mock_qs.get_deterministic_context.return_value = {
            "files": [], "definitions": []
        }
        MockQS.return_value = mock_qs

        # Directly set module-level globals that routers access
        import rag_pipeline.api.api as api_module
        api_module.config = mock_config
        api_module.index_manager = mock_im
        api_module.query_service = mock_qs

        yield api_module.app


@pytest.fixture()
def client(rag_app):
    """httpx.AsyncClient bound to the RAG app."""
    import httpx
    transport = httpx.ASGITransport(app=rag_app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture()
def auth_headers():
    return {"x-service-secret": os.environ["SERVICE_SECRET"]}
