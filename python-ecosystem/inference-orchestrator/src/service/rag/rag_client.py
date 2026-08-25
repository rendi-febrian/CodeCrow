"""Client for structural repository context and code search."""
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import httpx

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using %s", name, value, default)
        return default


def _http_error_detail(error: httpx.HTTPError) -> tuple[Optional[int], str]:
    if not isinstance(error, httpx.HTTPStatusError):
        detail = str(error).strip()
        if not detail:
            request = getattr(error, "request", None)
            request_target = (
                f" for {request.method} {request.url}"
                if request is not None
                else ""
            )
            detail = f"{type(error).__name__}{request_target}"
        return None, detail
    response = error.response
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = str(payload.get("detail") or payload.get("error") or "")
    except (ValueError, TypeError):
        detail = response.text.strip()
    return response.status_code, detail or str(error)


class RagClient:
    """Client for interacting with the RAG Pipeline API."""

    def __init__(self, base_url: Optional[str] = None, enabled: Optional[bool] = None):
        """
        Initialize RAG client.

        Args:
            base_url: RAG pipeline API URL (default from env RAG_API_URL)
            enabled: Whether RAG is enabled (default from env RAG_ENABLED)
        """
        self.base_url = base_url or os.environ.get("RAG_API_URL", "http://rag-pipeline:8001")
        self.enabled = enabled if enabled is not None else os.environ.get("RAG_ENABLED", "true").lower() == "true"
        self.timeout = 30.0
        self._client: Optional[httpx.AsyncClient] = None
        self._mutation_client: Optional[httpx.AsyncClient] = None
        self._service_secret = (
            os.environ.get("SERVICE_SECRET")
            or os.environ.get("CODECROW_RAG_API_SECRET", "")
        )
        self._cleanup_degraded = False

        if self.enabled:
            logger.info(f"RAG client initialized: {self.base_url}")
        else:
            logger.info("RAG client disabled")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get the query/health pool, isolated from long PR mutations."""
        if self._client is None or self._client.is_closed:
            headers = {}
            if self._service_secret:
                headers["x-service-secret"] = self._service_secret
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                headers=headers,
            )
        return self._client

    async def _get_mutation_client(self) -> httpx.AsyncClient:
        """Get the bounded PR overlay/cleanup connection pool."""
        if self._mutation_client is None or self._mutation_client.is_closed:
            headers = {}
            if self._service_secret:
                headers["x-service-secret"] = self._service_secret
            self._mutation_client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=4,
                    max_keepalive_connections=2,
                ),
                headers=headers,
            )
        return self._mutation_client
    
    async def close(self):
        """Close this instance's HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        if (
            self._mutation_client is not None
            and not self._mutation_client.is_closed
        ):
            await self._mutation_client.aclose()
            self._mutation_client = None

    def _record_cleanup_failure(self, detail: str) -> None:
        if not self._cleanup_degraded:
            logger.warning("RAG PR cleanup is degraded: %s", detail)
            self._cleanup_degraded = True
        else:
            logger.debug("RAG PR cleanup remains degraded: %s", detail)

    def _record_cleanup_success(self) -> None:
        if self._cleanup_degraded:
            logger.info("RAG PR cleanup recovered")
            self._cleanup_degraded = False

    async def search_code(
        self,
        query: str,
        workspace: str,
        project: str,
        branch: str,
        top_k: int = 8,
        repository_revision: Optional[str] = None,
        repository_generation_manifest_sha256: Optional[str] = None,
        collection_target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find code with deterministic lexical, symbol, and metadata matching.

        Search is optional for Ask. Transport, index, or binding failures are
        returned as an empty result set so the existing exact VCS MCP path can
        still answer questions when the repository search service is degraded.
        ``match_reasons`` explains the concrete fields that matched. Ordering
        weights remain an implementation detail and are never presented as
        relevance or confidence.
        """
        if not self.enabled:
            return {"results": []}

        try:
            payload = {
                "query": query,
                "workspace": workspace,
                "project": project,
                "branch": branch,
                "limit": top_k,
            }
            if repository_revision:
                payload["repository_revision"] = repository_revision
            if repository_generation_manifest_sha256:
                payload["repository_generation_manifest_sha256"] = (
                    repository_generation_manifest_sha256
                )
            if collection_target:
                payload["collection_target"] = collection_target

            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/query/code-search",
                json=payload
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            status_code, detail = _http_error_detail(e)
            logger.debug(
                "Code search failed: status=%s detail=%s",
                status_code or "transport-error",
                detail,
            )
            return {
                "status": "error",
                "status_code": status_code,
                "error": detail,
                "results": [],
            }
        except Exception as e:
            logger.debug("Unexpected error in code search: %s", e, exc_info=True)
            return {
                "status": "error",
                "status_code": None,
                "error": str(e),
                "results": [],
            }

    async def is_healthy(self) -> bool:
        """
        Check if RAG pipeline is healthy.

        Returns:
            True if RAG is enabled and healthy, False otherwise
        """
        if not self.enabled:
            return False

        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"RAG health check failed: {e}")
            return False

    async def get_deterministic_context(
        self,
        workspace: str,
        project: str,
        branches: List[str],
        file_paths: List[str],
        limit_per_file: Optional[int] = None,
        pr_number: Optional[int] = None,
        pr_changed_files: Optional[List[str]] = None,
        additional_identifiers: Optional[List[str]] = None,
        source_revision: Optional[str] = None,
        base_revision: Optional[str] = None,
        base_generation_manifest_sha256: Optional[str] = None,
        pr_generation_fingerprint: Optional[str] = None,
        pr_overlay_generation_manifest_sha256: Optional[str] = None,
        collection_target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get context using DETERMINISTIC metadata-based retrieval.
        
        Two-step process leveraging tree-sitter metadata:
        1. Get chunks for the changed file_paths
        2. Extract symbol_names/imports/extends from those chunks
        3. Find related definitions using extracted identifiers
        
        NO language-specific parsing needed - tree-sitter did it during indexing!
        Predictable: same input = same output.

        Args:
            workspace: Workspace identifier
            project: Project identifier
            branches: Branches to search (e.g., ['release/1.29', 'master'])
            file_paths: Changed file paths from diff
            limit_per_file: Optional explicit per-file result bound. Normal
                review retrieval leaves this unset and relies on exact
                structural selection plus the observable global scan limit.
            pr_number: If set, also search PR-indexed chunks and prefer them over branch data
            pr_changed_files: All files changed in the PR (for stale data replacement)

        Returns:
            Dict with chunks grouped by: changed_files, related_definitions
        """
        if not self.enabled:
            logger.debug("RAG disabled, returning empty deterministic context")
            return {"context": {"chunks": [], "changed_files": {}, "related_definitions": {}}}

        start_time = datetime.now()
        
        try:
            payload = {
                "workspace": workspace,
                "project": project,
                "branches": branches,
                "file_paths": file_paths,
            }
            if limit_per_file is not None:
                payload["limit_per_file"] = limit_per_file
            
            # Enable hybrid PR mode for deterministic lookup
            if pr_number:
                payload["pr_number"] = pr_number
            if pr_changed_files:
                payload["pr_changed_files"] = pr_changed_files
            if additional_identifiers:
                payload["additional_identifiers"] = additional_identifiers
            if source_revision:
                payload["source_revision"] = source_revision
            if base_revision:
                payload["base_revision"] = base_revision
            if base_generation_manifest_sha256:
                payload["base_generation_manifest_sha256"] = (
                    base_generation_manifest_sha256
                )
            if pr_generation_fingerprint:
                payload["pr_generation_fingerprint"] = (
                    pr_generation_fingerprint
                )
            if pr_overlay_generation_manifest_sha256:
                payload["pr_overlay_generation_manifest_sha256"] = (
                    pr_overlay_generation_manifest_sha256
                )
            if collection_target:
                payload["collection_target"] = collection_target

            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/query/deterministic",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Log timing and stats
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            context = result.get("context", {})
            chunk_count = len(context.get("chunks", []))
            logger.info(f"Deterministic RAG query completed in {elapsed_ms:.2f}ms, "
                       f"retrieved {chunk_count} chunks for {len(file_paths)} files")
            
            return result

        except httpx.HTTPError as e:
            status_code, detail = _http_error_detail(e)
            logger.debug(
                "Failed to retrieve deterministic context: status=%s detail=%s",
                status_code or "transport-error",
                detail,
            )
            return {
                "status": "error",
                "status_code": status_code,
                "error": detail,
            }
        except Exception as e:
            logger.debug(
                "Unexpected error in deterministic RAG query: %s",
                e,
                exc_info=True,
            )
            return {
                "status": "error",
                "status_code": None,
                "error": str(e),
            }

    # =========================================================================
    # PR File Indexing Methods (for PR-specific RAG layer)
    # =========================================================================

    async def index_pr_files(
        self,
        workspace: str,
        project: str,
        pr_number: int,
        branch: str,
        files: List[Dict[str, str]],
        base_branch: Optional[str] = None,
        source_revision: Optional[str] = None,
        base_revision: Optional[str] = None,
        repository_plugins: Optional[List[str]] = None,
        plugin_detection_evidence: Optional[Dict[str, List[str]]] = None,
        plugin_fingerprint: str = "sha256:" + "0" * 64,
        plugin_descriptor_fingerprint: str = "sha256:" + "0" * 64,
        base_generation_manifest_sha256: Optional[str] = None,
        collection_target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Index PR files into the main collection with PR-specific metadata.
        
        Files are indexed with metadata (pr=true, pr_number=X) to enable
        hybrid queries that prioritize PR data over branch data.
        
        An exact persisted generation may be reused by the RAG service. A
        changed generation replaces prior points only after preparation.

        Args:
            workspace: Workspace identifier
            project: Project identifier
            pr_number: PR number for metadata tagging
            branch: Source branch name
            files: List of {
                path: str,
                content: str,
                change_type: str,
                content_state: "complete" | "partial_diff",
            }. Partial diff content is identity/evidence only and is not
            eligible for source parsing or structural indexing.

        Returns:
            Dict with indexing status and chunk counts
        """
        if not self.enabled:
            logger.debug("RAG disabled, skipping PR file indexing")
            return {"status": "skipped", "chunks_indexed": 0}

        if not files:
            logger.debug("No files to index for PR")
            return {"status": "skipped", "chunks_indexed": 0}

        index_timeout = max(
            30.0,
            _env_float("REVIEW_PR_INDEX_TIMEOUT_SECONDS", 1200.0),
        )
        try:
            payload = {
                "workspace": workspace,
                "project": project,
                "pr_number": pr_number,
                "branch": branch,
                "base_branch": base_branch or branch,
                "source_revision": source_revision,
                "base_revision": base_revision,
                "repository_plugins": repository_plugins or [],
                "plugin_detection_evidence": plugin_detection_evidence or {},
                "plugin_fingerprint": plugin_fingerprint,
                "plugin_descriptor_fingerprint": plugin_descriptor_fingerprint,
                "files": files
            }
            if base_generation_manifest_sha256:
                payload["base_generation_manifest_sha256"] = (
                    base_generation_manifest_sha256
                )
            if collection_target:
                payload["collection_target"] = collection_target

            client = await self._get_mutation_client()
            response = await client.post(
                f"{self.base_url}/index/pr-files",
                json=payload,
                timeout=index_timeout,
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(
                "%s PR #%s overlay: %s chunks from %s files (%s partial)",
                "Reused" if result.get("status") == "reused" else "Indexed",
                pr_number,
                result.get("chunks_indexed", 0),
                result.get("files_processed", 0),
                len(result.get("partial_files") or ()),
            )
            return result

        except httpx.HTTPError as e:
            status_code, detail = _http_error_detail(e)
            logger.debug(
                "Failed to index PR files: status=%s detail=%s timeout=%.1fs",
                status_code or "transport-error",
                detail,
                index_timeout,
            )
            return {
                "status": "error",
                "status_code": status_code,
                "error": detail,
            }
        except Exception as e:
            logger.debug(
                "Unexpected error indexing PR files: %s",
                e,
                exc_info=True,
            )
            return {"status": "error", "error": str(e)}

    async def delete_pr_files(
        self,
        workspace: str,
        project: str,
        pr_number: int,
        collection_target: Optional[str] = None,
    ) -> bool:
        """
        Delete all indexed points for a specific PR.
        
        Called after analysis completes to clean up PR-specific data.

        Args:
            workspace: Workspace identifier
            project: Project identifier
            pr_number: PR number to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        if not self.enabled:
            return True

        try:
            client = await self._get_mutation_client()
            response = await client.delete(
                f"{self.base_url}/index/pr-files/{workspace}/{project}/{pr_number}",
                params=(
                    {"collection_target": collection_target}
                    if collection_target else None
                ),
            )
            response.raise_for_status()
            result = response.json()

            status = result.get("status")
            if status == "deleted":
                self._record_cleanup_success()
                logger.info("Deleted PR #%s indexed data", pr_number)
                return True
            if status == "skipped":
                self._record_cleanup_success()
                logger.info(
                    "PR #%s indexed-data cleanup was already complete: %s",
                    pr_number,
                    result.get("message") or "nothing to delete",
                )
                return True
            self._record_cleanup_failure(
                "PR #%s returned unexpected status %s"
                % (pr_number, status or "missing")
            )
            return False

        except httpx.HTTPError as e:
            status_code, detail = _http_error_detail(e)
            self._record_cleanup_failure(
                "status=%s detail=%s"
                % (status_code or "transport-error", detail)
            )
            return False
        except Exception as e:
            self._record_cleanup_failure(
                f"unexpected {type(e).__name__}: {e}"
            )
            return False
