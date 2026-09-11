"""
Google Vertex AI and Google AI Studio embedding wrapper for LlamaIndex.
Supports text-embedding-004 (Vertex AI, 768 dim) and gemini-embedding-001 (Google AI, 768/3072 dim).
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, List, Optional
import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding

from ..models.config import get_embedding_dim_for_model

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = float(os.getenv("GOOGLE_EMBEDDING_TIMEOUT", "120"))
DEFAULT_MAX_RETRIES = int(os.getenv("GOOGLE_EMBEDDING_MAX_RETRIES", "5"))
DEFAULT_MAX_CHARS = int(os.getenv("GOOGLE_EMBEDDING_MAX_CHARS", "10000"))
DEFAULT_BATCH_SIZE = int(os.getenv("GOOGLE_EMBEDDING_BATCH_SIZE", "15"))
DEFAULT_PACE_DELAY = float(os.getenv("GOOGLE_EMBEDDING_PACE_DELAY", "1.0"))


class EmbeddingError(Exception):
    """Raised when an embedding cannot be produced for a given text."""


class GoogleEmbedding(BaseEmbedding):
    """
    Custom embedding class for Google Vertex AI and Google AI Studio.
    """

    def __init__(
        self,
        mode: str = "google_ai",
        model: str = "gemini-embedding-001",
        service_account_path: Optional[str] = None,
        service_account_json: Optional[str] = None,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        timeout: float = DEFAULT_TIMEOUT,
        embed_batch_size: Optional[int] = None,
        expected_dim: Optional[int] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        pace_delay: float = DEFAULT_PACE_DELAY,
        **kwargs: Any
    ):
        norm_mode = (mode or "google_ai").lower()
        if embed_batch_size is None:
            embed_batch_size = DEFAULT_BATCH_SIZE

        super().__init__(embed_batch_size=embed_batch_size, **kwargs)

        chosen_model = model or ("gemini-embedding-001" if "google" in norm_mode else "text-embedding-004")
        if expected_dim is not None:
            embedding_dim = expected_dim
        else:
            embedding_dim = get_embedding_dim_for_model(chosen_model)

        object.__setattr__(self, '_mode', norm_mode)
        object.__setattr__(self, '_model', chosen_model)
        object.__setattr__(self, '_location', location)
        object.__setattr__(self, '_timeout', timeout)
        object.__setattr__(self, '_max_retries', max_retries)
        object.__setattr__(self, '_expected_dim', embedding_dim)
        object.__setattr__(self, '_api_key', api_key or os.getenv("GOOGLE_API_KEY", ""))
        object.__setattr__(self, '_batch_size', embed_batch_size)
        object.__setattr__(self, '_pace_delay', pace_delay)
        object.__setattr__(self, '_credentials', None)
        object.__setattr__(self, '_project_id', project_id)

        if self._mode == "vertex":
            from google.oauth2 import service_account
            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            if service_account_path and os.path.exists(service_account_path):
                creds = service_account.Credentials.from_service_account_file(
                    service_account_path, scopes=scopes
                )
                object.__setattr__(self, '_credentials', creds)
                with open(service_account_path) as f:
                    data = json.load(f)
                    object.__setattr__(self, '_project_id', self._project_id or data.get("project_id"))
            elif service_account_json:
                data = json.loads(service_account_json)
                creds = service_account.Credentials.from_service_account_info(
                    data, scopes=scopes
                )
                object.__setattr__(self, '_credentials', creds)
                object.__setattr__(self, '_project_id', self._project_id or data.get("project_id"))
            else:
                import google.auth
                creds, cred_project = google.auth.default(scopes=scopes)
                object.__setattr__(self, '_credentials', creds)
                object.__setattr__(self, '_project_id', self._project_id or cred_project)

        logger.info(
            f"GoogleEmbedding initialized: mode={self._mode}, model={self._model}, "
            f"dim={self._expected_dim}, batch_size={self._batch_size}"
        )

    def _get_vertex_token(self) -> str:
        from google.auth.transport.requests import Request
        if not self._credentials.valid:
            self._credentials.refresh(Request())
        return self._credentials.token

    def _get_embedding_endpoint(self) -> str:
        if self._mode == "vertex":
            return (
                f"https://{self._location}-aiplatform.googleapis.com/v1/projects/"
                f"{self._project_id}/locations/{self._location}/publishers/google/models/{self._model}:predict"
            )
        else:
            return f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:batchEmbedContents?key={self._api_key}"

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        res = self._get_text_embeddings([text])
        if res:
            return res[0]
        raise EmbeddingError("No embedding returned for single text")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        clean_texts = [t[:DEFAULT_MAX_CHARS].strip() or " " for t in texts]

        # Process in batches
        for i in range(0, len(clean_texts), self._batch_size):
            batch = clean_texts[i:i + self._batch_size]
            embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        if self._pace_delay > 0:
            time.sleep(self._pace_delay)

        for attempt in range(self._max_retries + 1):
            try:
                if self._mode == "vertex":
                    token = self._get_vertex_token()
                    headers = {
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    }
                    payload = {"instances": [{"content": t} for t in batch]}
                    endpoint = self._get_embedding_endpoint()

                    with httpx.Client(timeout=self._timeout) as client:
                        resp = client.post(endpoint, headers=headers, json=payload)
                        if resp.status_code == 429:
                            if len(batch) > 1:
                                mid = len(batch) // 2
                                logger.warning(f"Vertex 429 rate limit. Splitting batch of {len(batch)} -> {mid} + {len(batch)-mid}")
                                return self._embed_batch_with_retry(batch[:mid]) + self._embed_batch_with_retry(batch[mid:])
                            delay = 15.0 * (attempt + 1)
                            logger.warning(f"Vertex embedding 429 on single item (attempt {attempt+1}/{self._max_retries+1}), waiting {delay}s...")
                            time.sleep(delay)
                            continue
                        if resp.status_code >= 500:
                            resp.raise_for_status()
                        if resp.status_code == 400 and len(batch) > 1 and "token count" in resp.text:
                            mid = len(batch) // 2
                            return self._embed_batch_with_retry(batch[:mid]) + self._embed_batch_with_retry(batch[mid:])
                        if resp.status_code != 200:
                            raise EmbeddingError(f"Vertex embedding error {resp.status_code}: {resp.text}")
                        data = resp.json()
                        predictions = data.get("predictions", [])
                        return [p["embeddings"]["values"] for p in predictions]
                else:
                    headers = {"Content-Type": "application/json"}
                    reqs = [{"model": f"models/{self._model}", "content": {"parts": [{"text": t}]}} for t in batch]
                    payload = {"requests": reqs}
                    endpoint = self._get_embedding_endpoint()

                    with httpx.Client(timeout=self._timeout) as client:
                        resp = client.post(endpoint, headers=headers, json=payload)
                        if resp.status_code == 429:
                            if len(batch) > 1:
                                mid = len(batch) // 2
                                logger.warning(f"Google AI 429 rate limit. Splitting batch of {len(batch)} -> {mid} + {len(batch)-mid}")
                                return self._embed_batch_with_retry(batch[:mid]) + self._embed_batch_with_retry(batch[mid:])
                            delay = 20.0 * (attempt + 1)
                            logger.warning(f"Google AI 429 on single item (attempt {attempt+1}/{self._max_retries+1}), waiting {delay}s...")
                            time.sleep(delay)
                            continue
                        if resp.status_code >= 500:
                            resp.raise_for_status()
                        if resp.status_code != 200:
                            raise EmbeddingError(f"Google AI embedding error {resp.status_code}: {resp.text}")
                        data = resp.json()
                        return [emb["values"] for emb in data.get("embeddings", [])]

            except Exception as e:
                if attempt < self._max_retries:
                    delay = 3.0 * (attempt + 1)
                    logger.warning(f"Google embedding error (attempt {attempt+1}/{self._max_retries+1}), retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"Google embedding fatal error after {self._max_retries+1} attempts: {e}")
                    raise EmbeddingError(f"Google embedding failed: {e}") from e

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self._get_text_embeddings, texts)
