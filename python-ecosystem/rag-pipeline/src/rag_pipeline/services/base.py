"""Shared Qdrant infrastructure for structural repository queries."""

from typing import List
import logging

from qdrant_client import QdrantClient

from ..models.config import RAGConfig
from ..core.index_representation import (
    index_representation_fingerprint,
    observe_branch_representation,
)
from ..core.index_manager.collection_manager import CollectionManager

logger = logging.getLogger(__name__)


class RAGQueryBase:
    """Tenant-scoped structural query infrastructure."""

    def __init__(self, config: RAGConfig, plugin_catalog=None):
        self.config = config
        self.plugin_catalog = plugin_catalog
        self.index_representation_fingerprint = index_representation_fingerprint(
            config
        )
        self._observed_branch_cache: set[tuple[str, str]] = set()
        timeout = getattr(config, "qdrant_timeout_seconds", 30)
        if isinstance(timeout, bool) or not isinstance(timeout, int):
            timeout = 30
        self.qdrant_client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key or None,
            timeout=timeout,
        )

    def _observe_branches(
        self,
        collection_name: str,
        branches: List[str],
    ) -> None:
        for branch in dict.fromkeys(branches):
            cache_key = (collection_name, branch)
            if cache_key in self._observed_branch_cache:
                continue
            exists = observe_branch_representation(
                self.qdrant_client,
                collection_name,
                branch,
                expected_fingerprint=self.index_representation_fingerprint,
            )
            if exists:
                self._observed_branch_cache.add(cache_key)

    def _collection_or_alias_exists(self, name: str) -> bool:
        try:
            return CollectionManager(
                self.qdrant_client
            ).is_structural_collection(name)
        except Exception as exception:
            logger.debug(
                "Structural collection probe unavailable for %s: %s",
                name,
                exception,
            )
            return False

    def close(self) -> None:
        self.qdrant_client.close()
