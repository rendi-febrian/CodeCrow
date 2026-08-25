"""Facade for deterministic repository-index queries."""
from ..models.config import RAGConfig
from .base import RAGQueryBase
from .code_search import CodeSearchMixin
from .deterministic_context import DeterministicContextMixin


class RAGQueryService(
    CodeSearchMixin,
    DeterministicContextMixin,
    RAGQueryBase
):
    """Structural code search plus deterministic context retrieval."""

    def __init__(self, config: RAGConfig, plugin_catalog=None):
        super().__init__(config, plugin_catalog=plugin_catalog)
