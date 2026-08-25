"""CodeCrow structural repository indexing and retrieval."""

from .models.config import RAGConfig, IndexStats
from .core.index_manager import RAGIndexManager
from .services.query_service import RAGQueryService
from .core.loader import DocumentLoader
from .core.splitter import ASTCodeSplitter
from .utils.utils import make_namespace, detect_language_from_path

__all__ = [
    "RAGConfig",
    "IndexStats",
    "RAGIndexManager",
    "RAGQueryService",
    "DocumentLoader",
    "ASTCodeSplitter",
    "make_namespace",
    "detect_language_from_path",
]
