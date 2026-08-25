"""
Index statistics and metadata operations.
"""

import logging
from datetime import datetime, timezone
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from ...models.config import IndexStats
from ...utils.utils import make_namespace

logger = logging.getLogger(__name__)


class StatsManager:
    """Manages index statistics and metadata."""
    
    def __init__(self, client: QdrantClient):
        self.client = client
    
    def get_branch_stats(
        self,
        workspace: str,
        project: str,
        branch: str,
        collection_name: str
    ) -> IndexStats:
        """Get statistics for one branch in an exact repository generation."""
        namespace = make_namespace(workspace, project, branch)

        try:
            count_result = self.client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="branch",
                            match=MatchValue(value=branch)
                        )
                    ]
                )
            )
            chunk_count = count_result.count

            return IndexStats(
                namespace=namespace,
                document_count=0,
                chunk_count=chunk_count,
                last_updated=datetime.now(timezone.utc).isoformat(),
                workspace=workspace,
                project=project,
                branch=branch
            )
        except Exception:
            return IndexStats(
                namespace=namespace,
                document_count=0,
                chunk_count=0,
                last_updated="",
                workspace=workspace,
                project=project,
                branch=branch
            )
    
    def store_metadata(
        self,
        workspace: str,
        project: str,
        branch: str,
        commit: str,
        document_count: int,
        chunk_count: int
    ) -> None:
        """Store/log metadata for an indexing operation."""
        namespace = make_namespace(workspace, project, branch)
        
        metadata = {
            "namespace": namespace,
            "workspace": workspace,
            "project": project,
            "branch": branch,
            "commit": commit,
            "document_count": document_count,
            "chunk_count": chunk_count,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        
        logger.info(f"Indexed {namespace}: {document_count} docs, {chunk_count} chunks")
