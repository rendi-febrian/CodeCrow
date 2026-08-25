"""
Branch-level operations for structural repository indices.

Counts branch-bound records in exact generation collections.
"""

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)


class BranchManager:
    """Read branch-level record counts from exact generation collections."""
    
    def __init__(self, client: QdrantClient):
        self.client = client
    
    def get_branch_point_count(
        self,
        collection_name: str,
        branch: str
    ) -> int:
        """Get the number of points for a specific branch."""
        try:
            result = self.client.count(
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
            return result.count
        except Exception as e:
            logger.error(f"Failed to get point count for branch '{branch}': {e}")
            return 0
