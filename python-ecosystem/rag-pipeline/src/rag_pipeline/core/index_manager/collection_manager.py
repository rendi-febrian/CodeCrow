"""
Qdrant collection and alias management utilities.

Handles collection creation, alias operations, and resolution.
"""

import logging
import os
import re
import threading
import time
import uuid
from typing import Callable, Mapping, Optional, List

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)
from qdrant_client.models import (
    Distance, VectorParams,
    CreateAlias, DeleteAlias, CreateAliasOperation, DeleteAliasOperation,
    PayloadSchemaType, TextIndexParams, TokenizerType
)

from ..exact_index import ExactIndexPreconditionError

logger = logging.getLogger(__name__)


class CollectionManager:
    """Manages structural payload collections and aliases in Qdrant."""

    def __init__(self, client: QdrantClient):
        self.client = client
        self._payload_indexes_ensured: set[str] = set()
        self._payload_indexes_in_progress: set[str] = set()
        self._payload_index_condition = threading.Condition()
    
    @staticmethod
    def _has_storage_marker_schema(collection_info) -> bool:
        vectors = collection_info.config.params.vectors
        if isinstance(vectors, Mapping):
            return False
        distance = getattr(vectors, "distance", None)
        distance_value = getattr(distance, "value", distance)
        return (
            getattr(vectors, "size", None) == 1
            and str(distance_value).casefold() == "dot"
        )

    def is_structural_collection(self, collection_name: str) -> bool:
        """Return whether a collection/alias has the fixed marker schema."""
        physical = self.resolve_collection_target(collection_name)
        if physical is None:
            return False
        return self._has_storage_marker_schema(
            self.client.get_collection(physical)
        )

    def require_structural_collection(self, collection_name: str) -> str:
        """Reject mutations of pre-structural or otherwise incompatible data."""
        physical = self.resolve_collection_target(collection_name)
        if physical is None:
            raise ExactIndexPreconditionError(
                "structural repository collection is unavailable"
            )
        if not self._has_storage_marker_schema(
            self.client.get_collection(physical)
        ):
            raise ExactIndexPreconditionError(
                "repository collection predates structural payload storage; "
                "run a full repository index to publish a replacement generation"
            )
        return physical

    def create_pending_collection(
        self,
        base_name: str,
        *,
        operation_id: Optional[str] = None,
    ) -> str:
        """Create an unpublished collection for atomic index activation."""
        # Pending collections can be created by different workers or processes.
        # Timestamp + operation ownership lets the janitor distinguish a live
        # build from an expired orphan without touching another worker's work.
        for _ in range(3):
            token = re.sub(
                r"[^a-fA-F0-9]",
                "",
                operation_id or uuid.uuid4().hex,
            )[:32] or uuid.uuid4().hex[:32]
            pending_name = (
                f"{base_name}_pending_{int(time.time())}_{token}_"
                f"{uuid.uuid4().hex[:8]}"
            )
            logger.info(f"Creating pending collection: {pending_name}")
            if self._create_collection(pending_name):
                # Pending names are deliberately unique and short lived; do
                # not retain every generation in the process-wide cache.
                self._ensure_payload_indexes(pending_name)
                return pending_name
            logger.warning(
                "Pending collection name %s already exists; generating another",
                pending_name,
            )
        raise RuntimeError(
            f"Unable to allocate a unique pending collection for {base_name}"
        )

    def _create_collection(self, collection_name: str) -> bool:
        """Create one physical collection, accepting only a proven create race."""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1,
                    distance=Distance.DOT,
                    on_disk=True,
                ),
                on_disk_payload=True,
            )
            return True
        except UnexpectedResponse as exception:
            if (
                exception.status_code == 409
                and self.physical_collection_exists(collection_name)
            ):
                return False
            raise

    def physical_collection_exists(self, collection_name: str) -> bool:
        """Check a physical collection name without treating aliases as matches."""
        collections = self.client.get_collections().collections
        return any(collection.name == collection_name for collection in collections)

    @staticmethod
    def _payload_index_specs():
        """Fields used by bounded tenant, revision, and PR filters."""
        return (
            ("path", PayloadSchemaType.KEYWORD),
            ("branch", PayloadSchemaType.KEYWORD),
            ("workspace", PayloadSchemaType.KEYWORD),
            ("project", PayloadSchemaType.KEYWORD),
            ("commit", PayloadSchemaType.KEYWORD),
            ("primary_name", PayloadSchemaType.KEYWORD),
            ("search_terms", PayloadSchemaType.KEYWORD),
            ("structural_record_type", PayloadSchemaType.KEYWORD),
            ("architecture_paths", PayloadSchemaType.KEYWORD),
            ("architecture_group", PayloadSchemaType.KEYWORD),
            ("snapshot_plugin", PayloadSchemaType.KEYWORD),
            ("snapshot_kind", PayloadSchemaType.KEYWORD),
            ("pr", PayloadSchemaType.BOOL),
            ("pr_number", PayloadSchemaType.INTEGER),
            ("repository_generation_manifest", PayloadSchemaType.BOOL),
            ("generation_manifest_sha256", PayloadSchemaType.KEYWORD),
        )

    def ensure_payload_indexes(self, collection_name: str) -> None:
        """Repair required indexes once per physical collection and process.

        Collections created before a field was introduced are repaired on
        first use after restart. Failures remain fail-open for the current
        operation and retry on a later request instead of being cached.
        """
        with self._payload_index_condition:
            while collection_name in self._payload_indexes_in_progress:
                self._payload_index_condition.wait()
            if collection_name in self._payload_indexes_ensured:
                return
            self._payload_indexes_in_progress.add(collection_name)

        successful = False
        try:
            successful = self._ensure_payload_indexes(collection_name)
        finally:
            with self._payload_index_condition:
                self._payload_indexes_in_progress.discard(collection_name)
                if successful:
                    self._payload_indexes_ensured.add(collection_name)
                self._payload_index_condition.notify_all()

    def _existing_payload_index_types(
        self,
        collection_name: str,
    ) -> Optional[dict]:
        """Read schemas once so existing fields need no write request."""
        try:
            payload_schema = getattr(
                self.client.get_collection(collection_name),
                "payload_schema",
                {},
            )
        except Exception as exception:
            logger.warning(
                "Deferring payload index repair on %s because its schema "
                "could not be inspected: %s",
                collection_name,
                exception,
            )
            return None
        if not isinstance(payload_schema, Mapping):
            return {}
        return {
            field_name: getattr(index_info, "data_type", index_info)
            for field_name, index_info in payload_schema.items()
        }

    def _ensure_payload_indexes(self, collection_name: str) -> bool:
        """Create payload indexes for efficient filtering on common fields."""
        successful = True
        failures = []
        existing_types = self._existing_payload_index_types(collection_name)
        if existing_types is None:
            return False
        for field_name, field_schema in self._payload_index_specs():
            if existing_types.get(field_name) == field_schema:
                continue
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                )
            except ResponseHandlingException as exception:
                successful = False
                failures.append((field_name, exception))
                break
            except UnexpectedResponse as exception:
                successful = False
                failures.append((field_name, exception))
                if exception.status_code in {401, 403, 404, 408, 429} or (
                    exception.status_code >= 500
                ):
                    break
            except Exception as exception:
                successful = False
                failures.append((field_name, exception))
        if successful:
            logger.info("Payload indexes ensured for %s", collection_name)
        else:
            first_field, first_exception = failures[0]
            logger.warning(
                "Payload index repair failed for %s field(s) on %s; first "
                "failure was %s: %s",
                len(failures),
                collection_name,
                first_field,
                first_exception,
            )
            logger.info(
                "Payload index repair remains incomplete for %s; it will be "
                "retried on a later use",
                collection_name,
            )
        return successful
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            # A direct target can be recreated in the same process. Wait for
            # concurrent repair, then invalidate its receipt so the replacement
            # receives every required payload index.
            with self._payload_index_condition:
                while collection_name in self._payload_indexes_in_progress:
                    self._payload_index_condition.wait()
                self._payload_indexes_ensured.discard(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete collection {collection_name}: {e}")
            return False
    
    def get_collection_names(self) -> List[str]:
        """Get all collection names."""
        collections = self.client.get_collections().collections
        return [c.name for c in collections]
    
    # Alias operations
    
    def resolve_collection_target(self, collection_name: str) -> Optional[str]:
        """Resolve an alias or direct collection without hiding backend errors.

        Mutation leases use this strict resolver so a transient alias lookup
        failure cannot be mistaken for a direct collection.
        """
        aliases = self.client.get_aliases().aliases
        matching_aliases = [
            alias.collection_name
            for alias in aliases
            if alias.alias_name == collection_name
        ]
        if len(matching_aliases) > 1:
            raise RuntimeError(
                f"collection alias '{collection_name}' has multiple targets"
            )
        if matching_aliases:
            return matching_aliases[0]

        collections = self.client.get_collections().collections
        if collection_name in {collection.name for collection in collections}:
            return collection_name
        return None
    
    def read_alias_targets(self, alias_names: List[str]) -> dict[str, Optional[str]]:
        """Read several alias targets from one consistent Qdrant response."""
        requested = list(dict.fromkeys(name for name in alias_names if name))
        aliases = {
            alias.alias_name: alias.collection_name
            for alias in self.client.get_aliases().aliases
        }
        return {name: aliases.get(name) for name in requested}

    def atomic_assign_aliases(
        self,
        assignments: Mapping[str, Optional[str]],
    ) -> None:
        """Atomically point a set of aliases at already-validated collections.

        Exact generation target mappings move in one Qdrant transaction.
        """
        desired = {
            alias_name: collection_name
            for alias_name, collection_name in assignments.items()
            if alias_name
        }
        if not desired:
            return
        current = self.read_alias_targets(list(desired))
        operations = []
        for alias_name, collection_name in desired.items():
            if current.get(alias_name) == collection_name:
                continue
            if current.get(alias_name) is not None:
                operations.append(
                    DeleteAliasOperation(
                        delete_alias=DeleteAlias(alias_name=alias_name)
                    )
                )
            if collection_name is not None:
                operations.append(
                    CreateAliasOperation(
                        create_alias=CreateAlias(
                            alias_name=alias_name,
                            collection_name=collection_name,
                        )
                    )
                )
        if not operations:
            return
        self.client.update_collection_aliases(
            change_aliases_operations=operations
        )
        logger.info(
            "Atomically assigned Qdrant aliases: %s",
            ", ".join(sorted(desired)),
        )
    
    def delete_alias(self, alias_name: str) -> bool:
        """Delete an alias."""
        try:
            self.client.delete_alias(alias_name)
            logger.info(f"Deleted alias: {alias_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete alias {alias_name}: {e}")
            return False
    
    def cleanup_expired_pending_collections(
        self,
        *,
        is_operation_active: Callable[[str], bool],
        min_age_seconds: Optional[int] = None,
    ) -> int:
        """Delete only timestamped, non-aliased pending collections with no lease."""
        if min_age_seconds is None:
            min_age_seconds = max(
                300,
                int(os.getenv("RAG_PENDING_COLLECTION_MAX_AGE_SECONDS", "21600")),
            )
        now = int(time.time())
        # Alias membership is a safety precondition: an unavailable Qdrant
        # response must escape to the lifecycle janitor, which owns the
        # transition-bounded outage/recovery diagnostic. Returning zero here
        # would make an outage look like a healthy empty cleanup and emit one
        # warning on every scheduled pass.
        aliased_targets = {
            alias.collection_name for alias in self.client.get_aliases().aliases
        }

        pattern = re.compile(
            r"_pending_(\d{10})_([a-fA-F0-9]{8,32})_[a-fA-F0-9]{8}$"
        )
        cleaned = 0
        for collection_name in self.get_collection_names():
            match = pattern.search(collection_name)
            if match is None or collection_name in aliased_targets:
                continue
            created_at = int(match.group(1))
            operation_id = match.group(2)
            if now - created_at < min_age_seconds:
                continue
            if is_operation_active(operation_id):
                continue
            logger.info(
                "Cleaning expired pending collection %s operation_id=%s age_seconds=%s",
                collection_name,
                operation_id,
                now - created_at,
            )
            if self.delete_collection(collection_name):
                cleaned += 1
        return cleaned
