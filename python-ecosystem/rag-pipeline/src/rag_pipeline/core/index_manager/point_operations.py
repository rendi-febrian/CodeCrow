"""Create and persist structural-index records in Qdrant."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Dict, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ..documents import TextNode
from ..generation_manifest import (
    GENERATION_MEMBER_DIGEST_PAYLOAD_KEY,
    compute_generation_member_digest,
    verified_generation_member,
)

logger = logging.getLogger(__name__)

# Qdrant requires a vector for each point. This fixed marker carries no search
# meaning; all repository retrieval is payload-filtered and deterministic.
STORAGE_MARKER_VECTOR = [1.0]

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEARCH_METADATA_FIELDS = (
    "path",
    "primary_name",
    "symbol_names",
    "full_path",
    "namespace",
    "parent_class",
    "parent_context",
    "imports",
    "extends",
    "implements",
    "parent_types",
    "methods",
    "properties",
    "parameters",
    "return_type",
    "decorators",
    "calls",
    "referenced_types",
    "variables",
    "constants",
    "type_parameters",
    "architecture_identifiers",
    "symbol_qualified_name",
    "symbol_kind",
    "symbol_parents",
    "symbol_methods",
    "symbol_constructor_types",
)


class PointWriteInfrastructureError(RuntimeError):
    """Raised when Qdrant cannot persist structural-index records."""


@dataclass(frozen=True)
class PointWriteResult:
    successful: int = 0
    skipped_points: tuple[PointStruct, ...] = ()

    @property
    def failed(self) -> int:
        return len(self.skipped_points)


def normalize_search_terms(*values: object, limit: int = 2000) -> list[str]:
    """Return a bounded deterministic identifier/path/source term inventory."""
    terms: set[str] = set()

    def visit(value: object) -> None:
        if value is None or len(terms) >= limit:
            return
        if isinstance(value, str):
            for raw in _TOKEN_RE.findall(value):
                folded = raw.casefold()
                if folded != "_" and len(terms) < limit:
                    terms.add(folded)
                for part in _CAMEL_BOUNDARY_RE.split(raw):
                    part = part.casefold()
                    if part and part != "_" and len(terms) < limit:
                        terms.add(part)
                if len(terms) >= limit:
                    break
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
                if len(terms) >= limit:
                    break
            return
        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            for item in value:
                visit(item)
                if len(terms) >= limit:
                    break

    for value in values:
        visit(value)
        if len(terms) >= limit:
            break
    return sorted(terms)[:limit]


def _record_type(metadata: dict) -> str:
    if metadata.get("repository_generation_manifest"):
        return "repository_generation_manifest"
    if metadata.get("pr_overlay_generation_manifest"):
        return "pr_overlay_generation_manifest"
    if metadata.get("repository_snapshot"):
        return "repository_snapshot"
    if metadata.get("repository_facts_state"):
        return "repository_facts"
    if metadata.get("architecture_context"):
        return "architecture_fact"
    if metadata.get("architecture_source"):
        return "architecture_source"
    if metadata.get("symbol_definition"):
        return "symbol_definition"
    return "source_chunk"


class PointOperations:
    """Prepare structural records and write them to the payload store."""

    def __init__(
        self,
        client: QdrantClient,
        batch_size: int = 128,
        max_upsert_payload_bytes: int = 8 * 1024 * 1024,
        upsert_max_attempts: int = 3,
        upsert_retry_base_seconds: float = 0.25,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_upsert_payload_bytes <= 0:
            raise ValueError("max_upsert_payload_bytes must be positive")
        if upsert_max_attempts <= 0:
            raise ValueError("upsert_max_attempts must be positive")
        if upsert_retry_base_seconds < 0:
            raise ValueError("upsert_retry_base_seconds cannot be negative")
        self.client = client
        self.batch_size = batch_size
        self.max_upsert_payload_bytes = max_upsert_payload_bytes
        self.upsert_max_attempts = upsert_max_attempts
        self.upsert_retry_base_seconds = upsert_retry_base_seconds

    @staticmethod
    def generate_point_id(
        workspace: str,
        project: str,
        branch: str,
        path: str,
        chunk_index: int,
    ) -> str:
        key = f"{workspace}:{project}:{branch}:{path}:{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def prepare_chunks_for_storage(
        self,
        chunks: List[TextNode],
        workspace: str,
        project: str,
        branch: str,
    ) -> List[Tuple[str, TextNode]]:
        """Assign deterministic point identities within each repository path."""
        chunks_by_file: Dict[str, List[TextNode]] = {}
        for chunk in chunks:
            path = chunk.metadata.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError("structural record is missing path metadata")
            chunks_by_file.setdefault(path, []).append(chunk)

        indexed_at = datetime.now(timezone.utc).isoformat()
        prepared = []
        for path, file_chunks in chunks_by_file.items():
            for chunk_index, chunk in enumerate(file_chunks):
                storage_identity = chunk.metadata.get("storage_identity", path)
                point_id = self.generate_point_id(
                    workspace, project, branch, str(storage_identity), chunk_index
                )
                chunk.metadata["indexed_at"] = indexed_at
                prepared.append((point_id, chunk))
        return prepared

    @staticmethod
    def _search_terms(chunk: TextNode) -> list[str]:
        metadata = chunk.metadata
        if _record_type(metadata) in {
            "repository_generation_manifest",
            "pr_overlay_generation_manifest",
            "repository_snapshot",
            "repository_facts",
        }:
            return []
        selected = [metadata.get(key) for key in _SEARCH_METADATA_FIELDS]
        return normalize_search_terms(*selected, chunk.text)

    def create_points(
        self,
        chunk_data: List[Tuple[str, TextNode]],
    ) -> List[PointStruct]:
        """Create payload-bound Qdrant points with a non-semantic marker."""
        points = []
        for point_id, chunk in chunk_data:
            payload = {
                **chunk.metadata,
                "text": chunk.text,
                "structural_record_type": _record_type(chunk.metadata),
            }
            search_terms = self._search_terms(chunk)
            if search_terms:
                payload["search_terms"] = search_terms
            payload[GENERATION_MEMBER_DIGEST_PAYLOAD_KEY] = (
                compute_generation_member_digest(point_id, payload)
            )
            points.append(PointStruct(
                id=point_id,
                vector=STORAGE_MARKER_VECTOR,
                payload=payload,
            ))
        return points

    def verify_persisted_point_digests(
        self,
        collection_name: str,
        points: List[PointStruct],
    ) -> list[tuple[object, str]]:
        """Verify acknowledged payloads without reading marker vectors."""
        if not points:
            return []
        expected_ids = {str(point.id) for point in points}
        records = self.client.retrieve(
            collection_name=collection_name,
            ids=[point.id for point in points],
            with_payload=True,
            with_vectors=False,
        )
        if {str(record.id) for record in records} != expected_ids:
            raise RuntimeError("persisted point set is incomplete before sealing")
        return [verified_generation_member(record) for record in records]

    def upsert_points(
        self,
        collection_name: str,
        points: List[PointStruct],
    ) -> Tuple[int, int]:
        result = self.upsert_points_detailed(collection_name, points)
        return result.successful, result.failed

    def upsert_points_detailed(
        self,
        collection_name: str,
        points: List[PointStruct],
    ) -> PointWriteResult:
        successful = 0
        skipped_points: list[PointStruct] = []
        for offset in range(0, len(points), self.batch_size):
            result = self._upsert_resilient(
                collection_name,
                points[offset:offset + self.batch_size],
                batch_offset=offset,
            )
            successful += result.successful
            skipped_points.extend(result.skipped_points)
        return PointWriteResult(successful, tuple(skipped_points))

    @staticmethod
    def _serialized_point_size(point: PointStruct) -> int:
        payload = point.model_dump(mode="json", exclude_none=True)
        return len(json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8"))

    def _payload_bounded_batches(
        self,
        points: List[PointStruct],
    ) -> list[tuple[List[PointStruct], int]]:
        if not points:
            return []
        request_overhead = 256
        batches: list[tuple[List[PointStruct], int]] = []
        current: List[PointStruct] = []
        current_size = request_overhead
        for point in points:
            point_size = self._serialized_point_size(point) + 1
            if current and current_size + point_size > self.max_upsert_payload_bytes:
                batches.append((current, current_size))
                current = []
                current_size = request_overhead
            current.append(point)
            current_size += point_size
        if current:
            batches.append((current, current_size))
        return batches

    @staticmethod
    def _status_code(exception: Exception) -> int | None:
        status_code = getattr(exception, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(exception, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code if isinstance(status_code, int) else None

    @classmethod
    def _is_record_shape_failure(cls, exception: Exception | None) -> bool:
        if exception is None:
            return False
        message = str(exception).casefold()
        if any(marker in message for marker in (
            "collection not found",
            "doesn't exist",
            "does not exist",
            "api key",
            "authentication",
        )):
            return False
        if cls._status_code(exception) in {400, 413, 422}:
            return True
        return any(marker in message for marker in (
            "bad request",
            "payload too large",
            "request entity too large",
            "request too large",
            "validation error",
        ))

    @staticmethod
    def _point_label(point: PointStruct) -> str:
        payload = point.payload or {}
        return f"id={point.id} path={payload.get('path', '<unknown>')}"

    def _upsert_resilient(
        self,
        collection_name: str,
        points: List[PointStruct],
        *,
        batch_offset: int,
    ) -> PointWriteResult:
        if not points:
            return PointWriteResult()
        bounded_batches = self._payload_bounded_batches(points)
        if len(bounded_batches) > 1:
            successful = 0
            skipped: list[PointStruct] = []
            offset = batch_offset
            for batch, _estimated_bytes in bounded_batches:
                result = self._upsert_resilient(
                    collection_name, batch, batch_offset=offset
                )
                successful += result.successful
                skipped.extend(result.skipped_points)
                offset += len(batch)
            return PointWriteResult(successful, tuple(skipped))

        error = None
        for attempt in range(1, self.upsert_max_attempts + 1):
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )
                return PointWriteResult(successful=len(points))
            except Exception as exception:
                error = exception
                if self._is_record_shape_failure(exception):
                    break
                if attempt >= self.upsert_max_attempts:
                    raise PointWriteInfrastructureError(
                        "Qdrant structural storage is unavailable after "
                        f"{self.upsert_max_attempts} attempts"
                    ) from exception
                delay = self.upsert_retry_base_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Structural point write failed at offset %s "
                    "(%s points, attempt %s/%s); retrying in %.2fs: %s",
                    batch_offset, len(points), attempt,
                    self.upsert_max_attempts, delay, exception,
                )
                if delay:
                    time.sleep(delay)

        if len(points) == 1:
            logger.error(
                "Skipping one structural record rejected by Qdrant (%s): %s",
                self._point_label(points[0]), error,
            )
            return PointWriteResult(skipped_points=(points[0],))

        midpoint = len(points) // 2
        left = self._upsert_resilient(
            collection_name, points[:midpoint], batch_offset=batch_offset
        )
        right = self._upsert_resilient(
            collection_name,
            points[midpoint:],
            batch_offset=batch_offset + midpoint,
        )
        return PointWriteResult(
            successful=left.successful + right.successful,
            skipped_points=(*left.skipped_points, *right.skipped_points),
        )

    def process_and_store_chunks(
        self,
        chunks: List[TextNode],
        collection_name: str,
        workspace: str,
        project: str,
        branch: str,
        *,
        operation_id: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Prepare and persist one bounded structural-record batch."""
        operation_id = operation_id or uuid.uuid4().hex
        started = time.perf_counter()
        prepared = self.prepare_chunks_for_storage(
            chunks, workspace, project, branch
        )
        result = self.upsert_points_detailed(
            collection_name, self.create_points(prepared)
        )
        logger.info(
            "Structural point batch completed operation_id=%s records=%s "
            "successful=%s failed=%s duration_ms=%s qdrant_batch_size=%s",
            operation_id,
            len(prepared),
            result.successful,
            result.failed,
            round((time.perf_counter() - started) * 1000),
            self.batch_size,
        )
        return result.successful, result.failed
