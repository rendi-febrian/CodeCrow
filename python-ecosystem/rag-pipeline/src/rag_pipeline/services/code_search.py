"""Deterministic structural code search over indexed Qdrant payloads."""

from __future__ import annotations

from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from .base import RAGQueryBase
from .deterministic_context import DETERMINISTIC_MAX_MATCHING_POINTS
from ..core.index_manager.point_operations import normalize_search_terms


_NON_CODE_RECORD_TYPES = (
    "repository_generation_manifest",
    "pr_overlay_generation_manifest",
    "repository_snapshot",
    "repository_facts",
)

# Qdrant payload filters remain compact while every normalized query term is
# processed. This is a request batching size, not an evidence limit.
CODE_SEARCH_TERM_BATCH_SIZE = 128


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if isinstance(item, str)]
    return []


def _metadata_terms(payload: dict, *keys: str) -> set[str]:
    values: list[str] = []
    for key in keys:
        values.extend(_string_values(payload.get(key)))
    return set(normalize_search_terms(values))


def _rank_candidate(payload: dict, query: str, query_terms: set[str]) -> dict:
    stored_terms = {
        value for value in payload.get("search_terms", [])
        if isinstance(value, str)
    }
    matched_terms = sorted(query_terms.intersection(stored_terms))
    identifier_terms = _metadata_terms(
        payload,
        "primary_name",
        "symbol_names",
        "full_path",
        "architecture_identifiers",
        "symbol_qualified_name",
        "symbol_parents",
        "symbol_methods",
    )
    path_terms = _metadata_terms(payload, "path")
    identifier_matches = sorted(query_terms.intersection(identifier_terms))
    path_matches = sorted(query_terms.intersection(path_terms))

    folded_query = query.strip().casefold()
    path = str(payload.get("path", ""))
    primary_name = str(
        payload.get("primary_name")
        or payload.get("symbol_qualified_name")
        or ""
    )
    text = str(payload.get("text", ""))
    exact_identifier = bool(
        folded_query
        and folded_query in {
            value.casefold()
            for value in _string_values(payload.get("symbol_names"))
            + [primary_name]
            if value
        }
    )
    path_substring = bool(folded_query and folded_query in path.casefold())
    text_substring = bool(folded_query and folded_query in text.casefold())

    coverage = len(matched_terms)
    reasons: list[str] = []
    if matched_terms:
        reasons.append("exact indexed token: " + ", ".join(matched_terms))
    if identifier_matches:
        reasons.append("identifier token: " + ", ".join(identifier_matches))
    if path_matches:
        reasons.append("path token: " + ", ".join(path_matches))
    if exact_identifier:
        reasons.append("exact identifier")
    if path_substring:
        reasons.append("exact query substring in path")
    if text_substring:
        reasons.append("exact query substring in source")

    return {
        "_ordering_key": (
            int(exact_identifier),
            coverage,
            len(identifier_matches),
            len(path_matches),
            int(path_substring),
            int(text_substring),
        ),
        "matched_terms": matched_terms,
        "query_term_count": len(query_terms),
        "match_reasons": reasons,
    }


class CodeSearchMixin:
    """Exact-token candidate lookup with deterministic local ranking."""

    def search_code(
        self: RAGQueryBase,
        *,
        query: str,
        workspace: str,
        project: str,
        branch: str,
        repository_revision: str,
        collection_target: str,
        limit: int | None,
    ) -> dict:
        ordered_query_terms = normalize_search_terms(query)
        query_terms = set(ordered_query_terms)
        if not ordered_query_terms:
            return {
                "results": [],
                "coverage": {"complete": True, "matching_points_scanned": 0},
            }
        if not self._collection_or_alias_exists(collection_target):
            return {
                "results": [],
                "coverage": {"complete": True, "matching_points_scanned": 0},
            }

        self._observe_branches(collection_target, [branch])
        term_batches = [
            ordered_query_terms[start:start + CODE_SEARCH_TERM_BATCH_SIZE]
            for start in range(0, len(ordered_query_terms), CODE_SEARCH_TERM_BATCH_SIZE)
        ]

        # Code-search responses may be rendered into a prompt, so keep the
        # global exact-match scan bounded even when the shared retrieval limit
        # is configured above its normal 5,000-point ceiling.  Qdrant point ID
        # is the storage identity: equal source text in distinct points must be
        # retained, while a point repeated across pages must only be ranked and
        # returned once.
        candidate_limit = min(5000, DETERMINISTIC_MAX_MATCHING_POINTS)
        candidates_by_id: dict[str, Any] = {}
        matching_points_scanned = 0
        pagination_stalled = False
        global_limit_reached = False
        processed_term_batches = 0
        completed_term_batches = 0
        for term_batch in term_batches:
            if matching_points_scanned >= candidate_limit:
                global_limit_reached = True
                break

            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="workspace", match=MatchValue(value=workspace)
                    ),
                    FieldCondition(
                        key="project", match=MatchValue(value=project)
                    ),
                    FieldCondition(
                        key="branch", match=MatchValue(value=branch)
                    ),
                    FieldCondition(
                        key="commit",
                        match=MatchValue(value=repository_revision),
                    ),
                    FieldCondition(
                        key="search_terms",
                        match=MatchAny(any=term_batch),
                    ),
                ],
                must_not=[
                    FieldCondition(key="pr", match=MatchValue(value=True)),
                    FieldCondition(
                        key="structural_record_type",
                        match=MatchAny(any=list(_NON_CODE_RECORD_TYPES)),
                    ),
                ],
            )
            processed_term_batches += 1
            offset = None
            seen_offsets = set()
            while True:
                # The configured safety bound counts every provider match,
                # including a point repeated by disjoint term batches.  This
                # prevents a multi-batch query from bypassing the global scan
                # admission merely because repeated point IDs are de-duplicated
                # after retrieval.
                remaining_scan = candidate_limit - matching_points_scanned
                if remaining_scan <= 0:
                    global_limit_reached = True
                    break
                requested_page_size = min(256, remaining_scan)
                page, next_offset = self.qdrant_client.scroll(
                    collection_name=collection_target,
                    scroll_filter=search_filter,
                    limit=requested_page_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                admitted_page = page[:requested_page_size]
                page_overflow = len(page) > len(admitted_page)
                matching_points_scanned += len(admitted_page)
                for point in admitted_page:
                    point_id = str(point.id)
                    if point_id not in candidates_by_id:
                        candidates_by_id[point_id] = point

                if page_overflow:
                    global_limit_reached = True
                    break
                if next_offset is None:
                    completed_term_batches += 1
                    break
                if not admitted_page:
                    pagination_stalled = True
                    break
                offset_key = repr(next_offset)
                if offset_key in seen_offsets:
                    pagination_stalled = True
                    break
                seen_offsets.add(offset_key)
                offset = next_offset
                if matching_points_scanned >= candidate_limit:
                    global_limit_reached = True
                    break

            if global_limit_reached:
                break

        candidates = list(candidates_by_id.values())

        ranked = []
        for point in candidates:
            payload = dict(point.payload or {})
            rank = _rank_candidate(payload, query, query_terms)
            if not rank["matched_terms"]:
                continue
            ranked.append({
                "id": str(point.id),
                "path": payload.get("path"),
                "text": payload.get("text", ""),
                "record_type": payload.get(
                    "structural_record_type", "source_chunk"
                ),
                "metadata": {
                    key: value
                    for key, value in payload.items()
                    if key not in {"text", "search_terms"}
                },
                **rank,
            })

        ranked.sort(key=lambda item: (
            *(-value for value in item["_ordering_key"]),
            str(item.get("path") or ""),
            int(item["metadata"].get("start_line", 0) or 0),
            item["id"],
        ))
        explicit_limit_reached = limit is not None and len(ranked) > limit
        results = ranked if limit is None else ranked[:limit]
        for result in results:
            result.pop("_ordering_key", None)
        partial_reasons = []
        if global_limit_reached:
            partial_reasons.append("global_matching_point_limit")
        if pagination_stalled:
            partial_reasons.append("pagination_stalled")
        if explicit_limit_reached:
            partial_reasons.append("explicit_result_limit")
        return {
            "results": results,
            "coverage": {
                "complete": not partial_reasons,
                "partial_reasons": partial_reasons,
                "matching_points_scanned": matching_points_scanned,
                "unique_matching_points": len(candidates),
                "global_matching_point_limit": candidate_limit,
                "query_term_count": len(ordered_query_terms),
                "query_term_batch_size": CODE_SEARCH_TERM_BATCH_SIZE,
                "query_term_batches": len(term_batches),
                "processed_query_term_batches": processed_term_batches,
                "completed_query_term_batches": completed_term_batches,
                "matching_results": len(ranked),
                "returned_results": len(results),
            },
        }
