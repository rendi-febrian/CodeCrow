"""Neutral selection of exact structural evidence for Stage 1 prompts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from service.review.orchestrator.context_helpers import rag_retrieval_identity
from utils.path_identity import normalize_repository_path


logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %s", name, value, default)
        return default


DETERMINISTIC_RAG_MAX_CHUNKS = max(
    1,
    _env_int("REVIEW_DETERMINISTIC_RAG_MAX_CHUNKS", 80),
)


def _unwrap_rag_context(
    response: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    context = response.get("context")
    return context if isinstance(context, dict) else response


def _flatten_deterministic_context(
    deterministic_response: Optional[Dict[str, Any]],
    max_chunks: int = DETERMINISTIC_RAG_MAX_CHUNKS,
    reviewed_paths: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Flatten only exact structural RAG evidence into prompt chunks.

    The API's grouped views are the authority. Unclassified raw chunks are not
    review context because they do not carry a proved relationship type.
    """
    det_context = _unwrap_rag_context(deterministic_response)
    if not det_context:
        return []

    flattened: List[Dict[str, Any]] = []
    seen = set()
    explicit_reviewed_paths = {
        normalized
        for path in reviewed_paths or ()
        for normalized in (normalize_repository_path(path),)
        if normalized
    }

    def content_digest(value: Any) -> str:
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()

    def add_chunk(chunk: Any, source_group: str, group_key: str = "") -> None:
        if not isinstance(chunk, dict):
            return
        text = chunk.get("text") or chunk.get("content") or ""
        metadata = chunk.get("metadata") or {}
        path = metadata.get("path") or chunk.get("path") or chunk.get("file_path") or ""
        content_key = rag_retrieval_identity(chunk)
        if content_key in seen:
            return
        seen.add(content_key)

        merged = dict(chunk)
        merged.setdefault("text", text)
        merged.setdefault("content", text)
        merged.setdefault("metadata", metadata)
        merged.setdefault("file_path", path)
        merged.setdefault("path", path)
        # Preserve the freshness authority from the exact retrieval payload.
        # Without this, PR-scoped architecture packets are mislabeled as branch
        # data and the stale-evidence guard removes them before prompt assembly.
        merged["_source"] = (
            "pr_indexed" if metadata.get("pr") is True else "deterministic"
        )
        merged["_match_type"] = source_group
        if group_key:
            merged["definition_name"] = group_key
        flattened.append(merged)

    grouped_sources = (
        ("architecture_relation", det_context.get("architecture_context", {})),
        ("architecture_related", det_context.get("architecture_related", {})),
        ("changed_file", det_context.get("changed_files", {})),
        ("definition", det_context.get("related_definitions", {})),
    )
    for source_group, grouped in grouped_sources:
        if isinstance(grouped, dict):
            for group_key, chunks in grouped.items():
                for chunk in chunks or []:
                    add_chunk(chunk, source_group, str(group_key))

    allowed_match_types = {
        "architecture_relation",
        "architecture_related",
        "changed_file",
        "definition",
        "transitive_parent",
    }
    for chunk in det_context.get("chunks", []) or []:
        match_type = str(chunk.get("_match_type") or "")
        if match_type in allowed_match_types:
            add_chunk(chunk, match_type)

    def relation_attributes(relation: Dict[str, Any]) -> Dict[str, Any]:
        attributes = relation.get("attributes")
        if isinstance(attributes, dict):
            return attributes
        if not isinstance(attributes, list):
            return {}
        return {
            str(attribute["name"]): attribute.get("value")
            for attribute in attributes
            if (
                isinstance(attribute, dict)
                and isinstance(attribute.get("name"), str)
                and attribute["name"]
            )
        }

    def structural_relations(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        relation = metadata.get("structural_relation")
        if isinstance(relation, dict):
            return [relation]
        legacy = metadata.get("plugin_graph_facts")
        if not isinstance(legacy, list):
            return []
        return [fact for fact in legacy if isinstance(fact, dict)]

    def relation_edge_key(relation: Dict[str, Any]) -> tuple[str, ...]:
        """Identify one typed endpoint pair without collapsing its semantics."""
        return tuple(
            str(relation.get(field) or "")
            for field in ("kind", "source", "target", "path")
        )

    # Framework projections can add one area-scoped view of an edge for every
    # Magento area.  Keep those facts available, but do not let repeated
    # projections displace a distinct source-derived relationship for the same
    # typed endpoints at a finite admission boundary.
    non_area_edges = {
        relation_edge_key(fact)
        for chunk in flattened
        for fact in structural_relations(chunk.get("metadata") or {})
        if "area" not in relation_attributes(fact)
    }

    def redundant_area_projection_priority(
        fact_payloads: List[Dict[str, Any]],
    ) -> int:
        if not fact_payloads:
            return 0
        return int(all(
            "area" in relation_attributes(fact)
            and relation_edge_key(fact) in non_area_edges
            for fact in fact_payloads
        ))

    def fact_semantic_key(
        fact_payloads: List[Dict[str, Any]],
    ) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted(
            (
                str(fact.get("kind") or ""),
                str(fact.get("source") or ""),
                str(fact.get("relation") or ""),
                str(fact.get("target") or ""),
                str(fact.get("path") or ""),
                str(fact.get("line") or ""),
                json.dumps(
                    relation_attributes(fact),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ),
            )
            for fact in fact_payloads
        ))

    def stable_key(chunk: Dict[str, Any]) -> tuple:
        metadata = chunk.get("metadata") or {}
        fact_payloads = structural_relations(metadata)
        fact_priority = 2
        if fact_payloads:
            if any(
                relation_attributes(fact).get("semanticRole") == "diagnostic"
                for fact in fact_payloads
            ):
                fact_priority = 0
            elif any(
                fact.get("related_paths")
                or relation_attributes(fact)
                for fact in fact_payloads
            ):
                fact_priority = 1
        return (
            fact_priority,
            redundant_area_projection_priority(fact_payloads),
            str(metadata.get("architecture_kind", "")),
            str(chunk.get("_matched_on", "")),
            str(metadata.get("path", chunk.get("path", ""))),
            fact_semantic_key(fact_payloads),
            str(metadata.get("architecture_key", "")),
            content_digest(chunk.get("text", "")),
        )

    def matched_review_paths(chunk: Dict[str, Any]) -> tuple[str, ...]:
        raw = str(chunk.get("_matched_on") or "")
        normalized_paths = set()
        for path in raw.split(","):
            normalized = normalize_repository_path(path.strip())
            if normalized:
                normalized_paths.add(normalized)
        return tuple(sorted(normalized_paths))

    def chunk_path(chunk: Dict[str, Any]) -> str:
        metadata = chunk.get("metadata") or {}
        return normalize_repository_path(
            metadata.get("path")
            or chunk.get("path")
            or chunk.get("file_path")
            or ""
        )

    def relation_source_paths(chunk: Dict[str, Any]) -> tuple[str, ...]:
        """Return concrete non-reviewed paths proved by one focused packet."""
        metadata = chunk.get("metadata") or {}
        raw_paths = metadata.get("architecture_paths")
        if not isinstance(raw_paths, (list, tuple, set)) or not raw_paths:
            raw_paths = []
            for fact in structural_relations(metadata):
                raw_paths.append(fact.get("path"))
                related_paths = fact.get("related_paths")
                if isinstance(related_paths, (list, tuple, set)):
                    raw_paths.extend(related_paths)

        # ``_matched_on`` is retrieval provenance, not necessarily the Stage 1
        # ownership set. Architecture packets can be reached through an
        # unchanged related file and consequently name that file in
        # ``_matched_on``. Use the actual batch paths when the caller has them
        # so those related implementations still receive a source body.
        relation_reviewed_paths = (
            explicit_reviewed_paths
            if explicit_reviewed_paths
            else set(matched_review_paths(chunk))
        )
        normalized_paths = {
            normalized
            for path in raw_paths
            for normalized in (normalize_repository_path(path),)
            if normalized and normalized not in relation_reviewed_paths
        }
        return tuple(sorted(normalized_paths))

    def round_robin_architecture(
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        ordered_chunks = sorted(chunks, key=stable_key)
        matched_paths_by_id = {
            id(chunk): matched_review_paths(chunk)
            for chunk in ordered_chunks
        }
        uncovered_paths = {
            path
            for chunk in ordered_chunks
            for path in matched_paths_by_id[id(chunk)]
        }
        coverage_first: List[Dict[str, Any]] = []
        coverage_ids = set()
        while uncovered_paths:
            candidates = [
                chunk for chunk in ordered_chunks
                if id(chunk) not in coverage_ids
                and uncovered_paths.intersection(
                    matched_paths_by_id[id(chunk)]
                )
            ]
            if not candidates:
                break
            selected = min(
                candidates,
                key=lambda chunk: (
                    stable_key(chunk)[0],
                    -len(
                        uncovered_paths.intersection(
                            matched_paths_by_id[id(chunk)]
                        )
                    ),
                    stable_key(chunk),
                ),
            )
            coverage_first.append(selected)
            coverage_ids.add(id(selected))
            uncovered_paths.difference_update(matched_paths_by_id[id(selected)])

        by_kind: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in ordered_chunks:
            if id(chunk) in coverage_ids:
                continue
            metadata = chunk.get("metadata") or {}
            kind = str(metadata.get("architecture_kind") or "architecture")
            by_kind.setdefault(kind, []).append(chunk)
        ordered = list(coverage_first)
        while by_kind:
            # Every changed path with exact architecture evidence gets one
            # candidate before repeats. After that coverage pass, preserve kind
            # fairness and use the best remaining structural fact in each kind.
            # This prevents both a relationship-heavy file and a large set of
            # coarse topology kinds from consuming the bounded prompt input.
            for kind in sorted(
                tuple(by_kind),
                key=lambda value: (
                    stable_key(by_kind[value][0])[0],
                    value,
                ),
            ):
                ordered.append(by_kind[kind].pop(0))
                if not by_kind[kind]:
                    del by_kind[kind]
        return ordered

    architecture_relations = round_robin_architecture([
        chunk for chunk in flattened
        if chunk.get("_match_type") == "architecture_relation"
    ])
    architecture_sources = [
        chunk for chunk in flattened
        if chunk.get("_match_type") == "architecture_related"
    ]
    generic_structural = sorted([
        chunk for chunk in flattened
        if chunk.get("_match_type") in {
            "definition",
            "transitive_parent",
        }
    ], key=stable_key)
    direct = sorted([
        chunk for chunk in flattened
        if chunk.get("_match_type") == "changed_file"
    ], key=stable_key)
    if max_chunks is None:
        return (
            architecture_relations
            + architecture_sources
            + generic_structural
            + direct
        )

    relation_quota = max(1, (max_chunks * 3) // 4)
    support_quota = max(1, max_chunks // 5)
    direct_quota = max(0, max_chunks - relation_quota - support_quota)

    selected_generic = generic_structural[:support_quota]
    selected_direct = direct[:direct_quota]
    relation_capacity = max(
        0,
        max_chunks - len(selected_generic) - len(selected_direct),
    )
    if architecture_relations and relation_capacity == 0:
        # Even a one-chunk request keeps the exact relation as the authority.
        if selected_generic:
            selected_generic.pop()
        elif selected_direct:
            selected_direct.pop()
        relation_capacity = 1

    sources_by_path: Dict[str, List[Dict[str, Any]]] = {}
    for source in architecture_sources:
        path = chunk_path(source)
        if path:
            # The RAG service has already ranked exact method/line bodies within
            # each path. Preserve that deterministic order here.
            sources_by_path.setdefault(path, []).append(source)

    relational: List[Dict[str, Any]] = []
    used_source_ids = set()
    represented_source_paths = set()
    deferred_relations: List[Dict[str, Any]] = []

    # Select relation/source bundles first. A relation that names an available
    # implementation path is never admitted alone merely because its body would
    # fall beyond the independent support quota or prompt tail.
    for relation in architecture_relations:
        available_paths = tuple(
            path for path in relation_source_paths(relation)
            if path in sources_by_path
        )
        new_paths = tuple(
            path for path in available_paths
            if path not in represented_source_paths
        )
        bundle_sources = [
            sources_by_path[path][0]
            for path in new_paths
        ]
        bundle_size = 1 + len(bundle_sources)
        if bundle_sources and len(relational) + bundle_size <= relation_capacity:
            relational.append(relation)
            relational.extend(bundle_sources)
            represented_source_paths.update(new_paths)
            used_source_ids.update(id(source) for source in bundle_sources)
            continue
        deferred_relations.append(relation)

    # Surplus packets are safe only after every implementation path they expose
    # has already received a body, or when no body exists in the exact response.
    # This is what prevents a dense fact inventory from consuming the character
    # budget before the source needed to interpret those facts.
    for relation in deferred_relations:
        if len(relational) >= relation_capacity:
            break
        available_paths = tuple(
            path for path in relation_source_paths(relation)
            if path in sources_by_path
        )
        if available_paths and not all(
            path in represented_source_paths for path in available_paths
        ):
            continue
        relational.append(relation)

    # Once every selected path has one body, retain additional exact fragments
    # when capacity remains (for example, an oversized class implementation).
    for source in architecture_sources:
        if len(relational) >= relation_capacity:
            break
        if id(source) in used_source_ids:
            continue
        if chunk_path(source) not in represented_source_paths:
            continue
        relational.append(source)
        used_source_ids.add(id(source))

    selected = relational + selected_generic + selected_direct
    return selected[:max_chunks]
