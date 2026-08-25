"""
Deterministic context retrieval module for RAG query service.

Retrieves code context using metadata-based (non-semantic) queries against
tree-sitter extracted metadata: identifiers, parent classes, and imports.
"""
import os
import re
from typing import Dict, List, Optional
import logging

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

from .base import RAGQueryBase
from ..core.pr_overlay_manifest import PR_OVERLAY_MANIFEST_PAYLOAD_KEY
from ..core.exact_index import ExactIndexPreconditionError
from rag_pipeline.utils.path_identity import (
    normalize_repository_path,
    repository_path_suffix_candidates,
    repository_paths_match,
)

logger = logging.getLogger(__name__)

COMMON_RELATION_IDENTIFIERS = {
    "and", "array", "bool", "boolean", "call", "class", "clone", "count",
    "dict", "false", "float", "get", "hash", "int", "list", "long", "map",
    "new", "none", "null", "object", "print", "return", "run", "set",
    "str", "string", "this", "true", "void",
}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %s", name, value, default)
        return default


DETERMINISTIC_SCROLL_PAGE_SIZE = max(
    32,
    _env_int("RAG_DETERMINISTIC_SCROLL_PAGE_SIZE", 256),
)
DETERMINISTIC_MAX_MATCHING_POINTS = max(
    DETERMINISTIC_SCROLL_PAGE_SIZE,
    _env_int("RAG_DETERMINISTIC_MAX_MATCHING_POINTS", 5000),
)
HYDRATABLE_SOURCE_RECORD_TYPES = frozenset({
    "source_chunk",
    "architecture_source",
})


def _simple_relation_identifier(value: object) -> Optional[str]:
    """Normalize an indexed relation value into a primary-name lookup token."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip().rstrip(";").strip().strip("'\"")
    if not cleaned:
        return None
    parts = re.split(r'[\\./::\s]+', cleaned)
    for part in reversed(parts):
        part = part.strip().strip("{}()[]<>")
        if (
            len(part) > 1
            and not part.startswith(("{", "*", "$"))
            and part.lower() not in COMMON_RELATION_IDENTIFIERS
        ):
            return part
    return None


def _point_sort_key(point) -> tuple:
    payload = point.payload or {}
    return (
        str(payload.get("path", "")),
        int(payload.get("start_line", 0) or 0),
        int(payload.get("end_line", 0) or 0),
        int(payload.get("architecture_source_part", 0) or 0),
        str(payload.get("primary_name", "")),
        str(getattr(point, "id", "")),
    )


def _is_hydratable_source(point) -> bool:
    payload = point.payload or {}
    return payload.get("structural_record_type") in HYDRATABLE_SOURCE_RECORD_TYPES


def _claim_retrieval_point(seen_points: set, point, text: str) -> bool:
    """Deduplicate one indexed point without collapsing equal text across paths."""
    if not text:
        return False
    # Retain compatibility with callers/tests that seed a legacy text key.
    if text in seen_points:
        return False
    key = ("qdrant-point", str(getattr(point, "id", "")))
    if key in seen_points:
        return False
    seen_points.add(key)
    return True


def _payload_identifiers(payload: Dict) -> set[str]:
    values = []
    for key in (
        "primary_name",
        "full_path",
        "symbol_qualified_name",
        "symbol_names",
    ):
        value = payload.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (list, tuple)):
            values.extend(item for item in value if isinstance(item, str))
    return {value.casefold() for value in values if value}


def _payload_contains_line(payload: Dict, lines: set[int]) -> bool:
    if not lines:
        return False
    try:
        start_line = int(payload.get("start_line"))
        end_line = int(payload.get("end_line"))
    except (TypeError, ValueError):
        return False
    return any(start_line <= line <= end_line for line in lines)


def _payload_contains_identifier(
    payload: Dict,
    identifiers: set[str],
) -> bool:
    if not identifiers:
        return False
    if identifiers.intersection(_payload_identifiers(payload)):
        return True
    text = payload.get("text", payload.get("_node_content", ""))
    if not isinstance(text, str) or not text:
        return False
    folded_text = text.casefold()
    return any(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(identifier)}"
            r"(?![A-Za-z0-9_])",
            folded_text,
        )
        is not None
        for identifier in identifiers
        if identifier
    )


def _source_hydration_rank(
        point,
        preferred_identifiers: set[str],
        preferred_lines: set[int],
) -> tuple:
    payload = point.payload or {}
    return (
        0 if _payload_contains_line(payload, preferred_lines) else 1,
        0 if _payload_contains_identifier(payload, preferred_identifiers) else 1,
        0 if payload.get("structural_record_type") == "source_chunk" else 1,
        _point_sort_key(point),
    )


def _select_hydrated_sources(
        points: List[object],
        preferred_identifiers: set[str],
        preferred_lines: set[int],
) -> List[object]:
    """Select exact source bodies; resolver-only symbol records never qualify."""
    sources = [point for point in points if _is_hydratable_source(point)]
    ordered = sorted(
        sources,
        key=lambda point: _source_hydration_rank(
            point,
            preferred_identifiers,
            preferred_lines,
        ),
    )
    proved = [
        point for point in ordered
        if (
            _payload_contains_line(point.payload or {}, preferred_lines)
            or _payload_contains_identifier(
                point.payload or {}, preferred_identifiers
            )
        )
    ]
    # When the graph proves only a path, exact-path source is still useful.
    # When it nominates an identifier/line, keep only the matching body or
    # fragments so unrelated helpers from the same file do not consume context.
    return proved or ordered


def _failure(stage: str, exception: Exception, path: Optional[str] = None) -> Dict[str, str]:
    result = {
        "stage": stage,
        "reason": f"{stage}_error",
        "error_type": type(exception).__name__,
        "message": str(exception)[:500],
    }
    if path:
        result["path"] = path
    return result


def _graph_fact_paths(fact: object) -> set[str]:
    if not isinstance(fact, dict):
        return set()
    paths = set()
    path = fact.get("path")
    if isinstance(path, str) and path:
        paths.add(path)
    related_paths = fact.get("related_paths")
    if isinstance(related_paths, (list, tuple)):
        paths.update(
            value for value in related_paths
            if isinstance(value, str) and value
        )
    return paths


def _graph_fact_retrieval_identifiers(fact: object) -> set[str]:
    """
    Read neutral plugin-nominated exact lookup identifiers.

    Plugins may attach ``retrievalIdentifier:<stable-name>`` attributes to a
    graph fact. RAG treats only the values as primary-name lookup candidates;
    it does not interpret a plugin-specific attribute name.
    """
    if not isinstance(fact, dict):
        return set()
    attributes = fact.get("attributes")
    if not isinstance(attributes, dict):
        return set()
    return {
        value
        for key, value in attributes.items()
        if (
            isinstance(key, str)
            and key.startswith("retrievalIdentifier:")
            and isinstance(value, str)
            and value.strip()
        )
    }


def _focused_architecture_payload(
        payload: Dict,
        requested_paths: set[str],
) -> Optional[Dict]:
    """Project a compacted graph node onto facts touching the requested paths.

    Architecture storage intentionally packs multiple graph facts into one
    Qdrant point. A metadata match therefore identifies a candidate point, not
    permission to forward every co-located fact into the review prompt.
    """
    facts = payload.get("plugin_graph_facts")
    if not isinstance(facts, list):
        return dict(payload)

    focused_facts = [
        fact for fact in facts
        if _graph_fact_paths(fact).intersection(requested_paths)
    ]
    if not focused_facts:
        return None

    focused_paths = sorted({
        path for fact in focused_facts for path in _graph_fact_paths(fact)
    })
    focused_identifiers = sorted({
        value
        for fact in focused_facts
        for value in (
            fact.get("source"),
            fact.get("target"),
            *_graph_fact_retrieval_identifiers(fact),
        )
        if isinstance(value, str) and value
    })
    packet_keys = sorted({
        str(fact.get("packetKey"))
        for fact in focused_facts
        if fact.get("packetKey")
    })

    focused = dict(payload)
    focused["plugin_graph_facts"] = focused_facts
    focused["architecture_paths"] = focused_paths
    focused["architecture_identifiers"] = focused_identifiers
    if packet_keys:
        focused["architecture_keys"] = packet_keys
    return focused


def _render_focused_architecture_text(payload: Dict, matched_paths: List[str]) -> str:
    """Render only exact selected graph facts, preserving their provenance."""
    facts = payload.get("plugin_graph_facts")
    if not isinstance(facts, list):
        return str(payload.get("text", payload.get("_node_content", "")))

    lines = [
        "Deterministic repository architecture context",
        f"Plugin: {payload.get('architecture_plugin', 'unknown')}",
        f"Kind: {payload.get('architecture_kind', 'unknown')}",
        "Matched paths: " + ", ".join(matched_paths),
        "Facts:",
    ]
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        attributes = fact.get("attributes")
        attribute_text = ""
        if isinstance(attributes, dict) and attributes:
            attribute_text = " {" + ", ".join(
                f"{key}={attributes[key]}" for key in sorted(attributes)
            ) + "}"
        packet_key = fact.get("packetKey")
        packet_text = f"Packet {packet_key}: " if packet_key else ""
        lines.append(
            f"- {packet_text}[{fact.get('kind', 'relation')}] "
            f"{fact.get('source', '')} {fact.get('relation', '')} "
            f"{fact.get('target', '')} "
            f"({fact.get('path', 'unknown')}:{fact.get('line', 1)})"
            f"{attribute_text}"
        )
    return "\n".join(lines)


class DeterministicContextMixin:
    """Deterministic (metadata-based) context retrieval for RAGQueryService.

    Uses tree-sitter metadata already extracted during indexing. Exact changed
    paths are loaded directly; imports, inheritance declarations, and
    plugin-nominated identifiers may resolve dependency definitions. Ambiguous
    unqualified names abstain instead of expanding to every matching file.

    Same input always produces the same exact-filter result.
    """

    def get_deterministic_context(
            self: RAGQueryBase,
            workspace: str,
            project: str,
            branches: List[str],
            file_paths: List[str],
            collection_target: str,
            limit_per_file: Optional[int] = None,
            pr_number: Optional[int] = None,
            pr_changed_files: Optional[List[str]] = None,
            additional_identifiers: Optional[List[str]] = None,
            expected_revisions: Optional[Dict[str, str]] = None,
            pr_source_revision: Optional[str] = None,
            pr_base_revision: Optional[str] = None,
            pr_base_generation_manifest_sha256: Optional[str] = None,
            pr_generation_fingerprint: Optional[str] = None,
    ) -> Dict:
        """
        Get context using DETERMINISTIC metadata-based retrieval.

        Leverages ALL tree-sitter metadata extracted during indexing:
        - symbol_names: function/method/class names
        - primary_name: main identifier
        - parent_class: containing class/inheritance hint
        - full_path: qualified name (e.g., "Data.getConfigData")
        - imports: import statements
        - extends: parent classes/interfaces
        - namespace: package/namespace
        - node_type: method_declaration, class_definition, etc.

        Multi-step process:
        1. Query chunks for changed file_paths
        2. Extract metadata (identifiers, parent classes, imports)
        3. Find related definitions by:
           a) primary_name match (definitions of used identifiers)

        NO LANGUAGE-SPECIFIC PARSING NEEDED - tree-sitter already did that!
        Same input always produces same output (deterministic).

        Args:
            workspace: VCS workspace
            project: Project name
            branches: Branches to search (target + base for PRs)
            file_paths: Changed file paths from diff
            limit_per_file: Optional caller-requested chunks per file. Normal
                review retrieval has no per-file cap; the global exact-match
                safety limit remains authoritative and observable.

        Returns:
            Dict with chunks grouped by retrieval type and rich metadata
        """
        if pr_generation_fingerprint and not all((
            pr_number,
            pr_source_revision,
            pr_base_revision,
            pr_base_generation_manifest_sha256,
        )):
            raise ExactIndexPreconditionError(
                "PR generation fingerprint requires PR number and complete "
                "source/base generation identity"
            )
        if pr_generation_fingerprint and len([
            branch for branch in branches if branch
        ]) != 1:
            raise ExactIndexPreconditionError(
                "revision-bound deterministic context requires exactly one "
                "authoritative branch"
            )

        collection_name = collection_target

        if not self._collection_or_alias_exists(collection_name):
            logger.warning(f"Collection {collection_name} does not exist")
            return {"chunks": [], "changed_files": {}, "related_definitions": {},
                    "_metadata": {
                        "error": "collection_not_found",
                        "retrieval_state": "unavailable",
                        "retrieval_scope": "none",
                        "coverage_state": "unavailable",
                        "context_usable": False,
                        "bounded_files": [],
                        "partial_reasons": ["collection_not_found"],
                        "failures": [{
                            "stage": "collection",
                            "reason": "collection_not_found",
                            "error_type": "CollectionNotFound",
                            "message": f"Collection {collection_name} does not exist",
                        }],
                    }}

        file_paths = sorted({path.lstrip("/") for path in file_paths if path})
        pr_changed_path_set = {
            normalize_repository_path(path)
            for path in (pr_changed_files or [])
            if normalize_repository_path(path)
        }
        branches = list(dict.fromkeys(branch for branch in branches if branch))
        self._observe_branches(collection_name, branches)
        logger.info(f"Deterministic context: files={file_paths[:5]}, branches={branches}")

        # ── Build branch filter ──
        target_branch = branches[0] if branches else None

        repository_branch_filters = []
        for branch in branches:
            conditions = [
                FieldCondition(
                    key="branch",
                    match=MatchValue(value=branch),
                ),
            ]
            if expected_revisions and branch in expected_revisions:
                conditions.append(FieldCondition(
                    key="commit",
                    match=MatchValue(value=expected_revisions[branch]),
                ))
            repository_branch_filters.append(Filter(must=conditions))
        base_branch_condition = (
            repository_branch_filters[0]
            if len(repository_branch_filters) == 1
            else Filter(should=repository_branch_filters)
        )

        if pr_number:
            pr_conditions = [
                FieldCondition(key="pr", match=MatchValue(value=True)),
                FieldCondition(
                    key="pr_number",
                    match=MatchValue(value=pr_number),
                ),
            ]
            if pr_generation_fingerprint:
                pr_conditions.extend([
                    FieldCondition(
                        key="pr_source_revision",
                        match=MatchValue(value=pr_source_revision),
                    ),
                    FieldCondition(
                        key="pr_base_revision",
                        match=MatchValue(value=pr_base_revision),
                    ),
                    FieldCondition(
                        key="pr_base_generation_manifest_sha256",
                        match=MatchValue(
                            value=pr_base_generation_manifest_sha256
                        ),
                    ),
                    FieldCondition(
                        key="pr_generation_fingerprint",
                        match=MatchValue(value=pr_generation_fingerprint),
                    ),
                ])
            base_conditions = [base_branch_condition]
            base_exclusions = (
                [
                    FieldCondition(
                        key="path",
                        match=MatchAny(any=sorted(pr_changed_path_set)),
                    ),
                ]
                if pr_changed_path_set
                else []
            )
            branch_filter = Filter(should=[
                Filter(
                    must=base_conditions,
                    must_not=base_exclusions,
                ),
                Filter(
                    must=pr_conditions,
                    must_not=[
                        FieldCondition(
                            key=PR_OVERLAY_MANIFEST_PAYLOAD_KEY,
                            match=MatchValue(value=True),
                        ),
                    ],
                ),
            ])
            logger.info(f"Deterministic hybrid mode: also searching PR-indexed data (pr_number={pr_number})")
        else:
            branch_filter = base_branch_condition

        # ── Tracking state ──
        all_chunks = []
        changed_files_chunks = {}
        architecture_context = {}
        architecture_related = {}
        related_definitions = {}
        failures = []
        file_status = {}
        changed_file_retrieval = {}

        identifiers_to_find = set()
        imports_raw = set()
        extends_raw = set()

        # The request is the authority for invalidating materialized branch
        # context.  Do not rely on finding a PR-indexed chunk: deleted files and
        # architecture-only files legitimately have no replacement code chunk.
        changed_file_paths = set(pr_changed_path_set)
        seen_texts = set()
        target_branch_paths = set()

        # ========== STEP 1: Get chunks from changed files ==========
        for file_path in file_paths:
            try:
                retrieval_diagnostic = {}
                chunks_for_file = self._query_changed_file(
                    collection_name, branch_filter, file_path, limit_per_file,
                    branches, target_branch, seen_texts, target_branch_paths,
                    changed_file_paths, imports_raw, extends_raw, all_chunks,
                    retrieval_diagnostic,
                )
                changed_files_chunks[file_path] = chunks_for_file
                changed_file_retrieval[file_path] = retrieval_diagnostic
                if retrieval_diagnostic.get("retrieval_state") == "partial":
                    file_status[file_path] = "partial"
                    partial_reasons = (
                        retrieval_diagnostic.get("partial_reasons")
                        or ["unknown_scan_truncation"]
                    )
                    for reason in partial_reasons:
                        failures.append({
                            "stage": "changed_file",
                            "path": file_path,
                            "reason": reason,
                            "error_type": "ResultLimit",
                            "message": (
                                "Exact changed-file retrieval did not exhaust "
                                f"the matching index points ({reason}); this "
                                "file's context is explicitly partial"
                            ),
                        })
                else:
                    file_status[file_path] = "hit" if chunks_for_file else "miss"
            except Exception as e:
                logger.warning(f"Error querying file '{file_path}': {e}")
                file_status[file_path] = "error"
                failures.append(_failure("changed_file", e, file_path))

        logger.info(
            "Step 1: %s chunks from changed files; extracted %s imports and "
            "%s inheritance/type references",
            len(all_chunks),
            len(imports_raw),
            len(extends_raw),
        )

        # ========== STEP 1b: Expand exact repository architecture edges ==========
        # Framework plugins index bounded architecture packets keyed by every path
        # participating in a relation. This query is metadata-only and introduces
        # no model call or similarity-dependent behavior.
        architecture_retrieval = {}
        if file_paths:
            try:
                architecture_retrieval = self._query_architecture_context(
                    collection_name,
                    branch_filter,
                    file_paths,
                    limit_per_file,
                    branches,
                    target_branch,
                    target_branch_paths,
                    changed_file_paths,
                    seen_texts,
                    all_chunks,
                    architecture_context,
                    architecture_related,
                )
                if architecture_retrieval.get("truncated"):
                    partial_reasons = (
                        architecture_retrieval.get("partial_reasons")
                        or ["unknown_scan_truncation"]
                    )
                    for reason in partial_reasons:
                        failures.append({
                            "stage": "architecture_context",
                            "reason": reason,
                            "error_type": "ResultLimit",
                            "message": (
                                "Exact architecture retrieval did not exhaust "
                                f"the matching index points ({reason}); context "
                                "is explicitly partial"
                            ),
                        })
            except Exception as exception:
                failures.append(_failure("architecture_context", exception))

        # ── Inject enrichment-supplied dependency identifiers ──
        # These are restricted by the orchestrator to imported and inherited
        # types. Calls and declarations from the changed file are intentionally
        # excluded because common method names are not repository-wide edges.
        if additional_identifiers:
            pre_count = len(identifiers_to_find | imports_raw | extends_raw)
            for name in sorted(additional_identifiers):
                name = name.strip()
                if name and len(name) > 1:
                    identifiers_to_find.add(name)
            post_count = len(identifiers_to_find | imports_raw | extends_raw)
            logger.info(f"Enrichment injection: {post_count - pre_count} new identifiers "
                       f"from {len(additional_identifiers)} additional_identifiers")

        # ========== STEP 2: Find definitions by primary_name ==========
        all_to_find = identifiers_to_find | imports_raw | extends_raw
        if all_to_find:
            try:
                self._query_definitions(
                    collection_name, branch_filter, all_to_find,
                    branches, target_branch, target_branch_paths,
                    changed_file_paths, seen_texts, all_chunks, related_definitions
                )
            except Exception as exception:
                failures.append(_failure("definitions", exception))

        # ========== STEP 2b: Transitive parent type resolution ==========
        # Extract extends/implements/parent_class from the definitions found
        # in Step 2, then do one more hop to find THEIR parent types.
        # This ensures the full inheritance chain is visible (depth=2).
        transitive_parents = set()
        for def_name, def_chunks in sorted(related_definitions.items()):
            for chunk in def_chunks:
                meta = chunk.get("metadata", {})
                if isinstance(meta.get("extends"), list):
                    transitive_parents.update(meta["extends"])
                if meta.get("parent_class"):
                    transitive_parents.add(meta["parent_class"])

        # Remove names already looked up to avoid redundant queries
        transitive_parents -= all_to_find
        transitive_parents -= changed_file_paths  # Skip changed file paths
        transitive_parents = {p for p in transitive_parents if p and len(p) > 1}

        if transitive_parents:
            try:
                self._query_transitive_parents(
                    collection_name, branch_filter, transitive_parents,
                    branches, target_branch, target_branch_paths,
                    changed_file_paths, seen_texts, all_chunks, related_definitions
                )
            except Exception as exception:
                failures.append(_failure("transitive_parents", exception))

        logger.info(f"Deterministic context complete: {len(all_chunks)} total chunks "
                   f"(changed: {sum(len(v) for v in changed_files_chunks.values())}, "
                   f"definitions: {sum(len(v) for v in related_definitions.values())})")

        for chunk in all_chunks:
            metadata = chunk.get("metadata") or {}
            if metadata.get("pr") is True:
                if pr_generation_fingerprint and (
                    metadata.get("pr_generation_fingerprint")
                    != pr_generation_fingerprint
                    or metadata.get("pr_source_revision")
                    != pr_source_revision
                    or metadata.get("pr_base_revision") != pr_base_revision
                    or metadata.get(
                        "pr_base_generation_manifest_sha256"
                    ) != pr_base_generation_manifest_sha256
                ):
                    raise ExactIndexPreconditionError(
                        "deterministic retrieval returned a PR point outside "
                        "the requested overlay generation"
                    )
                continue
            expected_revision = (
                expected_revisions.get(metadata.get("branch"))
                if expected_revisions
                else None
            )
            if (
                expected_revision is not None
                and metadata.get("commit") != expected_revision
            ):
                raise ExactIndexPreconditionError(
                    "deterministic retrieval returned a repository point "
                    "outside the requested immutable revision"
                )

        bounded_files = sorted(
            path
            for path, diagnostic in changed_file_retrieval.items()
            if diagnostic.get("retrieval_scope") == "bounded"
        )
        retrieval_state = (
            "complete" if not failures else "partial" if all_chunks else "failed"
        )
        failure_reasons = sorted({
            str(failure.get("reason") or "unknown_failure")
            for failure in failures
        })
        retrieval_scope = "bounded" if bounded_files else "exhaustive"

        return {
            "chunks": all_chunks,
            "changed_files": changed_files_chunks,
            "architecture_context": architecture_context,
            "architecture_related": architecture_related,
            "related_definitions": related_definitions,
            "_metadata": {
                "branches_searched": branches,
                "target_branch": target_branch,
                "files_requested": file_paths,
                "identifiers_extracted": sorted(identifiers_to_find),
                "imports_extracted": sorted(imports_raw),
                "extends_extracted": sorted(extends_raw),
                "definition_identifiers_queried": sorted(all_to_find),
                "transitive_parents_extracted": sorted(transitive_parents),
                "changed_file_retrieval": changed_file_retrieval,
                "architecture_packets_found": sum(len(value) for value in architecture_context.values()),
                "architecture_related_found": sum(len(value) for value in architecture_related.values()),
                "architecture_retrieval": architecture_retrieval,
                "target_branch_paths_found": len(target_branch_paths),
                "file_status": file_status,
                # ``retrieval_state`` reports whether every query operation
                # completed.  A caller-requested per-file result bound is a
                # successfully fulfilled query, not a retrieval failure.  Its
                # narrower coverage is exposed separately through scope and the
                # per-file diagnostics below.
                "retrieval_state": retrieval_state,
                "retrieval_scope": retrieval_scope,
                "coverage_state": (
                    "bounded_complete"
                    if retrieval_state == "complete" and bounded_files
                    else retrieval_state
                ),
                "context_usable": bool(all_chunks),
                "bounded_files": bounded_files,
                "partial_reasons": failure_reasons,
                "failures": failures,
            }
        }

    # ── Internal helpers ──

    def _query_architecture_context(
            self, collection_name, branch_filter, file_paths, limit_per_file,
            branches, target_branch, target_branch_paths, changed_file_paths,
            seen_texts, all_chunks, architecture_context, architecture_related
    ) -> Dict[str, object]:
        """Retrieve exact framework relations and their concrete source files."""
        packet_points = {}
        batch_size = 64
        packet_scan_truncated = False
        packet_partial_reasons = set()
        for offset in range(0, len(file_paths), batch_size):
            path_batch = file_paths[offset:offset + batch_size]
            remaining = DETERMINISTIC_MAX_MATCHING_POINTS - len(packet_points)
            if remaining <= 0:
                packet_scan_truncated = True
                packet_partial_reasons.add("global_matching_point_limit")
                break
            scroll_diagnostic = {}
            results, truncated = self._scroll_bounded(
                collection_name,
                Filter(must=[
                    branch_filter,
                    FieldCondition(
                        key="architecture_paths",
                        match=MatchAny(any=path_batch),
                    ),
                ]),
                remaining,
                diagnostic=scroll_diagnostic,
            )
            packet_scan_truncated = packet_scan_truncated or truncated
            packet_partial_reasons.update(
                scroll_diagnostic.get("partial_reasons") or []
            )
            for point in results:
                packet_points[str(point.id)] = point

        selected_packets = self._apply_branch_priority(
            list(packet_points.values()),
            target_branch,
            branches,
            target_branch_paths,
        )
        related_paths = set()
        preferred_identifiers_by_path = {}
        preferred_lines_by_path = {}
        requested_paths = set(file_paths)
        for point in selected_packets:
            payload = _focused_architecture_payload(
                point.payload or {},
                requested_paths,
            )
            if payload is None:
                continue
            packet_paths = {
                path for path in payload.get("architecture_paths", [])
                if isinstance(path, str) and path
            }
            # A branch architecture packet is a materialized view of every
            # source path listed in its payload.  If the PR replaces any one
            # of those paths, the packet is no longer evidence for the reviewed
            # revision.  PR-scoped architecture packets, when present, are the
            # only safe replacement.
            if (
                not payload.get("pr")
                and packet_paths.intersection(changed_file_paths)
            ):
                continue
            matched_paths = sorted(packet_paths & requested_paths)
            text = _render_focused_architecture_text(payload, matched_paths)
            if not _claim_retrieval_point(seen_texts, point, text):
                continue
            related_paths.update(packet_paths - requested_paths)
            for fact in payload.get("plugin_graph_facts", []) or []:
                if not isinstance(fact, dict):
                    continue
                fact_related_paths = (
                    _graph_fact_paths(fact) - requested_paths
                )
                identifiers = {
                    identifier.casefold()
                    for identifier in (
                        fact.get("source"),
                        fact.get("target"),
                        *_graph_fact_retrieval_identifiers(fact),
                    )
                    if isinstance(identifier, str) and identifier
                }
                if identifiers:
                    for path in fact_related_paths:
                        preferred_identifiers_by_path.setdefault(
                            path,
                            set(),
                        ).update(identifiers)

                # GraphFact.line belongs only to GraphFact.path. Related-path
                # line attributes are plugin-specific and are intentionally
                # not interpreted by this neutral host.
                fact_path = fact.get("path")
                try:
                    fact_line = int(fact.get("line"))
                except (TypeError, ValueError):
                    fact_line = 0
                if (
                    isinstance(fact_path, str)
                    and fact_path in fact_related_paths
                    and fact_line > 0
                ):
                    preferred_lines_by_path.setdefault(
                        fact_path,
                        set(),
                    ).add(fact_line)
            key = str(payload.get("architecture_key", "architecture"))
            chunk = {
                "text": text,
                "metadata": {
                    key: value for key, value in payload.items()
                    if key not in ("text", "_node_content")
                },
                "_match_type": "architecture_relation",
                "_match_priority": 0,
                "_matched_on": ",".join(matched_paths),
            }
            all_chunks.append(chunk)
            architecture_context.setdefault(key, []).append(chunk)

        related_paths = sorted(
            path for path in related_paths
            if not path.startswith("__analysis_architecture__/")
        )
        if not related_paths:
            return {
                "packet_candidates": len(packet_points),
                "packet_chunks": sum(len(value) for value in architecture_context.values()),
                "related_candidates": 0,
                "related_chunks": 0,
                "truncated": packet_scan_truncated,
                "partial_reasons": sorted(packet_partial_reasons),
            }

        related_points = {}
        related_scan_truncated = False
        related_partial_reasons = set()
        for offset in range(0, len(related_paths), batch_size):
            path_batch = related_paths[offset:offset + batch_size]
            remaining = DETERMINISTIC_MAX_MATCHING_POINTS - len(related_points)
            if remaining <= 0:
                related_scan_truncated = True
                related_partial_reasons.add("global_matching_point_limit")
                break
            scroll_diagnostic = {}
            results, truncated = self._scroll_bounded(
                collection_name,
                Filter(must=[
                    branch_filter,
                    FieldCondition(key="path", match=MatchAny(any=path_batch)),
                    FieldCondition(
                        key="structural_record_type",
                        match=MatchAny(any=sorted(
                            HYDRATABLE_SOURCE_RECORD_TYPES
                        )),
                    ),
                ]),
                remaining,
                diagnostic=scroll_diagnostic,
            )
            related_scan_truncated = related_scan_truncated or truncated
            related_partial_reasons.update(
                scroll_diagnostic.get("partial_reasons") or []
            )
            for point in results:
                if _is_hydratable_source(point):
                    related_points[str(point.id)] = point

        selected_related = self._apply_branch_priority(
            list(related_points.values()),
            target_branch,
            branches,
            target_branch_paths,
        )
        related_by_path = {}
        for point in selected_related:
            path = str((point.payload or {}).get("path", ""))
            if path:
                related_by_path.setdefault(path, []).append(point)

        for path in sorted(related_by_path):
            preferred_identifiers = preferred_identifiers_by_path.get(
                path,
                set(),
            )
            selected_sources = _select_hydrated_sources(
                related_by_path[path],
                preferred_identifiers,
                preferred_lines_by_path.get(path, set()),
            )
            for point in selected_sources:
                payload = point.payload or {}
                text = payload.get("text", payload.get("_node_content", ""))
                if not _claim_retrieval_point(seen_texts, point, text):
                    continue
                chunk = {
                    "text": text,
                    "metadata": {
                        key: value for key, value in payload.items()
                        if key not in ("text", "_node_content")
                    },
                    "_match_type": "architecture_related",
                    "_match_priority": 1,
                    "_matched_on": path,
                }
                all_chunks.append(chunk)
                architecture_related.setdefault(path, []).append(chunk)

        logger.info(
            "Architecture expansion: %s relation chunks, %s related code chunks from %s paths",
            sum(len(value) for value in architecture_context.values()),
            sum(len(value) for value in architecture_related.values()),
            len(related_paths),
        )
        return {
            "packet_candidates": len(packet_points),
            "packet_chunks": sum(len(value) for value in architecture_context.values()),
            "related_candidates": len(related_points),
            "related_chunks": sum(len(value) for value in architecture_related.values()),
            "truncated": packet_scan_truncated or related_scan_truncated,
            "partial_reasons": sorted(
                packet_partial_reasons | related_partial_reasons
            ),
        }

    def _scroll_bounded(
            self,
            collection_name: str,
            scroll_filter,
            max_points: int,
            diagnostic: Optional[Dict[str, object]] = None,
    ) -> tuple[list, bool]:
        """Paginate an exact lookup and explain why it could not be exhausted."""
        points = []
        offset = None
        seen_offsets = set()

        def finish(partial_reason: Optional[str]) -> tuple[list, bool]:
            if diagnostic is not None:
                diagnostic.update({
                    "matching_points_loaded": len(points),
                    "global_matching_point_limit": max_points,
                    "scan_complete": partial_reason is None,
                    "partial_reasons": (
                        [partial_reason] if partial_reason else []
                    ),
                })
            return points, partial_reason is not None

        while len(points) < max_points:
            page_limit = min(
                DETERMINISTIC_SCROLL_PAGE_SIZE,
                max_points - len(points),
            )
            kwargs = {
                "collection_name": collection_name,
                "scroll_filter": scroll_filter,
                "limit": page_limit,
                "with_payload": True,
                "with_vectors": False,
            }
            if offset is not None:
                kwargs["offset"] = offset
            page, next_offset = self.qdrant_client.scroll(**kwargs)
            points.extend(page)
            if next_offset is None:
                return finish(None)
            offset_key = repr(next_offset)
            if offset_key in seen_offsets:
                logger.warning(
                    "Qdrant exact scroll repeated offset %s; returning partial context",
                    offset_key,
                )
                return finish("pagination_stalled")
            seen_offsets.add(offset_key)
            offset = next_offset

        return finish(
            "global_matching_point_limit" if offset is not None else None
        )

    def _load_exact_source_points(
            self,
            collection_name: str,
            branch_filter,
            paths: set[str],
            branches: List[str],
            target_branch: str,
            target_branch_paths: set,
    ) -> tuple[list, bool]:
        """Hydrate exact repository paths without returning resolver records."""
        if not paths:
            return [], False

        source_points = {}
        truncated = False
        ordered_paths = sorted(paths)
        for offset in range(0, len(ordered_paths), 64):
            remaining = DETERMINISTIC_MAX_MATCHING_POINTS - len(source_points)
            if remaining <= 0:
                truncated = True
                break
            results, page_truncated = self._scroll_bounded(
                collection_name,
                Filter(must=[
                    branch_filter,
                    FieldCondition(
                        key="path",
                        match=MatchAny(any=ordered_paths[offset:offset + 64]),
                    ),
                    FieldCondition(
                        key="structural_record_type",
                        match=MatchAny(any=sorted(
                            HYDRATABLE_SOURCE_RECORD_TYPES
                        )),
                    ),
                ]),
                remaining,
            )
            truncated = truncated or page_truncated
            for point in results:
                if _is_hydratable_source(point):
                    source_points[str(point.id)] = point

        selected = self._apply_branch_priority(
            list(source_points.values()),
            target_branch,
            branches,
            target_branch_paths,
        )
        return selected, truncated

    def _resolve_unique_definition_sources(
            self,
            collection_name,
            branch_filter,
            candidates_by_name,
            branches,
            target_branch,
            target_branch_paths,
            changed_file_paths,
    ) -> tuple[Dict[str, List[object]], List[str], bool]:
        """Resolve one definition path, then return only its source bodies."""
        resolved = {}
        ambiguous_names = []
        missing_paths = set()

        for primary_name in sorted(candidates_by_name):
            paths = candidates_by_name[primary_name]
            if len(paths) != 1:
                ambiguous_names.append(primary_name)
                continue
            unique_path = next(iter(paths))
            if any(
                repository_paths_match(unique_path, changed_path)
                for changed_path in changed_file_paths
            ):
                continue

            resolver_points = sorted(
                paths[unique_path],
                key=_point_sort_key,
            )
            source_points = [
                point for point in resolver_points
                if _is_hydratable_source(point)
            ]
            preferred_lines = set()
            for point in resolver_points:
                payload = point.payload or {}
                if payload.get("structural_record_type") != "symbol_definition":
                    continue
                try:
                    line = int(payload.get("start_line"))
                except (TypeError, ValueError):
                    continue
                if line > 0:
                    preferred_lines.add(line)

            resolved[primary_name] = {
                "path": unique_path,
                "points": source_points,
                "lines": preferred_lines,
            }
            if not source_points:
                missing_paths.add(unique_path)

        hydrated_points, truncated = self._load_exact_source_points(
            collection_name,
            branch_filter,
            missing_paths,
            branches,
            target_branch,
            target_branch_paths,
        )
        hydrated_by_path = {}
        for point in hydrated_points:
            path = normalize_repository_path(
                (point.payload or {}).get("path", "")
            )
            if path:
                hydrated_by_path.setdefault(path, []).append(point)

        selected_by_name = {}
        for primary_name, resolution in resolved.items():
            source_points = resolution["points"] or hydrated_by_path.get(
                resolution["path"],
                [],
            )
            selected = _select_hydrated_sources(
                source_points,
                {primary_name.casefold()},
                resolution["lines"],
            )
            if selected:
                selected_by_name[primary_name] = selected

        return selected_by_name, ambiguous_names, truncated

    def _apply_branch_priority(
            self,
            points: list,
            target: str,
            branches: List[str],
            target_branch_paths: set
    ) -> list:
        """Filter points to prioritize: PR-indexed > target branch > base branch."""
        points = list(points)
        if not points:
            return points

        by_path = {}
        for p in sorted(points, key=_point_sort_key):
            path = p.payload.get("path", "")
            if path not in by_path:
                by_path[path] = []
            by_path[path].append(p)

        result = []
        for path, path_points in sorted(by_path.items()):
            pr_points = [p for p in path_points if p.payload.get("pr") is True]
            if pr_points:
                result.extend(pr_points)
                continue

            branch_points = [p for p in path_points if p.payload.get("pr") is not True]
            if not target or len(branches) == 1:
                result.extend(branch_points)
                continue

            has_target = any(p.payload.get("branch") == target for p in branch_points)
            if has_target:
                result.extend([p for p in branch_points if p.payload.get("branch") == target])
            elif path not in target_branch_paths:
                result.extend(branch_points)

        return sorted(result, key=_point_sort_key)

    def _query_changed_file(
            self, collection_name, branch_filter, file_path, limit_per_file,
            branches, target_branch, seen_texts, target_branch_paths,
            changed_file_paths, imports_raw, extends_raw, all_chunks,
            retrieval_diagnostic=None,
    ) -> List[Dict]:
        """Query chunks for a single changed file and extract metadata."""
        normalized_path = normalize_repository_path(file_path)

        # Try exact path match
        scan_attempts = []
        scan_partial_reasons = set()
        exact_scan_diagnostic = {}
        results, scan_truncated = self._scroll_bounded(
            collection_name,
            Filter(must=[
                branch_filter,
                FieldCondition(key="path", match=MatchValue(value=normalized_path))
            ]),
            DETERMINISTIC_MAX_MATCHING_POINTS,
            diagnostic=exact_scan_diagnostic,
        )
        scan_attempts.append({
            "lookup": "exact_path",
            **exact_scan_diagnostic,
        })
        scan_partial_reasons.update(
            exact_scan_diagnostic.get("partial_reasons") or []
        )

        # If the caller included an archive/checkout root, try only exact
        # multi-segment suffixes. A basename query is unsafe in framework
        # repositories where hundreds of modules may contain ``etc/di.xml``.
        if not results:
            suffix_candidates = repository_path_suffix_candidates(
                normalized_path
            )[1:]
            if suffix_candidates:
                suffix_scan_diagnostic = {}
                results, suffix_scan_truncated = self._scroll_bounded(
                    collection_name,
                    Filter(must=[
                        branch_filter,
                        FieldCondition(
                            key="path",
                            match=MatchAny(any=suffix_candidates),
                        ),
                    ]),
                    DETERMINISTIC_MAX_MATCHING_POINTS,
                    diagnostic=suffix_scan_diagnostic,
                )
                scan_attempts.append({
                    "lookup": "path_suffix",
                    "candidate_count": len(suffix_candidates),
                    **suffix_scan_diagnostic,
                })
                scan_truncated = scan_truncated or suffix_scan_truncated
                scan_partial_reasons.update(
                    suffix_scan_diagnostic.get("partial_reasons") or []
                )

        results = [
            point
            for point in results
            if repository_paths_match(
                point.payload.get("path", ""),
                normalized_path,
            )
        ]

        results = sorted(results, key=_point_sort_key)

        # A revision-bound PR request's changed-path manifest is authoritative.
        # Modified paths may have an exact overlay member; deleted and
        # architecture-only paths legitimately may not. In either case, never
        # fall back to the pre-PR target-branch source for that path.
        if any(
            repository_paths_match(normalized_path, changed_path)
            for changed_path in changed_file_paths
        ):
            results = [
                point
                for point in results
                if (point.payload or {}).get("pr") is True
            ]

        # Apply branch priority
        if target_branch and len(branches) > 1:
            has_target = any(p.payload.get("branch") == target_branch for p in results)
            if has_target:
                results = [p for p in results if p.payload.get("branch") == target_branch]
                logger.debug(f"Branch priority: keeping target branch '{target_branch}' for {normalized_path}")

        matching_points = len(results)
        explicit_limit_truncated = (
            isinstance(limit_per_file, int)
            and limit_per_file > 0
            and matching_points > limit_per_file
        )
        if isinstance(limit_per_file, int) and limit_per_file > 0:
            results = results[:limit_per_file]
        if retrieval_diagnostic is not None:
            retrieval_state = "partial" if scan_truncated else "complete"
            retrieval_scope = (
                "bounded" if explicit_limit_truncated else "exhaustive"
            )
            retrieval_diagnostic.update({
                "matching_points": matching_points,
                "chunks_returned": len(results),
                "global_matching_point_limit": (
                    DETERMINISTIC_MAX_MATCHING_POINTS
                ),
                "explicit_limit_per_file": limit_per_file,
                "explicit_limit_applied": explicit_limit_truncated,
                "scan_truncated": scan_truncated,
                "scan_attempts": scan_attempts,
                "retrieval_state": retrieval_state,
                "retrieval_scope": retrieval_scope,
                "coverage_state": (
                    "bounded_complete"
                    if retrieval_state == "complete"
                    and retrieval_scope == "bounded"
                    else retrieval_state
                ),
                "partial_reasons": sorted(scan_partial_reasons),
                "bound_reasons": (
                    ["explicit_limit_per_file"]
                    if explicit_limit_truncated else []
                ),
                "truncated": bool(
                    scan_truncated or explicit_limit_truncated
                ),
            })

        chunks_for_file = []
        for point in results:
            payload = point.payload
            text = payload.get("text", payload.get("_node_content", ""))

            if not _claim_retrieval_point(seen_texts, point, text):
                continue

            if payload.get("branch") == target_branch:
                target_branch_paths.add(payload.get("path", ""))

            chunk = {
                "text": text,
                "metadata": {k: v for k, v in payload.items() if k not in ("text", "_node_content")},
                "_match_type": "changed_file",
                "_match_priority": 1,
                "_matched_on": file_path
            }
            chunks_for_file.append(chunk)
            all_chunks.append(chunk)
            changed_file_paths.add(payload.get("path", ""))

            # Extract tree-sitter dependency metadata for definition lookup.
            # NOTE: We deliberately do NOT add symbol_names or primary_name
            # to identifiers_to_find. Those are the file's OWN definitions
            # (e.g., __construct, getAliases, apply, _toHtml) and looking
            # them up via primary_name MatchAny finds hundreds of unrelated
            # files with the same boilerplate method names. Actual external
            # dependencies come from imports, extends, and enrichment.
            if isinstance(payload.get("imports"), list):
                for imp in payload["imports"]:
                    name = _simple_relation_identifier(imp)
                    if name:
                        imports_raw.add(name)

            if isinstance(payload.get("extends"), list):
                for value in payload["extends"]:
                    name = _simple_relation_identifier(value)
                    if name:
                        extends_raw.add(name)
            if isinstance(payload.get("implements"), list):
                for value in payload["implements"]:
                    name = _simple_relation_identifier(value)
                    if name:
                        extends_raw.add(name)
            if isinstance(payload.get("referenced_types"), list):
                for type_name in payload["referenced_types"]:
                    name = _simple_relation_identifier(type_name)
                    if name:
                        extends_raw.add(name)
            if payload.get("parent_class"):
                extends_raw.add(payload["parent_class"])

        if retrieval_diagnostic is not None:
            # Deduplication can make the number of prompt-usable chunks smaller
            # than the selected index-point count, so publish the final value.
            retrieval_diagnostic["chunks_returned"] = len(chunks_for_file)
            retrieval_diagnostic["context_usable"] = bool(chunks_for_file)

        return chunks_for_file

    def _query_definitions(
            self, collection_name, branch_filter, all_to_find,
            branches, target_branch, target_branch_paths,
            changed_file_paths, seen_texts, all_chunks, related_definitions
    ):
        """STEP 2: Find definitions by primary_name."""
        try:
            identifiers = sorted(set(all_to_find))
            batch_size = max(
                1,
                int(self.config.max_identifiers_per_query),
            )
            points_by_id = {}
            matching_points_scanned = 0
            identifiers_queried = 0
            query_truncated = False
            for offset in range(0, len(identifiers), batch_size):
                remaining = (
                    DETERMINISTIC_MAX_MATCHING_POINTS
                    - matching_points_scanned
                )
                if remaining <= 0:
                    query_truncated = True
                    break
                batch = identifiers[offset:offset + batch_size]
                batch_results, batch_truncated = self._scroll_bounded(
                    collection_name,
                    Filter(must=[
                        branch_filter,
                        FieldCondition(
                            key="primary_name",
                            match=MatchAny(any=batch),
                        ),
                    ]),
                    remaining,
                )
                identifiers_queried += len(batch)
                matching_points_scanned += len(batch_results)
                for point in batch_results:
                    points_by_id[str(point.id)] = point
                if batch_truncated:
                    query_truncated = True
                    break
            if query_truncated or identifiers_queried != len(identifiers):
                raise RuntimeError(
                    "exact definition lookup exceeded its matching-point limit; "
                    "abstaining because identifier uniqueness cannot be proven"
                )

            results = list(points_by_id.values())
            results = self._apply_branch_priority(results, target_branch, branches, target_branch_paths)

            candidates_by_name: Dict[str, Dict[str, List[object]]] = {}
            for point in results:
                payload = point.payload or {}
                primary_name = str(payload.get("primary_name") or "")
                path = normalize_repository_path(payload.get("path", ""))
                if not primary_name or not path:
                    continue
                candidates_by_name.setdefault(primary_name, {}).setdefault(
                    path, []
                ).append(point)

            (
                source_points_by_name,
                ambiguous_names,
                hydration_truncated,
            ) = self._resolve_unique_definition_sources(
                collection_name,
                branch_filter,
                candidates_by_name,
                branches,
                target_branch,
                target_branch_paths,
                changed_file_paths,
            )
            if hydration_truncated:
                raise RuntimeError(
                    "exact definition source hydration exceeded its matching-"
                    "point limit; returning no partial implementation body"
                )

            if ambiguous_names:
                logger.info(
                    "Definition lookup abstained for %d ambiguous identifier(s): %s",
                    len(ambiguous_names),
                    ", ".join(ambiguous_names[:20]),
                )

            for primary_name in sorted(source_points_by_name):
                for point in source_points_by_name[primary_name]:
                    payload = point.payload or {}
                    text = payload.get("text", payload.get("_node_content", ""))
                    if not _claim_retrieval_point(
                        seen_texts,
                        point,
                        text,
                    ):
                        continue

                    chunk = {
                        "text": text,
                        "metadata": {
                            key: value for key, value in payload.items()
                            if key not in ("text", "_node_content")
                        },
                        "_match_type": "definition",
                        "_match_priority": 2,
                        "_matched_on": primary_name,
                    }
                    all_chunks.append(chunk)
                    related_definitions.setdefault(primary_name, []).append(
                        chunk
                    )

            logger.info(f"Step 2: Found {len(related_definitions)} definitions by primary_name")

        except Exception as e:
            logger.warning(f"Error in primary_name query: {e}")
            raise

    def _query_transitive_parents(
            self, collection_name, branch_filter, transitive_parents,
            branches, target_branch, target_branch_paths,
            changed_file_paths, seen_texts, all_chunks, related_definitions
    ):
        """STEP 2b: Second-hop lookup for parent types of definitions found in Step 2.

        Uses bounded MatchAny query batches while resolving every parent. The
        global matching-point ceiling remains authoritative and observable.
        """
        try:
            parents = sorted(set(transitive_parents))
            batch_size = max(
                1,
                int(self.config.max_identifiers_per_query),
            )
            points_by_id = {}
            matching_points_scanned = 0
            parents_queried = 0
            query_truncated = False
            for offset in range(0, len(parents), batch_size):
                remaining = (
                    DETERMINISTIC_MAX_MATCHING_POINTS
                    - matching_points_scanned
                )
                if remaining <= 0:
                    query_truncated = True
                    break
                batch = parents[offset:offset + batch_size]
                batch_results, batch_truncated = self._scroll_bounded(
                    collection_name,
                    Filter(must=[
                        branch_filter,
                        FieldCondition(
                            key="primary_name",
                            match=MatchAny(any=batch),
                        ),
                    ]),
                    remaining,
                )
                parents_queried += len(batch)
                matching_points_scanned += len(batch_results)
                for point in batch_results:
                    points_by_id[str(point.id)] = point
                if batch_truncated:
                    query_truncated = True
                    break
            if query_truncated or parents_queried != len(parents):
                raise RuntimeError(
                    "exact parent lookup exceeded its matching-point limit; "
                    "abstaining because identifier uniqueness cannot be proven"
                )

            results = list(points_by_id.values())
            results = self._apply_branch_priority(results, target_branch, branches, target_branch_paths)

            candidates_by_name: Dict[str, Dict[str, List[object]]] = {}
            for point in results:
                payload = point.payload or {}
                primary_name = str(payload.get("primary_name") or "")
                path = normalize_repository_path(payload.get("path", ""))
                if not primary_name or not path:
                    continue
                candidates_by_name.setdefault(primary_name, {}).setdefault(
                    path, []
                ).append(point)

            (
                source_points_by_name,
                _ambiguous_names,
                hydration_truncated,
            ) = self._resolve_unique_definition_sources(
                collection_name,
                branch_filter,
                candidates_by_name,
                branches,
                target_branch,
                target_branch_paths,
                changed_file_paths,
            )
            if hydration_truncated:
                raise RuntimeError(
                    "exact parent source hydration exceeded its matching-point "
                    "limit; returning no partial implementation body"
                )

            added = 0
            for primary_name in sorted(source_points_by_name):
                for point in source_points_by_name[primary_name]:
                    payload = point.payload or {}
                    text = payload.get("text", payload.get("_node_content", ""))
                    if not _claim_retrieval_point(
                        seen_texts,
                        point,
                        text,
                    ):
                        continue

                    chunk = {
                        "text": text,
                        "metadata": {
                            key: value for key, value in payload.items()
                            if key not in ("text", "_node_content")
                        },
                        "_match_type": "transitive_parent",
                        "_match_priority": 2,
                        "_matched_on": primary_name,
                    }
                    all_chunks.append(chunk)
                    related_definitions.setdefault(primary_name, []).append(
                        chunk
                    )
                    added += 1

            logger.info(f"Step 2b: Found {added} transitive parent definitions "
                       f"from {len(transitive_parents)} parent types")

        except Exception as e:
            logger.warning(f"Error in transitive parent query: {e}")
            raise
