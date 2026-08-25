"""
Context and diff extraction helpers for the review orchestrator.
"""
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Set

from utils.path_identity import (
    normalize_repository_path,
    repository_paths_match,
)

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


# Chunk count alone is not a prompt budget: AST chunks and repository
# architecture packets can differ by an order of magnitude in size. Keep exact
# structural context first, but bound the total RAG section deterministically.
RAG_CONTEXT_CHAR_BUDGET = max(
    4_000,
    _env_int("REVIEW_RAG_CONTEXT_CHAR_BUDGET", 32_000),
)
RAG_CONTEXT_CHUNK_CHAR_BUDGET = max(
    1_000,
    _env_int("REVIEW_RAG_CONTEXT_CHUNK_CHAR_BUDGET", 12_000),
)


_RAG_POINT_ID_FIELDS = (
    "point_id",
    "pointId",
    "qdrant_point_id",
    "qdrantPointId",
)
_RAG_POSITION_FIELDS = {
    "start_line": ("start_line", "startLine", "line_start", "lineStart"),
    "end_line": ("end_line", "endLine", "line_end", "lineEnd"),
    "start_byte": ("start_byte", "startByte", "byte_start", "byteStart"),
    "end_byte": ("end_byte", "endByte", "byte_end", "byteEnd"),
    "chunk_id": ("chunk_id", "chunkId"),
    "chunk_index": ("chunk_index", "chunkIndex"),
    "sub_chunk_index": ("sub_chunk_index", "subChunkIndex"),
    "parent_chunk_id": ("parent_chunk_id", "parentChunkId"),
    "architecture_source_part": (
        "architecture_source_part",
        "architectureSourcePart",
    ),
}


def _identity_scalar(value: Any) -> Optional[str]:
    """Normalize one scalar without treating zero as missing."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _first_identity_field(
    chunk: Dict[str, Any],
    metadata: Dict[str, Any],
    names: tuple[str, ...],
) -> Optional[str]:
    for container in (chunk, metadata):
        for name in names:
            if name not in container:
                continue
            value = _identity_scalar(container.get(name))
            if value is not None:
                return value
    return None


def rag_retrieval_identity(chunk: Dict[str, Any]) -> tuple[str, ...]:
    """Identify one retrieval occurrence without collapsing equal text elsewhere."""
    metadata = chunk.get("metadata") or {}
    point_id = _first_identity_field(
        chunk,
        metadata,
        _RAG_POINT_ID_FIELDS + ("id",),
    )
    if point_id is not None:
        return ("point", point_id)

    path = normalize_repository_path(
        metadata.get("path")
        or chunk.get("path")
        or chunk.get("file_path")
        or ""
    )
    architecture_key = _identity_scalar(
        metadata.get("architecture_key")
    ) or ""
    locator = {
        canonical_name: value
        for canonical_name, aliases in _RAG_POSITION_FIELDS.items()
        for value in (_first_identity_field(chunk, metadata, aliases),)
        if value is not None
    }
    text = str(chunk.get("text", chunk.get("content", "")))
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if locator or architecture_key:
        return (
            "position",
            path,
            architecture_key,
            json.dumps(
                locator,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            text_digest,
        )
    return ("unlocated", path, text_digest)


def _structural_relations(metadata: Dict[str, Any]) -> tuple[Dict[str, Any], ...]:
    """Read canonical relation evidence, with an old-generation adapter."""
    relation = metadata.get("structural_relation")
    if isinstance(relation, dict):
        return (relation,)
    facts = metadata.get("plugin_graph_facts")
    if not isinstance(facts, list):
        return ()
    return tuple(fact for fact in facts if isinstance(fact, dict))


def _render_unique_plugin_fact_prefix(
    text: str,
    metadata: Dict[str, Any],
    match_type: str,
    visible_fact_lines: Set[str],
) -> tuple[str, tuple[str, ...]]:
    """Render file-level graph facts once, outside stored source text.

    Current indexes keep facts in neutral payload metadata. Deterministic
    retrieval can legitimately return several source chunks from one file,
    but repeating those facts adds no proof. Focused architecture packets already
    render their exact attributes and provenance, so they remain untouched.

    The caller records complete prompt-visible fact lines.
    """
    if (
        match_type in {"architecture_relation", "architecture_related"}
        or metadata.get("architecture_key")
    ):
        return text, ()

    facts = _structural_relations(metadata)
    fact_lines = tuple(
        f"[{fact.get('kind', 'relation')}] "
        f"{fact.get('source', '')} "
        f"{fact.get('relation') or fact.get('kind', '')} "
        f"{fact.get('target', '')}".rstrip()
        for fact in facts
    )
    if not fact_lines:
        return text, ()

    unique_lines = tuple(
        line for line in fact_lines if line not in visible_fact_lines
    )
    parts: List[str] = []
    if unique_lines:
        parts.append(
            "Plugin graph facts (deduplicated within this prompt):\n"
            + "\n".join(unique_lines)
        )
    if text:
        parts.append(text)
    return "\n\n".join(parts), fact_lines


def rag_evidence_id(chunk: Dict[str, Any]) -> str:
    """Return a stable prompt citation ID for one retrieved evidence chunk."""
    identity = rag_retrieval_identity(chunk)
    digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"RAG-{digest}"


def format_rag_context(
    rag_context: Optional[Dict[str, Any]], 
    relevant_files: Optional[Set[str]] = None,
    pr_changed_files: Optional[List[str]] = None,
    deleted_files: Optional[List[str]] = None,
    *,
    max_chars: Optional[int] = None,
    max_chunk_chars: Optional[int] = None,
    current_file_complete_paths: Optional[Set[str]] = None,
    visible_evidence_by_id: Optional[
        Dict[str, tuple[Dict[str, Any], ...]]
    ] = None,
) -> str:
    """
    Format exact repository context selected for this Stage 1 batch.

    Exact chunks are classified into fixed structural tiers. Unused capacity is
    not handed to broader name matches.

    Tier 1 — Exact graph relations and structural dependencies:
        Focused repository-architecture facts, their concrete source, definitions,
        and transitive parent types.
    
    Tier 2 — Direct context:
        Exact source from the current PR overlay.
    
    Args:
        rag_context: RAG response with code chunks
        relevant_files: Batch files associated with this already-retrieved context.
            This formatter does not perform fuzzy relevance filtering.
        pr_changed_files: Files modified in the PR - chunks from these may be stale
        deleted_files: Files deleted in the PR - chunks from these are always stale
        max_chars: Optional deterministic total character budget for this section.
        max_chunk_chars: Optional per-chunk text budget.
        current_file_complete_paths: Batch files whose complete post-change source
            is already present in the prompt. Their RAG chunks are redundant and
            are omitted; truncated/unavailable current files are not included here.
    """
    if not rag_context:
        logger.debug("RAG context is empty or None")
        return ""
    
    # Handle both "chunks" and "relevant_code" keys (RAG API uses "relevant_code")
    chunks = rag_context.get("relevant_code", []) or rag_context.get("chunks", [])
    if not chunks:
        logger.debug("No chunks found in RAG context (keys: %s)", list(rag_context.keys()))
        return ""

    logger.info("Processing %d RAG chunks with tiered budgeting", len(chunks))
    
    # Normalize PR changed files for stale-data detection only
    pr_changed_set = {
        normalize_repository_path(path)
        for path in pr_changed_files or []
        if normalize_repository_path(path)
    }
    
    # Normalize deleted files for filtering (chunks from deleted files are always stale)
    deleted_set = {
        normalize_repository_path(path)
        for path in deleted_files or []
        if normalize_repository_path(path)
    }
    
    # ── Pre-filter: remove stale, deleted, and corrupt chunks ──
    valid_chunks = []
    _seen_content_keys = set()
    skipped_stale = 0
    skipped_deleted = 0
    skipped_redundant_current = 0
    skipped_exact_duplicates = 0
    complete_current_paths = {
        str(path).lstrip("/")
        for path in current_file_complete_paths or set()
        if isinstance(path, str) and path
    }
    
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        path = metadata.get("path") or chunk.get("path") or chunk.get("file_path", "unknown")
        source = chunk.get("_source", chunk.get("source", ""))
        
        # Skip corrupted chunks
        if not path or path in ("unknown", "None"):
            continue
        
        normalized_chunk_path = normalize_repository_path(path)
        structural_relations = _structural_relations(metadata)
        is_exact_architecture_chunk = (
            chunk.get("_match_type", "") in {
                "architecture_relation",
                "architecture_related",
            }
            or bool(metadata.get("architecture_key"))
            or bool(structural_relations)
        )
        if complete_current_paths and any(
            repository_paths_match(normalized_chunk_path, complete_path)
            for complete_path in complete_current_paths
        ) and not is_exact_architecture_chunk:
            # Raw source from a complete current file is redundant. Exact
            # architecture facts are not: they are the deterministic proof
            # used to label and validate plugin-governed claims.
            skipped_redundant_current += 1
            continue
        
        # Filter: chunks from deleted files are ALWAYS stale
        if deleted_set:
            is_from_deleted_file = any(
                repository_paths_match(normalized_chunk_path, deleted_path)
                for deleted_path in deleted_set
            )
            if is_from_deleted_file:
                skipped_deleted += 1
                continue
        
        # Architecture packets use a synthetic storage path.  Their real
        # provenance is the exact set of files in ``architecture_paths``.  A
        # base-branch packet that depends on any PR-modified file is stale even
        # though its synthetic path is unchanged, and must never reach the LLM.
        architecture_paths = {
            value for value in metadata.get("architecture_paths", [])
            if isinstance(value, str) and value
        }
        architecture_touches_modified_file = bool(
            pr_changed_set
            and any(
                repository_paths_match(architecture_path, changed_path)
                for architecture_path in architecture_paths
                for changed_path in pr_changed_set
            )
        )

        # Filter stale chunks from PR-modified files
        if pr_changed_set:
            is_from_modified_file = any(
                repository_paths_match(normalized_chunk_path, changed_path)
                for changed_path in pr_changed_set
            )
            
            is_pr_indexed = (source == "pr_indexed")

            if architecture_touches_modified_file and not is_pr_indexed:
                skipped_stale += 1
                continue

            if is_from_modified_file and not is_pr_indexed:
                skipped_stale += 1
                continue
        
        text = chunk.get("text", chunk.get("content", ""))
        if not text:
            continue
        
        # Deduplicate only repeated retrieval of the same evidence identity.
        # A basename is not an identity: framework repositories commonly have
        # many meaningful files such as module-local ``etc/di.xml`` documents.
        # Likewise, hashing only a prefix can collapse distinct definitions
        # whose headers are identical.
        _content_key = rag_retrieval_identity(chunk)
        if _content_key in _seen_content_keys:
            skipped_exact_duplicates += 1
            continue
        _seen_content_keys.add(_content_key)
        
        valid_chunks.append(chunk)
    
    if not valid_chunks:
        logger.warning(
            "No RAG chunks passed pre-filter (total: %d, skipped_stale: %d, "
            "skipped_deleted: %d, skipped_redundant_current: %d, "
            "skipped_exact_duplicates: %d)",
            len(chunks),
            skipped_stale,
            skipped_deleted,
            skipped_redundant_current,
            skipped_exact_duplicates,
        )
        return ""

    if skipped_exact_duplicates:
        logger.info(
            "RAG pre-filter omitted %d exact duplicate retrieval result(s)",
            skipped_exact_duplicates,
        )
    
    # ── Classify chunks into tiers ──
    TIER_2_BUDGET = 16

    tier_1 = []  # Structural: definitions, transitive parents
    tier_2 = []  # Direct changed-file context
    
    for chunk in valid_chunks:
        match_type = chunk.get("_match_type", "")
        if match_type in (
            "architecture_relation",
            "architecture_related",
            "definition",
            "transitive_parent",
        ):
            # Tier 1: type definitions the reviewed code depends on
            tier_1.append(chunk)
        elif match_type == "changed_file":
            # Tier 2: direct current-revision context
            tier_2.append(chunk)

    tier_1_selected = tier_1
    tier_2_selected = tier_2[:TIER_2_BUDGET]
    
    logger.info(
        f"Tiered assembly: T1={len(tier_1_selected)}/{len(tier_1)} structural, "
        f"T2={len(tier_2_selected)}/{len(tier_2)} direct, "
        f"(skipped: {skipped_stale} stale, {skipped_deleted} deleted, "
        f"{skipped_redundant_current} redundant-current)"
    )
    
    # ── Format selected chunks in tier order ──
    all_selected = tier_1_selected + tier_2_selected

    context_char_budget = max(
        1_000,
        max_chars if max_chars is not None else RAG_CONTEXT_CHAR_BUDGET,
    )
    chunk_char_budget = max(
        256,
        max_chunk_chars
        if max_chunk_chars is not None
        else RAG_CONTEXT_CHUNK_CHAR_BUDGET,
    )

    formatted_parts = []
    included_entry_count = 0
    used_chars = 0
    truncated_chunks = 0
    skipped_for_budget = 0
    visible_plugin_fact_lines: Set[str] = set()
    
    for chunk in all_selected:
        metadata = chunk.get("metadata", {})
        path = metadata.get("path") or chunk.get("path") or chunk.get("file_path", "unknown")
        chunk_type = metadata.get("content_type", metadata.get("type", "code"))
        text = str(chunk.get("text", chunk.get("content", "")))
        text, candidate_plugin_fact_lines = _render_unique_plugin_fact_prefix(
            text,
            metadata,
            str(chunk.get("_match_type", "")),
            visible_plugin_fact_lines,
        )
        
        # Build rich metadata context
        evidence_id = rag_evidence_id(chunk)
        meta_lines = [
            f"Evidence ID: {evidence_id}",
            f"File: {path}",
        ]
        match_type = str(chunk.get("_match_type") or "exact_metadata")
        meta_lines.append(f"Match reason: {match_type}")
        matched_on = str(chunk.get("_matched_on") or "").strip()
        if matched_on:
            meta_lines.append(f"Matched on: {matched_on}")
        
        if metadata.get("namespace"):
            meta_lines.append(f"Namespace: {metadata['namespace']}")
        elif metadata.get("package"):
            meta_lines.append(f"Package: {metadata['package']}")
        
        if metadata.get("primary_name"):
            meta_lines.append(f"Definition: {metadata['primary_name']}")
        elif metadata.get("symbol_names"):
            meta_lines.append(f"Definitions: {', '.join(metadata['symbol_names'][:5])}")
        
        if metadata.get("extends"):
            extends = metadata["extends"]
            meta_lines.append(f"Extends: {', '.join(extends) if isinstance(extends, list) else extends}")
        
        if metadata.get("implements"):
            implements = metadata["implements"]
            meta_lines.append(f"Implements: {', '.join(implements) if isinstance(implements, list) else implements}")
        
        if metadata.get("imports"):
            imports = metadata["imports"]
            if isinstance(imports, list):
                if len(imports) <= 5:
                    meta_lines.append(f"Imports: {'; '.join(imports)}")
                else:
                    meta_lines.append(
                        f"Imports: {'; '.join(imports[:5])}... "
                        f"(+{len(imports) - 5} more)"
                    )
        
        if metadata.get("parent_context"):
            parent_ctx = metadata["parent_context"]
            if isinstance(parent_ctx, list):
                meta_lines.append(f"Parent: {'.'.join(parent_ctx)}")
        
        if chunk_type and chunk_type != "code":
            meta_lines.append(f"Type: {chunk_type}")
        
        meta_text = "\n".join(meta_lines)
        
        entry_prefix = (
            f"### Exact repository context from `{path}`\n"
            f"{meta_text}\n"
            "```\n"
        )
        entry_suffix = "\n```\n"
        separator_chars = 2 if included_entry_count else 0
        available_text_chars = min(
            chunk_char_budget,
            context_char_budget
            - used_chars
            - separator_chars
            - len(entry_prefix)
            - len(entry_suffix),
        )
        if available_text_chars < 256:
            skipped_for_budget += 1
            continue

        bounded_text = text
        if len(text) > available_text_chars:
            truncation_marker = (
                "\n[Context chunk truncated by deterministic prompt budget]"
            )
            retained_chars = max(
                1,
                available_text_chars - len(truncation_marker),
            )
            bounded_text = text[:retained_chars].rstrip() + truncation_marker
            truncated_chunks += 1

        formatted_entry = entry_prefix + bounded_text + entry_suffix
        prospective_chars = used_chars + separator_chars + len(formatted_entry)
        if prospective_chars > context_char_budget:
            skipped_for_budget += 1
            continue
        formatted_parts.append(formatted_entry)
        if visible_evidence_by_id is not None:
            # Record every prompt-visible citation ID. Source chunks remain
            # valid citation sources without plugin-owned graph relationships.
            visible_evidence_by_id.setdefault(evidence_id, ())
            facts = _structural_relations(metadata)
            if facts:
                visible_facts = tuple(
                    dict(fact)
                    for fact in facts
                    if all(
                        isinstance(fact.get(field), str)
                        and fact[field]
                        and fact[field] in bounded_text
                        for field in ("kind", "source", "target")
                    )
                )
                if visible_facts:
                    visible_evidence_by_id[evidence_id] = visible_facts
        visible_plugin_fact_lines.update(
            line
            for line in candidate_plugin_fact_lines
            if line in bounded_text
        )
        included_entry_count += 1
        used_chars = prospective_chars
    
    if not formatted_parts:
        logger.warning(f"No RAG chunks included after tiered selection")
        return ""

    result = "\n".join(formatted_parts)
    logger.info(
        "RAG prompt budget: included=%d/%d chunks, chars=%d/%d, "
        "truncated=%d, omitted_for_budget=%d",
        included_entry_count,
        len(all_selected),
        len(result),
        context_char_budget,
        truncated_chunks,
        skipped_for_budget,
    )
    return result
