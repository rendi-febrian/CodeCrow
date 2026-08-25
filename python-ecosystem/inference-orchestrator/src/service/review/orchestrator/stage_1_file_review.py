"""Stage 1: Parallel file reviews with exact repository context."""
import asyncio
from collections import Counter
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

from model.dtos import ReviewRequestDto
from model.output_schemas import CodeReviewIssue
from model.multi_stage import ReviewPlan, FileReviewBatchOutput
from utils.prompts.prompt_builder import PromptBuilder
from utils.diff_processor import (
    DiffChangeType,
    ProcessedDiff,
    DiffProcessor,
)
from utils.task_context_builder import build_task_context
from utils.dependency_graph import create_smart_batches_async

from utils.llm_response import extract_llm_response_text
from service.review.orchestrator.json_utils import (
    parse_llm_response,
    resolve_structured_output,
)
from service.review.orchestrator.structured_output import (
    format_response_diagnostics,
    invoke_structured_output,
)
from service.review.orchestrator.reconciliation import (
    issue_matches_files,
    format_previous_issues_for_batch,
)
from service.review.orchestrator.context_helpers import (
    format_rag_context,
    rag_evidence_id,
)
from utils.path_identity import (
    normalize_repository_path,
    repository_paths_match,
)
from service.review.orchestrator.stage_helpers import (
    emit_progress,
    emit_status,
    format_project_rules,
)
from service.review.orchestrator.inference_policy import (
    ReviewInferenceProfile,
    build_review_inference_profile,
)
from service.review.orchestrator.stage_1_local_packing import (
    Stage1LocalPackingInput,
    Stage1LocalPackingRuntime,
    Stage1PreparedContext,
    Stage1PromptMaterial,
    Stage1ReviewUnitState,
    _COMPLETE_ADDED_SOURCE_MARKER,
    _DIFF_HUNK_HEADER,
    _DiffReviewChunk,
    _Stage1EvidenceAtom,
    _add_path_lookup,
    _allocate_stage1_invocation_quotas,
    _all_stage1_hunk_ids,
    _apply_compacted_stage_1_omission,
    _cap_stage1_core_batches,
    _chunk_diff_preserving_hunks,
    _chunk_diff_with_ownership,
    _compacted_stage_1_hunk_ids,
    _diff_contains_complete_added_source,
    _diff_limit_reason_allows_full_review,
    _ensure_stage1_review_unit,
    _exact_stage1_diff,
    _expand_oversized_diff_batches as _pack_expand_oversized_diff_batches,
    _expand_oversized_stage1_evidence_batches as _pack_stage1_evidence_batches,
    _fallback_hunk_id,
    _find_diff_file_for_path,
    _is_compacted_stage_1_diff,
    _item_requests_full_diff,
    _joint_stage1_units_for_item as _pack_joint_stage1_units_for_item,
    _lookup_by_path,
    _partition_oversized_stage1_batch as _pack_partition_stage1_batch,
    _path_lookup_keys,
    _repack_stage1_batches_by_rendered_input as _pack_repack_stage1_batches,
    _review_unit_id,
    _reviewable_manifest_context_chars,
    _reviewable_manifest_hunk_ids,
    _split_hunk_by_lines,
    _split_source_at_semantic_line_boundaries,
    _stage1_atom_marker,
    _stage1_evidence_atoms,
    _stage1_item_with_overrides,
    _stage1_unit_from_atoms,
    _text_character_budget,
    pack_stage1_local_batches,
)
from service.review.orchestrator.stage_1_rag_retrieval import (
    Stage1RagState,
    capture_deterministic_retrieval_state as _capture_deterministic_retrieval_state,
    deduplicate_pr_stale_chunks as _deduplicate_pr_stale_chunks,
    deterministic_retrieval_state as _deterministic_retrieval_state,
    fetch_structural_context as _fetch_structural_context,
    has_exact_base_binding as _has_exact_base_binding,
    is_exact_revision_bound as _is_exact_revision_bound,
    rag_response_error as _rag_response_error,
    unwrap_rag_context as _unwrap_rag_context,
)

if TYPE_CHECKING:
    from service.agent import AgentExecutionService
from service.review.orchestrator.stage_1_rag_selection import (
    DETERMINISTIC_RAG_MAX_CHUNKS,
    _flatten_deterministic_context,
)
from service.review.plugin_context import (
    apply_plugin_file_policy,
    review_plugin_context,
)
from service.review.candidate_ledger import CandidateEvidenceLedger
from service.review.prompt_diagnostics import record_prompt_diagnostic
from llm.reasoning_policy import ReasoningEffort, reasoning_request_kwargs

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %s", name, value, default)
        return default


STAGE1_MAX_FILES_PER_BATCH = max(1, _env_int("REVIEW_STAGE1_MAX_FILES_PER_BATCH", 15))
STAGE1_AGENT_MAX_STEPS = max(
    1,
    _env_int("REVIEW_STAGE1_AGENT_MAX_STEPS", 12),
)
STAGE1_AGENT_TOOL_NAMES = frozenset({
    "getBranchFileContent",
    "getRootDirectory",
    "getDirectoryByPath",
    "searchRepositoryCode",
})
STAGE1_BATCH_TOKEN_BUDGET = max(10_000, _env_int("REVIEW_STAGE1_BATCH_TOKEN_BUDGET", 60_000))
STAGE1_DIFF_CHUNK_TOKEN_BUDGET = max(
    8_000,
    _env_int(
        "REVIEW_STAGE1_DIFF_CHUNK_TOKEN_BUDGET",
        STAGE1_BATCH_TOKEN_BUDGET,
    ),
)
# Current source is primary evidence, not optional RAG context. Keep a bounded
# copy in each Stage 1 prompt so small/medium files are reviewed as a coherent
# post-change unit while the full source remains available to verification.
STAGE1_MAX_CURRENT_FILE_CHARS = max(
    2_000,
    _env_int("REVIEW_STAGE1_MAX_CURRENT_FILE_CHARS", 12_000),
)
# Allocate one neutral batch-wide source budget fairly across files. The
# complete diff remains primary evidence, and verification retains full source.
STAGE1_CURRENT_SOURCE_BATCH_CHAR_BUDGET = max(
    8_000,
    _env_int("REVIEW_STAGE1_CURRENT_SOURCE_BATCH_CHAR_BUDGET", 48_000),
)
# These limits apply only to prompt serialization. Full parser metadata remains
# available to batching and deterministic retrieval.
STAGE1_METADATA_CHAR_BUDGET = max(
    4_000,
    _env_int("REVIEW_STAGE1_METADATA_CHAR_BUDGET", 24_000),
)
STAGE1_METADATA_PER_FILE_CHAR_BUDGET = max(
    1_000,
    _env_int("REVIEW_STAGE1_METADATA_PER_FILE_CHAR_BUDGET", 6_000),
)
STRUCTURED_OUTPUT_ENABLED = _env_bool("REVIEW_STRUCTURED_OUTPUT_ENABLED", True)
CLOUDFLARE_STRUCTURED_OUTPUT_ENABLED = _env_bool("REVIEW_CLOUDFLARE_STRUCTURED_OUTPUT_ENABLED", False)
FULL_DIFF_REVIEW_FOCUS = "FULL_DIFF_REVIEW"
_STAGE1_ESTIMATOR_SAFETY_TOKENS = 256


def _stage1_schema_declaration_bytes() -> int:
    try:
        schema = FileReviewBatchOutput.model_json_schema()
    except (AttributeError, TypeError, ValueError):
        return 0
    return len(json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"))


_STAGE1_SCHEMA_DECLARATION_BYTES = _stage1_schema_declaration_bytes()


def _build_stage_1_prepared_context(
    request: ReviewRequestDto,
    processed_diff: Optional[ProcessedDiff],
    is_incremental: bool,
) -> Stage1PreparedContext:
    diff_source = processed_diff
    if is_incremental and request.deltaDiff:
        diff_source = apply_plugin_file_policy(
            request,
            DiffProcessor().process(request.deltaDiff),
        )

    diff_by_path: Dict[str, Optional[Any]] = {}
    if diff_source:
        for diff_file in diff_source.files:
            _add_path_lookup(diff_by_path, diff_file.path, diff_file)

    enrichment_metadata_by_path: Dict[str, Optional[Any]] = {}
    if request.enrichmentData and request.enrichmentData.fileMetadata:
        for meta in request.enrichmentData.fileMetadata:
            _add_path_lookup(enrichment_metadata_by_path, meta.path, meta)

    file_content_by_path: Dict[str, Optional[str]] = {}
    if request.enrichmentData and request.enrichmentData.fileContents:
        for file_content in request.enrichmentData.fileContents:
            if file_content.content and getattr(file_content, "skipped", False) is not True:
                _add_path_lookup(
                    file_content_by_path,
                    file_content.path,
                    file_content.content,
                )

    return Stage1PreparedContext(
        diff_source=diff_source,
        diff_by_path=diff_by_path,
        full_diff_raw=None,
        file_content_by_path=file_content_by_path,
        enrichment_metadata_by_path=enrichment_metadata_by_path,
        task_context=(
            build_task_context(request.taskContext)
            or "No task context available."
        ),
    )


def _bounded_current_file_context(
    content: Optional[str],
    diff_content: str = "",
    *,
    context_lines: int = 20,
    max_chars: Optional[int] = None,
) -> str:
    """Return explicitly labelled, bounded current-source evidence for Stage 1."""
    if not content:
        return "(Current file content unavailable; use the diff evidence.)"
    char_budget = max(
        1,
        min(
            STAGE1_MAX_CURRENT_FILE_CHARS,
            max_chars
            if max_chars is not None
            else STAGE1_MAX_CURRENT_FILE_CHARS,
        ),
    )
    if len(content) <= char_budget:
        return content

    source_lines = content.splitlines()
    windows: List[tuple[int, int]] = []
    for diff_line in diff_content.splitlines():
        match = _DIFF_HUNK_HEADER.match(diff_line)
        if match is None:
            continue
        new_start = max(1, int(match.group("new_start")))
        new_count = int(match.group("new_count") or "1")
        affected_count = max(1, new_count)
        start = max(1, new_start - max(0, context_lines))
        end = min(
            len(source_lines),
            new_start + affected_count - 1 + max(0, context_lines),
        )
        if end >= start:
            windows.append((start, end))

    if windows:
        merged_windows: List[tuple[int, int]] = []
        for start, end in sorted(windows):
            if merged_windows and start <= merged_windows[-1][1] + 1:
                prior_start, prior_end = merged_windows[-1]
                merged_windows[-1] = (prior_start, max(prior_end, end))
            else:
                merged_windows.append((start, end))

        prefix = (
            "[Post-change source windows around reviewed diff hunks; "
            "the complete file remains available to deterministic verification]"
        )
        rendered = [prefix]
        used = len(prefix)
        omitted_windows = 0
        for window_index, (start, end) in enumerate(merged_windows):
            heading = f"\n[Post-change lines {start}-{end}]"
            if used + len(heading) > char_budget:
                omitted_windows = len(merged_windows) - window_index
                break
            rendered.append(heading)
            used += len(heading)
            window_complete = True
            for line_number in range(start, end + 1):
                source_line = (
                    f"\n{line_number:>7}: "
                    f"{source_lines[line_number - 1]}"
                )
                if used + len(source_line) > char_budget:
                    window_complete = False
                    break
                rendered.append(source_line)
                used += len(source_line)
            if not window_complete:
                omitted_windows = len(merged_windows) - window_index
                break

        if omitted_windows:
            marker = (
                f"\n[{omitted_windows} additional post-change source "
                "window(s) omitted by prompt budget]"
            )
            while rendered and used + len(marker) > char_budget:
                removed = rendered.pop()
                used -= len(removed)
            if len(marker) <= char_budget:
                rendered.append(marker)
        return "".join(rendered)

    # A malformed or metadata-only diff has no usable new-side coordinates.
    # Preserve both ends without assigning language-specific meaning to either.
    half = max(1, (char_budget - 160) // 2)
    omitted = len(content) - (half * 2)
    return (
        content[:half]
        + f"\n\n[Current file context truncated: {omitted} characters omitted]\n\n"
        + content[-half:]
    )


def _needs_unbounded_stage_1_diff(diff_source: Optional[ProcessedDiff]) -> bool:
    if not diff_source:
        return False
    for diff_file in diff_source.files:
        if _diff_limit_reason_allows_full_review(diff_file.skip_reason):
            return True
    return False


def _ensure_full_diff_index(prepared_context: Stage1PreparedContext) -> None:
    if prepared_context.full_diff_index_loaded:
        return
    prepared_context.full_diff_index_loaded = True

    raw_diff = prepared_context.full_diff_raw
    if not raw_diff:
        return

    # Stage 1 can split very large diffs into multiple bounded prompts. Parse
    # the original hunks only when Stage 0 explicitly asks for full-diff review.
    raw_diff_source = DiffProcessor().process(raw_diff)
    for diff_file in raw_diff_source.files:
        _add_path_lookup(prepared_context.full_diff_by_path, diff_file.path, diff_file)
    logger.info(
        "Stage 1 prepared unbounded raw diff index for %d file(s)",
        len(raw_diff_source.files),
    )


def _iter_batch_enrichment_metadata(
    request: ReviewRequestDto,
    batch_file_paths: List[str],
    prepared_context: Optional[Stage1PreparedContext],
) -> List[Any]:
    if not request.enrichmentData or not request.enrichmentData.fileMetadata:
        return []

    result: List[Any] = []
    seen: set[int] = set()
    if prepared_context:
        for path in batch_file_paths:
            meta = _lookup_by_path(prepared_context.enrichment_metadata_by_path, path)
            if meta is not None and id(meta) not in seen:
                result.append(meta)
                seen.add(id(meta))

    if len(result) >= len(batch_file_paths):
        return result

    # Collision/path-format fallback.
    for meta in request.enrichmentData.fileMetadata:
        if id(meta) in seen:
            continue
        if any(
            repository_paths_match(meta.path, batch_path)
            for batch_path in batch_file_paths
        ):
            result.append(meta)
            seen.add(id(meta))

    return result


def _format_batch_metadata_json(
    batch_metadata: List[Any],
    *,
    max_chars: Optional[int] = None,
    max_chars_per_file: Optional[int] = None,
) -> str:
    """Serialize arbitrary parser metadata within a deterministic prompt budget.

    This projection is deliberately schema-neutral so analysis-plugin fields do
    not require host-side dispatch. Omission markers distinguish a bounded
    prompt view from evidence that a metadata value is absent.
    """
    if not batch_metadata:
        return ""

    metadata_payload = [_metadata_to_payload(meta) for meta in batch_metadata]
    total_budget = max(256, max_chars or STAGE1_METADATA_CHAR_BUDGET)
    configured_per_file = max(
        256,
        max_chars_per_file or STAGE1_METADATA_PER_FILE_CHAR_BUDGET,
    )
    # Reserve JSON list punctuation and distribute the hard total cap evenly.
    per_file_budget = min(
        configured_per_file,
        max(256, (total_budget - 2) // len(metadata_payload)),
    )
    projected = [
        _bounded_metadata_payload(payload, per_file_budget)
        for payload in metadata_payload
    ]
    rendered = json.dumps(
        projected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    # Enforce the total boundary independently of JSON punctuation and unusual
    # payload shapes.
    while len(rendered) > total_budget and per_file_budget > 256:
        overflow_per_file = max(
            1,
            (len(rendered) - total_budget + len(projected) - 1)
            // len(projected),
        )
        per_file_budget = max(256, per_file_budget - overflow_per_file)
        projected = [
            _bounded_metadata_payload(payload, per_file_budget)
            for payload in metadata_payload
        ]
        rendered = json.dumps(
            projected,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    if len(rendered) > total_budget:
        projected = [
            _metadata_identity_fallback(payload, per_file_budget)
            for payload in metadata_payload
        ]
        rendered = json.dumps(
            projected,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    omitted_entries = sum(
        _count_metadata_omission_markers(value) for value in projected
    )
    if omitted_entries:
        logger.info(
            "Stage 1 parser metadata prompt view bounded to %d chars "
            "(rendered=%d, omission_markers=%d); full metadata retained for retrieval",
            total_budget,
            len(rendered),
            omitted_entries,
        )
    return rendered


def _bounded_metadata_payload(payload: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    canonical = _project_metadata_detail(payload, detail_limit=None)
    if _json_char_length(canonical) <= max_chars:
        return canonical

    low = 1
    high = max(1, _metadata_detail_ceiling(payload))
    best: Optional[Dict[str, Any]] = None
    while low <= high:
        detail_limit = (low + high) // 2
        candidate = _project_metadata_detail(payload, detail_limit=detail_limit)
        if _json_char_length(candidate) <= max_chars:
            best = candidate
            low = detail_limit + 1
        else:
            high = detail_limit - 1

    if best is not None:
        return best
    return _metadata_identity_fallback(payload, max_chars)


def _project_metadata_detail(value: Any, detail_limit: Optional[int]) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _project_metadata_detail(value[key], detail_limit)
            for key in sorted(value, key=lambda item: str(item))
            if value[key] is not None
        }
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if isinstance(value, set):
            items.sort(
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            )
        selected = items if detail_limit is None else items[:detail_limit]
        result = [
            _project_metadata_detail(item, detail_limit)
            for item in selected
        ]
        omitted = len(items) - len(selected)
        if omitted:
            result.append({"_codecrowOmittedItems": omitted})
        return result
    if isinstance(value, str) and detail_limit is not None:
        string_limit = max(64, detail_limit * 64)
        if len(value) > string_limit:
            omitted = len(value) - string_limit
            return value[:string_limit] + f"… [CodeCrow omitted {omitted} chars]"
    return value


def _metadata_detail_ceiling(value: Any) -> int:
    if isinstance(value, dict):
        return max(
            [1] + [_metadata_detail_ceiling(nested) for nested in value.values()]
        )
    if isinstance(value, (list, tuple, set)):
        return max(
            [len(value), 1]
            + [_metadata_detail_ceiling(nested) for nested in value]
        )
    if isinstance(value, str):
        return max(1, (len(value) + 63) // 64)
    return 1


def _metadata_identity_fallback(
    payload: Dict[str, Any],
    max_chars: int,
) -> Dict[str, Any]:
    identity: Dict[str, Any] = {
        "_codecrowMetadataOmitted": {
            "sourceFieldCount": len(payload),
            "reason": "prompt-character-budget",
        }
    }
    for key in ("path", "language", "namespace", "parentClass"):
        value = payload.get(key)
        if value is None:
            continue
        text = str(value)
        candidate = dict(identity)
        candidate[key] = text
        if _json_char_length(candidate) <= max_chars:
            identity = candidate
            continue

        low = 0
        high = len(text)
        best = ""
        while low <= high:
            prefix_chars = (low + high) // 2
            bounded_text = text[:prefix_chars] + (
                "…" if prefix_chars < len(text) else ""
            )
            candidate = dict(identity)
            candidate[key] = bounded_text
            if _json_char_length(candidate) <= max_chars:
                best = bounded_text
                low = prefix_chars + 1
            else:
                high = prefix_chars - 1
        if best:
            identity[key] = best
    return identity


def _json_char_length(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


def _count_metadata_omission_markers(value: Any) -> int:
    if isinstance(value, dict):
        own = int(
            "_codecrowOmittedItems" in value
            or "_codecrowMetadataOmitted" in value
        )
        return own + sum(
            _count_metadata_omission_markers(nested)
            for nested in value.values()
        )
    if isinstance(value, list):
        return sum(_count_metadata_omission_markers(item) for item in value)
    if isinstance(value, str) and "[CodeCrow omitted " in value:
        return 1
    return 0


def _metadata_to_payload(meta: Any) -> Dict[str, Any]:
    if hasattr(meta, "model_dump"):
        return meta.model_dump(mode="json", by_alias=False, exclude_none=True)
    if isinstance(meta, dict):
        return {
            key: value
            for key, value in meta.items()
            if value is not None
        }
    return {
        key: value
        for key, value in vars(meta).items()
        if not key.startswith("_") and value is not None
    }


def _extract_metadata_identifiers(
    batch_metadata: List[Any],
    limit: int = 200,
) -> Optional[List[str]]:
    """Collect only parser fields that prove a symbol relationship.

    Paths, languages, namespaces, diagnostics, and arbitrary plugin strings are
    not definition lookup keys. Framework-specific relations reach the prompt
    through typed plugin graph facts instead of this generic symbol expansion.
    """
    identifier_fields = (
        "imports",
        "extends",
        "extendsClasses",
        "implements",
        "implementsInterfaces",
        "parent_class",
        "parentClass",
    )
    seen = set()
    identifiers: List[str] = []

    def visit(value: Any) -> None:
        if len(identifiers) >= limit or value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if text and text not in seen:
                seen.add(text)
                identifiers.append(text)
            return
        if isinstance(value, (list, tuple, set)):
            for nested in value:
                visit(nested)
            return

    for meta in batch_metadata:
        payload = _metadata_to_payload(meta)
        for field_name in identifier_fields:
            if field_name in payload:
                visit(payload[field_name])

    return identifiers or None




def _supports_structured_output(llm) -> bool:
    if not STRUCTURED_OUTPUT_ENABLED:
        return False
    if CLOUDFLARE_STRUCTURED_OUTPUT_ENABLED:
        return True

    from utils.llm_delegate import llm_class_names

    class_names = llm_class_names(llm)
    if "ChatCloudflareOpenAI" in class_names:
        return False
    return True


def _positive_int_or_default(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _relationship_type_text(relationship: Any) -> str:
    value = getattr(relationship, "relationshipType", "DEPENDENCY")
    return str(getattr(value, "value", value) or "DEPENDENCY")


def _stage1_boundary_context(
    request: ReviewRequestDto,
    batch_items: Sequence[Dict[str, Any]],
    batch_file_paths: Sequence[str],
) -> str:
    """Render exact graph edges cut by input packing, without source guessing."""
    batch_paths = {
        normalize_repository_path(path)
        for path in batch_file_paths
        if normalize_repository_path(path)
    }
    changed_paths = {
        normalize_repository_path(path)
        for path in (getattr(request, "changedFiles", None) or [])
        if normalize_repository_path(path)
    }
    records: Dict[tuple[str, str, str, str], Dict[str, str]] = {}
    enrichment = getattr(request, "enrichmentData", None)
    for relationship in getattr(enrichment, "relationships", None) or []:
        source = normalize_repository_path(
            getattr(relationship, "sourceFile", "")
        )
        target = normalize_repository_path(
            getattr(relationship, "targetFile", "")
        )
        if not source or not target:
            continue
        if (source in batch_paths) == (target in batch_paths):
            continue
        if changed_paths and not ({source, target} <= changed_paths):
            continue
        relation_type = _relationship_type_text(relationship)
        matched_on = str(getattr(relationship, "matchedOn", "") or "")
        key = (source, target, relation_type, matched_on)
        records[key] = {
            "source": source,
            "target": target,
            "type": relation_type,
            "matchedOn": matched_on,
        }

    # RAG-discovered graph edges may not exist in enrichment. Preserve their
    # endpoints as neutral dependency facts rather than silently losing them.
    for item in batch_items:
        file_info = item.get("file")
        source = normalize_repository_path(getattr(file_info, "path", ""))
        for related_path in item.get("related_files", ()) or ():
            target = normalize_repository_path(related_path)
            if not source or not target or target in batch_paths:
                continue
            ordered = tuple(sorted((source, target)))
            key = (ordered[0], ordered[1], "DEPENDENCY", "")
            records.setdefault(key, {
                "source": ordered[0],
                "target": ordered[1],
                "type": "DEPENDENCY",
                "matchedOn": "",
            })

    parts: List[str] = []
    if records:
        parts.append(
            "Exact cross-pack relationship records (deterministic; no source "
            "was inferred or truncated):\n"
            + json.dumps(
                [records[key] for key in sorted(records)],
                ensure_ascii=False,
                indent=2,
            )
        )
    diagnostics = sorted({
        str(item.get("_stage1_budget_diagnostic") or "")
        for item in batch_items
        if item.get("_stage1_budget_diagnostic")
    })
    parts.extend(diagnostics)
    return "\n\n".join(parts)


def _prepare_stage1_prompt_material(
    request: ReviewRequestDto,
    batch_items: List[Dict[str, Any]],
    prepared_context: Stage1PreparedContext,
    is_incremental: bool,
) -> Stage1PromptMaterial:
    """Build the exact non-RAG prompt material once for packing and review."""
    batch_files_data: List[Dict[str, Any]] = []
    batch_file_paths: List[str] = []
    complete_current_file_paths: set[str] = set()
    available_current_source_count = sum(
        1
        for item in batch_items
        if _lookup_by_path(
            prepared_context.file_content_by_path,
            item["file"].path,
        )
    )
    current_source_per_file_budget = min(
        STAGE1_MAX_CURRENT_FILE_CHARS,
        max(
            1,
            STAGE1_CURRENT_SOURCE_BATCH_CHAR_BUDGET
            // max(1, available_current_source_count),
        ),
    )

    for item in batch_items:
        file_info = item["file"]
        batch_file_paths.append(file_info.path)
        current_file_content = _lookup_by_path(
            prepared_context.file_content_by_path,
            file_info.path,
        )
        diff_file = _find_diff_file_for_path(
            prepared_context,
            file_info.path,
            use_full_diff=_item_requests_full_diff(item),
        )
        if "_diff_override" in item:
            file_diff = str(item.get("_diff_override") or "")
        else:
            file_diff = ""
        if "_diff_override" not in item and diff_file:
            file_diff = diff_file.content
        if file_diff:
            chunk_total = int(item.get("_diff_chunk_total") or 0)
            if chunk_total > 1:
                chunk_index = int(item.get("_diff_chunk_index") or 1)
                file_diff = (
                    f"[Large diff segment {chunk_index}/{chunk_total} for "
                    f"{file_info.path}. All segments are reviewed independently "
                    "and merged after Stage 1.]\n"
                    f"{file_diff}"
                )
        change_type = (
            diff_file.change_type
            if diff_file is not None
            else DiffChangeType.MODIFIED
        )
        source_override = item.get("_current_source_override")
        if isinstance(source_override, str):
            current_code = source_override
        else:
            complete_added_source_in_diff = (
                change_type is DiffChangeType.ADDED
                and int(item.get("_diff_chunk_total") or 0) <= 1
                and _diff_contains_complete_added_source(
                    current_file_content,
                    file_diff,
                )
            )
            if complete_added_source_in_diff:
                current_code = _COMPLETE_ADDED_SOURCE_MARKER
                complete_current_file_paths.add(file_info.path)
            else:
                current_code = _bounded_current_file_context(
                    current_file_content,
                    file_diff,
                    max_chars=current_source_per_file_budget,
                )
                if (
                    current_file_content
                    and len(current_file_content) <= current_source_per_file_budget
                ):
                    complete_current_file_paths.add(file_info.path)

        batch_files_data.append({
            "path": file_info.path,
            "type": change_type.value.upper(),
            "focus_areas": file_info.focus_areas,
            "current_code": current_code,
            "diff": file_diff or "(Diff unavailable)",
            "is_incremental": is_incremental,
        })

    singleton_item = batch_items[0] if len(batch_items) == 1 else None
    if singleton_item is not None and "_project_rules_override" in singleton_item:
        project_rules = str(singleton_item.get("_project_rules_override") or "")
    else:
        project_rules = format_project_rules(request.projectRules, batch_file_paths)
    batch_metadata = _iter_batch_enrichment_metadata(
        request,
        batch_file_paths,
        prepared_context,
    )
    enrichment_identifiers = (
        _extract_metadata_identifiers(batch_metadata)
        if batch_metadata
        else None
    )
    previous_issues_for_batch = ""
    previous_issues = getattr(request, "previousCodeAnalysisIssues", None)
    if isinstance(previous_issues, (list, tuple)) and previous_issues:
        relevant_previous_issues = [
            issue
            for issue in previous_issues
            if issue_matches_files(issue, batch_file_paths)
        ]
        if relevant_previous_issues:
            previous_issues_for_batch = format_previous_issues_for_batch(
                relevant_previous_issues
            )

    if singleton_item is not None and "_previous_issues_override" in singleton_item:
        previous_issues_for_batch = str(
            singleton_item.get("_previous_issues_override") or ""
        )
    if singleton_item is not None and "_metadata_override" in singleton_item:
        file_metadata_text = str(singleton_item.get("_metadata_override") or "")
    else:
        file_metadata_text = _format_batch_metadata_json(batch_metadata)
    if singleton_item is not None and "_task_context_override" in singleton_item:
        task_context = str(singleton_item.get("_task_context_override") or "")
    else:
        task_context = prepared_context.task_context
    plugin_context_override = None
    if singleton_item is not None and "_plugin_context_override" in singleton_item:
        plugin_context_override = str(
            singleton_item.get("_plugin_context_override") or ""
        )
    if singleton_item is not None and "_boundary_context_override" in singleton_item:
        boundary_context = str(
            singleton_item.get("_boundary_context_override") or ""
        )
    else:
        boundary_context = _stage1_boundary_context(
            request,
            batch_items,
            batch_file_paths,
        )

    return Stage1PromptMaterial(
        request=request,
        batch_items=batch_items,
        batch_files_data=batch_files_data,
        batch_file_paths=batch_file_paths,
        complete_current_file_paths=complete_current_file_paths,
        current_source_per_file_budget=current_source_per_file_budget,
        batch_metadata=batch_metadata,
        enrichment_identifiers=enrichment_identifiers,
        project_rules=project_rules,
        previous_issues_for_batch=previous_issues_for_batch,
        file_metadata_text=file_metadata_text,
        task_context=task_context,
        plugin_context_override=plugin_context_override,
        prepared_context=prepared_context,
        is_incremental=is_incremental,
        boundary_context=boundary_context,
    )


def _render_stage1_prompt(
    material: Stage1PromptMaterial,
    rag_context_text: str,
    *,
    visible_evidence_by_id: Optional[
        Dict[str, tuple[Dict[str, Any], ...]]
    ] = None,
    use_mcp_tools: Optional[bool] = None,
) -> tuple[str, str]:
    if material.plugin_context_override is not None:
        plugin_context_text = material.plugin_context_override
    else:
        try:
            plugin_context_text = review_plugin_context(
                material.request,
                material.batch_file_paths,
                visible_evidence_by_id=visible_evidence_by_id or {},
            )
        except Exception as exception:
            logger.warning(
                "Optional Stage 1 plugin prompt context is unavailable; "
                "continuing with local and RAG evidence: %s",
                exception,
            )
            plugin_context_text = ""
    prompt = PromptBuilder.build_stage_1_batch_prompt(
        files=material.batch_files_data,
        priority=(
            material.batch_items[0]["priority"]
            if material.batch_items
            else "MEDIUM"
        ),
        project_rules=material.project_rules,
        file_outlines=material.file_metadata_text,
        rag_context=rag_context_text,
        is_incremental=material.is_incremental,
        previous_issues=material.previous_issues_for_batch,
        all_pr_files=getattr(material.request, "changedFiles", None),
        deleted_files=getattr(material.request, "deletedFiles", None),
        task_context=material.task_context,
        use_mcp_tools=(
            bool(getattr(material.request, "useMcpTools", False))
            if use_mcp_tools is None
            else use_mcp_tools
        ),
        target_branch=str(
            getattr(material.request, "localRepoRevision", None)
            or material.request.get_target_head_commit_hash()
            or getattr(material.request, "targetBranchName", "")
            or ""
        ),
        vcs_workspace=str(
            getattr(material.request, "projectVcsWorkspace", "") or ""
        ),
        vcs_repo_slug=str(
            getattr(material.request, "projectVcsRepoSlug", "") or ""
        ),
        plugin_context=plugin_context_text,
        batch_boundary_context=material.boundary_context,
    )
    return prompt, plugin_context_text


def _estimated_prompt_tokens(prompt: str) -> int:
    """Estimate rendered UTF-8 input plus the structured-output declaration."""
    request_bytes = (
        len(prompt.encode("utf-8"))
        + _STAGE1_SCHEMA_DECLARATION_BYTES
    )
    return max(
        1,
        (request_bytes + 3) // 4 + _STAGE1_ESTIMATOR_SAFETY_TOKENS,
    )


# ── Batching ──────────────────────────────────────────────────


def chunk_files(
    file_groups: List[Any],
    max_files_per_batch: int = STAGE1_MAX_FILES_PER_BATCH,
    processed_diff: Optional[ProcessedDiff] = None,
    max_allowed_tokens: int = STAGE1_BATCH_TOKEN_BUDGET,
    token_cost_by_path: Optional[Dict[str, int]] = None,
) -> List[List[Dict[str, Any]]]:
    estimated_cost_by_path = {
        diff_file.path: (len(diff_file.content.encode("utf-8")) // 4) + 1000
        for diff_file in getattr(processed_diff, "files", [])
    }
    estimated_cost_by_path.update(token_cost_by_path or {})
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_tokens = 0
    for group in file_groups:
        for f in group.files:
            file_tokens = estimated_cost_by_path.get(f.path, 2000)
            if current and (
                len(current) >= max_files_per_batch
                or current_tokens + file_tokens > max_allowed_tokens
            ):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append({"file": f, "priority": group.priority})
            current_tokens += file_tokens
    if current:
        batches.append(current)
    return batches


def _stage1_batch_token_limit(request: ReviewRequestDto) -> int:
    model_context_tokens = _positive_int_or_default(
        getattr(request, "maxAllowedTokens", None),
        200000,
    )
    return min(
        max(10_000, model_context_tokens - 20_000),
        STAGE1_BATCH_TOKEN_BUDGET,
    )


async def create_smart_batches_wrapper(
    file_groups: List[Any],
    processed_diff: Optional[ProcessedDiff],
    request: ReviewRequestDto,
    rag_client,
    max_files_per_batch: int = 15,
    prepared_context: Optional[Stage1PreparedContext] = None,
    is_incremental: bool = False,
) -> List[List[Dict[str, Any]]]:
    branches = []
    rag_branch = request.get_rag_branch()
    base_branch = request.get_rag_base_branch()
    if rag_branch:
        branches.append(rag_branch)
    if base_branch and base_branch not in branches:
        branches.append(base_branch)
    batching_rag_client = rag_client
    if not branches:
        logger.warning(
            "Stage 1 batching has no authoritative target branch; "
            "using local/enrichment grouping without a repository RAG lookup"
        )
        batching_rag_client = None
    exact_receipt_values = tuple(
        value
        for value in (
            getattr(request, "ragBaseGenerationManifestSha256", None),
            getattr(request, "ragPrGenerationFingerprint", None),
            getattr(
                request,
                "ragPrOverlayGenerationManifestSha256",
                None,
            ),
        )
        if isinstance(value, str) and value
    )
    if any(exact_receipt_values):
        logger.info(
            "Stage 1 smart-batching RAG discovery disabled while exact "
            "base/overlay generation receipts are active"
        )
        batching_rag_client = None

    enrichment_data = getattr(request, 'enrichmentData', None)
    token_cost_by_path: Optional[Dict[str, int]] = None
    shared_prompt_tokens = 0
    if prepared_context is not None:
        token_cost_by_path = {}
        shared_material = _prepare_stage1_prompt_material(
            request,
            [],
            prepared_context,
            is_incremental,
        )
        shared_prompt, _ = _render_stage1_prompt(shared_material, "")
        shared_prompt_tokens = _estimated_prompt_tokens(shared_prompt)
        for group in file_groups:
            for review_file in group.files:
                material = _prepare_stage1_prompt_material(
                    request,
                    [{"file": review_file, "priority": group.priority}],
                    prepared_context,
                    is_incremental,
                )
                local_prompt, _ = _render_stage1_prompt(material, "")
                token_cost_by_path[review_file.path] = (
                    max(
                        1,
                        _estimated_prompt_tokens(local_prompt)
                        - shared_prompt_tokens,
                    )
                )

    try:
        # Preserve coherent small/medium PRs while keeping every prompt inside
        # the configured model budget. Large components split only when their
        # actual diff cost or the explicit file ceiling requires it.
        model_context_tokens = _positive_int_or_default(
            getattr(request, "maxAllowedTokens", None),
            200000,
        )
        model_safe_limit = max(10_000, model_context_tokens - 20_000)
        batch_token_limit = _stage1_batch_token_limit(request)
        graph_content_limit = max(
            1_000,
            batch_token_limit - shared_prompt_tokens,
        )
        if batch_token_limit < model_safe_limit:
            logger.info(
                "Stage 1 batch token budget capped at %d tokens "
                "(model-safe limit=%d, env REVIEW_STAGE1_BATCH_TOKEN_BUDGET)",
                batch_token_limit,
                model_safe_limit,
            )

        batches = await create_smart_batches_async(
            file_groups=file_groups,
            workspace=request.projectWorkspace,
            project=request.projectNamespace,
            branches=branches,
            rag_client=batching_rag_client,
            max_batch_size=max_files_per_batch,
            enrichment_data=enrichment_data,
            max_allowed_tokens=graph_content_limit,
            processed_diff=processed_diff,
            token_cost_by_path=token_cost_by_path,
        )
        total_files = sum(len(b) for b in batches)
        related_files = sum(1 for b in batches for f in b if f.get('has_relationships'))
        enrichment_source = "enrichment data" if enrichment_data else "RAG discovery"
        logger.info(
            f"Smart batching ({enrichment_source}): {total_files} files in "
            f"{len(batches)} batches, {related_files} files have cross-file relationships"
        )
        return batches
    except Exception as e:
        logger.warning(f"Smart batching failed, falling back to capacity batching: {e}")
        return chunk_files(
            file_groups,
            max_files_per_batch,
            processed_diff=processed_diff,
            max_allowed_tokens=max(
                1_000,
                _stage1_batch_token_limit(request) - shared_prompt_tokens,
            ),
            token_cost_by_path=token_cost_by_path,
        )


def _complete_stage1_plugin_context(
    request: ReviewRequestDto,
    path: str,
) -> str:
    try:
        # Packing needs the complete deterministic contribution. RAG visibility
        # may later reduce evidence targets, but it must never reveal a larger
        # plugin block than the packer measured.
        return review_plugin_context(request, [path])
    except Exception as exception:
        logger.warning(
            "Optional Stage 1 plugin prompt context is unavailable during "
            "lossless packing; continuing without it: %s",
            exception,
        )
        return ""


def _stage1_material_prompt_tokens(material: Stage1PromptMaterial) -> int:
    prompt, _ = _render_stage1_prompt(material, "")
    return _estimated_prompt_tokens(prompt)


def _stage1_local_packing_runtime() -> Stage1LocalPackingRuntime:
    return Stage1LocalPackingRuntime(
        prepare_material=_prepare_stage1_prompt_material,
        material_prompt_tokens=_stage1_material_prompt_tokens,
        complete_plugin_context=_complete_stage1_plugin_context,
    )


def _expand_oversized_diff_batches(
    batches: List[List[Dict[str, Any]]],
    prepared_context: Stage1PreparedContext,
    diff_chunk_token_budget: int = STAGE1_DIFF_CHUNK_TOKEN_BUDGET,
) -> List[List[Dict[str, Any]]]:
    return _pack_expand_oversized_diff_batches(
        batches,
        prepared_context,
        diff_chunk_token_budget=diff_chunk_token_budget,
    )


def _joint_stage1_units_for_item(
    item: Dict[str, Any],
    request: ReviewRequestDto,
    prepared_context: Stage1PreparedContext,
    is_incremental: bool,
    token_budget: int,
    max_units: int = 3,
) -> List[Dict[str, Any]]:
    """Backward-compatible facade over the local-packing runtime boundary."""
    return _pack_joint_stage1_units_for_item(
        item,
        request,
        prepared_context,
        is_incremental,
        token_budget,
        max_units=max_units,
        runtime=_stage1_local_packing_runtime(),
    )


def _expand_oversized_stage1_evidence_batches(
    batches: List[List[Dict[str, Any]]],
    request: ReviewRequestDto,
    prepared_context: Stage1PreparedContext,
    is_incremental: bool,
    token_budget: int,
    max_units_per_item: int = 3,
    max_total_batches: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    """Preserve the historical adapter while using typed packing inputs."""
    return pack_stage1_local_batches(
        batches,
        Stage1LocalPackingInput(
            request=request,
            prepared_context=prepared_context,
            is_incremental=is_incremental,
            token_budget=token_budget,
        ),
        _stage1_local_packing_runtime(),
        max_units_per_item=max_units_per_item,
        max_total_batches=max_total_batches,
    )


def _expand_oversized_current_source_batches(
    batches: List[List[Dict[str, Any]]],
    request: ReviewRequestDto,
    prepared_context: Stage1PreparedContext,
    is_incremental: bool,
    token_budget: int,
) -> List[List[Dict[str, Any]]]:
    """Compatibility wrapper for the joint local-evidence packer."""
    return _expand_oversized_stage1_evidence_batches(
        batches,
        request,
        prepared_context,
        is_incremental,
        token_budget,
    )


def _repack_stage1_batches_by_rendered_input(
    batches: List[List[Dict[str, Any]]],
    request: ReviewRequestDto,
    prepared_context: Stage1PreparedContext,
    is_incremental: bool,
    token_budget: int,
) -> List[List[Dict[str, Any]]]:
    return _pack_repack_stage1_batches(
        batches,
        request,
        prepared_context,
        is_incremental,
        token_budget,
        runtime=_stage1_local_packing_runtime(),
    )


def _partition_oversized_stage1_batch(
    batch_items: List[Dict[str, Any]],
    request: ReviewRequestDto,
    prepared_context: Stage1PreparedContext,
    is_incremental: bool,
    token_budget: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return _pack_partition_stage1_batch(
        batch_items,
        request,
        prepared_context,
        is_incremental,
        token_budget,
        runtime=_stage1_local_packing_runtime(),
    )


# ── RAG Context ───────────────────────────────────────────────

async def fetch_batch_rag_context(
    rag_client,
    request: ReviewRequestDto,
    batch_file_paths: List[str],
    pr_indexed: bool = False,
    enrichment_identifiers: Optional[List[str]] = None,
    rag_state: Optional[Stage1RagState] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch exact structural evidence through the isolated retrieval boundary."""
    return await _fetch_structural_context(
        rag_client,
        request,
        batch_file_paths,
        flatten_context=_flatten_deterministic_context,
        max_chunks=DETERMINISTIC_RAG_MAX_CHUNKS,
        pr_indexed=pr_indexed,
        enrichment_identifiers=enrichment_identifiers,
        rag_state=rag_state,
    )


# ── Batch Review ──────────────────────────────────────────────


async def execute_stage_1_file_reviews(
    llm,
    request: ReviewRequestDto,
    plan: ReviewPlan,
    rag_client,
    processed_diff: Optional[ProcessedDiff] = None,
    is_incremental: bool = False,
    max_parallel: int = 5,
    event_callback: Optional[Callable[[Dict], None]] = None,
    pr_indexed: bool = False,
    fallback_llm=None,
    rag_state: Optional[Stage1RagState] = None,
    review_unit_state: Optional[Stage1ReviewUnitState] = None,
    candidate_ledger: Optional[CandidateEvidenceLedger] = None,
    inference_profile: Optional[ReviewInferenceProfile] = None,
    agent_service: Optional["AgentExecutionService"] = None,
) -> List[CodeReviewIssue]:
    prepared_context = _build_stage_1_prepared_context(request, processed_diff, is_incremental)
    inference_profile = inference_profile or build_review_inference_profile(
        request,
        processed_diff,
    )
    rag_state = rag_state or Stage1RagState()
    review_unit_state = review_unit_state or Stage1ReviewUnitState()
    batches = await create_smart_batches_wrapper(
        file_groups=plan.file_groups,
        processed_diff=prepared_context.diff_source,
        request=request,
        rag_client=rag_client,
        max_files_per_batch=STAGE1_MAX_FILES_PER_BATCH,
        prepared_context=prepared_context,
        is_incremental=is_incremental,
    )
    stage1_token_budget = _stage1_batch_token_limit(request)
    batches = _repack_stage1_batches_by_rendered_input(
        batches,
        request,
        prepared_context,
        is_incremental,
        stage1_token_budget,
    )
    batches = _cap_stage1_core_batches(
        batches,
        prepared_context,
        inference_profile.invocation_cap("stage_1_total"),
    )
    batches = _expand_oversized_stage1_evidence_batches(
        batches,
        request,
        prepared_context,
        is_incremental,
        stage1_token_budget,
        max_units_per_item=inference_profile.invocation_cap(
            "stage_1_per_unit"
        ),
        max_total_batches=inference_profile.invocation_cap("stage_1_total"),
    )
    admitted_stage1_invocations = _allocate_stage1_invocation_quotas(
        batches,
        inference_profile.invocation_cap("stage_1_total"),
        inference_profile.invocation_cap("stage_1_per_unit"),
    )
    logger.info(
        "Stage 1 provider-call ceiling: semantic_invocations=%d, "
        "primary_calls=%d, direct_output_recovery_calls<=%d, total_calls<=%d",
        admitted_stage1_invocations,
        admitted_stage1_invocations,
        admitted_stage1_invocations,
        admitted_stage1_invocations * 2,
    )
    review_unit_state.register_batches(batches)

    total_review_units = sum(len(batch) for batch in batches)
    unique_file_paths = {
        item["file"].path
        for batch in batches
        for item in batch
        if item.get("file") is not None
    }
    total_files = len(unique_file_paths)
    related_batches = sum(1 for b in batches if any(f.get('has_relationships') for f in b))
    logger.info(
        f"Stage 1: Processing {total_files} files as {total_review_units} review units "
        f"in {len(batches)} batches "
        f"({related_batches} batches with cross-file relationships)"
    )

    all_issues: List[CodeReviewIssue] = []
    if not batches:
        logger.info("Stage 1 Complete: no batches to review")
        return all_issues

    max_parallel = max(1, max_parallel)
    semaphore = asyncio.Semaphore(max_parallel)
    started_at = time.time()
    batch_results: Dict[int, List[CodeReviewIssue]] = {}
    completed_batches = 0

    logger.info(
        "Stage 1: scheduling %d batches with bounded concurrency=%d",
        len(batches),
        max_parallel,
    )

    async def _run_batch(
        batch_idx: int,
        batch: List[Dict[str, Any]],
    ) -> tuple[int, List[CodeReviewIssue], tuple[str, ...]]:
        unit_ids = review_unit_state.unit_ids_for_batch(batch_idx, batch)
        async with semaphore:
            batch_paths = [item["file"].path for item in batch]
            has_rels = any(item.get('has_relationships') for item in batch)
            logger.debug(f"Batch {batch_idx}: {batch_paths} (cross-file relationships: {has_rels})")
            result = await _review_batch_with_timing(
                batch_idx, llm, request, batch, rag_client, prepared_context,
                is_incremental, pr_indexed,
                fallback_llm=fallback_llm,
                rag_state=rag_state,
                candidate_ledger=candidate_ledger,
                agent_service=agent_service,
                event_callback=event_callback,
            )
            return batch_idx, result, unit_ids

    tasks = [
        asyncio.create_task(_run_batch(batch_idx, batch))
        for batch_idx, batch in enumerate(batches, start=1)
    ]

    for completed_task in asyncio.as_completed(tasks):
        try:
            batch_num, res, unit_ids = await completed_task
            review_unit_state.mark_completed(unit_ids)
            batch_results[batch_num] = res or []
            if res:
                logger.info(f"Batch {batch_num} completed: {len(res)} issues found")
            else:
                logger.info(f"Batch {batch_num} completed: no issues found")
        except Exception as exc:
            logger.debug("Stage 1 batch failed; cancelling sibling batches: %s", exc)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise RuntimeError(
                "Stage 1 review is incomplete because at least one batch failed"
            ) from exc
        finally:
            completed_batches += 1
            progress = 10 + int((completed_batches / len(batches)) * 50)
            emit_progress(
                event_callback,
                progress,
                f"Stage 1: Reviewed {completed_batches}/{len(batches)} batches",
            )

    review_unit_state.assert_complete()
    for batch_idx in range(1, len(batches) + 1):
        all_issues.extend(batch_results.get(batch_idx, []))

    elapsed = time.time() - started_at
    logger.info(
        f"Stage 1 Complete: {len(all_issues)} issues found across "
        f"{total_files} files in {elapsed:.2f}s"
    )
    return all_issues


async def _review_batch_with_timing(
    batch_idx: int,
    llm,
    request: ReviewRequestDto,
    batch: List[Dict[str, Any]],
    rag_client,
    prepared_context: Optional[Stage1PreparedContext],
    is_incremental: bool,
    pr_indexed: bool,
    fallback_llm=None,
    rag_state: Optional[Stage1RagState] = None,
    candidate_ledger: Optional[CandidateEvidenceLedger] = None,
    agent_service: Optional["AgentExecutionService"] = None,
    event_callback: Optional[Callable[[Dict], None]] = None,
) -> List[CodeReviewIssue]:
    start_time = time.time()
    batch_paths = [item["file"].path for item in batch]
    logger.info(f"[Batch {batch_idx}] STARTED - files: {batch_paths}")

    try:
        result = await review_file_batch(
            llm, request, batch, rag_client, prepared_context, is_incremental,
            pr_indexed=pr_indexed,
            fallback_llm=fallback_llm,
            rag_state=rag_state,
            candidate_ledger=candidate_ledger,
            agent_service=agent_service,
            event_callback=event_callback,
        )
        elapsed = time.time() - start_time
        logger.info(f"[Batch {batch_idx}] FINISHED in {elapsed:.2f}s - {len(result)} issues")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        logger.debug(f"[Batch {batch_idx}] FAILED after {elapsed:.2f}s: {e}")
        raise


def _stage1_batch_owns_rag_sequence(
    batch_items: List[Dict[str, Any]],
) -> bool:
    """Assign one exact-RAG sequence to a losslessly sharded local file.

    Joint local units already cover every source/diff/shared slice. Replaying
    every RAG bundle for every sibling would multiply two independent packing
    dimensions. The first unit owns the file's RAG sequence; all siblings still
    have to complete before their shared hunk ownership is satisfied.
    """
    if len(batch_items) != 1:
        return True
    item = batch_items[0]
    unit_total = item.get("_joint_unit_total")
    unit_index = item.get("_joint_unit_index")
    if not isinstance(unit_total, int) or unit_total <= 1:
        return True
    return isinstance(unit_index, int) and unit_index == 1


async def review_file_batch(
    llm,
    request: ReviewRequestDto,
    batch_items: List[Dict[str, Any]],
    rag_client,
    prepared_context: Optional[Stage1PreparedContext] = None,
    is_incremental: bool = False,
    pr_indexed: bool = False,
    fallback_llm=None,
    rag_state: Optional[Stage1RagState] = None,
    candidate_ledger: Optional[CandidateEvidenceLedger] = None,
    agent_service: Optional["AgentExecutionService"] = None,
    event_callback: Optional[Callable[[Dict], None]] = None,
) -> List[CodeReviewIssue]:
    if prepared_context is not None and not isinstance(prepared_context, Stage1PreparedContext):
        # Backwards compatibility for older direct callers/tests that pass
        # ProcessedDiff as the fifth positional argument.
        prepared_context = _build_stage_1_prepared_context(request, prepared_context, is_incremental)
    elif prepared_context is None:
        prepared_context = _build_stage_1_prepared_context(request, None, is_incremental)
    material = _prepare_stage1_prompt_material(
        request,
        batch_items,
        prepared_context,
        is_incremental,
    )
    batch_file_paths = material.batch_file_paths
    if material.enrichment_identifiers:
        logger.info(
            "Metadata identifiers for batch retrieval: %d",
            len(material.enrichment_identifiers),
        )

    rag_context_text = ""
    batch_rag_context = None
    batch_visible_evidence_by_id: Dict[
        str, tuple[Dict[str, Any], ...]
    ] = {}

    exact_context_bound = (
        _is_exact_revision_bound(request, pr_indexed)
        or _has_exact_base_binding(request)
    )
    owns_rag_sequence = _stage1_batch_owns_rag_sequence(batch_items)
    if owns_rag_sequence and (rag_client or exact_context_bound):
        batch_rag_context = await fetch_batch_rag_context(
            rag_client,
            request,
            batch_file_paths,
            pr_indexed,
            enrichment_identifiers=material.enrichment_identifiers,
            rag_state=rag_state,
        )
    elif not owns_rag_sequence:
        logger.info(
            "Stage 1 local joint unit uses the file's shared RAG sequence "
            "owned by unit 1/%s: paths=%s unit=%s",
            batch_items[0].get("_joint_unit_total"),
            batch_file_paths,
            batch_items[0].get("_joint_unit_index"),
        )

    if _rag_context_has_chunks(batch_rag_context):
        logger.info(f"Using per-batch RAG context for: {batch_file_paths}")
        rag_context_text = format_rag_context(
            batch_rag_context,
            set(batch_file_paths),
            pr_changed_files=request.changedFiles,
            deleted_files=request.deletedFiles,
            current_file_complete_paths=material.complete_current_file_paths,
            visible_evidence_by_id=batch_visible_evidence_by_id,
        )
    logger.info(f"RAG context for batch: {len(rag_context_text)} chars")
    if not material.file_metadata_text:
        logger.debug(f"No structured parser metadata for batch {batch_file_paths}")
    prompt, plugin_context_text = _render_stage1_prompt(
        material,
        rag_context_text,
        visible_evidence_by_id=batch_visible_evidence_by_id,
        use_mcp_tools=agent_service is not None,
    )
    token_budget = _stage1_batch_token_limit(request)
    estimated_tokens = _estimated_prompt_tokens(prompt)
    if estimated_tokens > token_budget and len(batch_items) > 1:
        # The pre-admission repacker already consumed the review-wide batch
        # budget. Recursing here used to bypass that ceiling after RAG was
        # attached. The RAG packer below now projects evidence into this
        # batch's pre-allocated concrete invocation quota.
        logger.warning(
            "Stage 1 final multi-file prompt exceeded the packing target after "
            "optional enrichment; retaining the admitted core batch and "
            "bounding RAG within its existing quota: paths=%s "
            "estimated_tokens=%d target_tokens=%d",
            batch_file_paths,
            estimated_tokens,
            token_budget,
        )

    invocations = _build_stage1_rag_invocations(
        material,
        batch_rag_context,
        token_budget,
        max_invocations=max(
            1,
            int(batch_items[0].get("_stage1_invocation_quota", 1) or 1),
        ),
        use_mcp_tools=agent_service is not None,
    )
    all_issues: List[CodeReviewIssue] = []
    for invocation_index, (
        invocation_prompt,
        invocation_rag_text,
        invocation_plugin_context,
        invocation_visible_evidence,
        omitted_rag_shards,
        omitted_rag_chunks,
    ) in enumerate(invocations, start=1):
        invocation_tokens = _estimated_prompt_tokens(invocation_prompt)
        logger.info(
            "Stage 1 prompt assembled: total=%d chars, estimated_tokens=%d, "
            "target_tokens=%d, metadata=%d, rag=%d, plugin=%d, files=%d, "
            "evidence_shard=%d/%d",
            len(invocation_prompt),
            invocation_tokens,
            token_budget,
            len(material.file_metadata_text),
            len(invocation_rag_text),
            len(invocation_plugin_context),
            len(batch_file_paths),
            invocation_index,
            len(invocations),
        )
        record_prompt_diagnostic({
            "stage": "stage_1",
            "batchPaths": sorted(batch_file_paths),
            "fileCount": len(batch_file_paths),
            "totalPromptChars": len(invocation_prompt),
            "currentSourceChars": sum(
                len(str(item.get("current_code") or ""))
                for item in material.batch_files_data
            ),
            "currentSourcePerFileBudget": material.current_source_per_file_budget,
            "diffChars": sum(
                len(str(item.get("diff") or ""))
                for item in material.batch_files_data
            ),
            "metadataChars": len(material.file_metadata_text),
            "ragChars": len(invocation_rag_text),
            "pluginChars": len(invocation_plugin_context),
            "projectRulesChars": len(material.project_rules),
            "taskContextChars": len(material.task_context),
            "previousIssuesChars": len(material.previous_issues_for_batch),
            "boundaryContextChars": len(material.boundary_context),
            "estimatedInputTokens": invocation_tokens,
            "inputPackingTargetTokens": token_budget,
            "evidenceShardIndex": invocation_index,
            "evidenceShardTotal": len(invocations),
            "omittedRagShards": omitted_rag_shards,
            "omittedRagChunks": omitted_rag_chunks,
            "omittedStage1Units": sum(
                int(item.get("_omitted_stage1_unit_count", 0) or 0)
                for item in batch_items
            ),
            "omittedStage1Hunks": sum(
                len(item.get("_omitted_hunk_ids", ()) or ())
                for item in batch_items
            ),
        })

        direct_invocation_prompt = _render_stage1_prompt(
            material,
            invocation_rag_text,
            visible_evidence_by_id=invocation_visible_evidence,
            use_mcp_tools=False,
        )[0]
        issues = await _invoke_stage_1_batch_llm(
            llm,
            invocation_prompt,
            batch_file_paths,
            label="structured primary",
            agent_service=agent_service,
            event_callback=event_callback,
            direct_fallback_prompt=direct_invocation_prompt,
        )
        retry_llm = (
            fallback_llm
            if fallback_llm is not None and fallback_llm is not llm
            else llm
        )
        if issues is None:
            logger.info(
                "Stage 1 structured response was unusable for %s evidence "
                "shard %d/%d; retrying once as a reasoning-free direct "
                "output request",
                batch_file_paths,
                invocation_index,
                len(invocations),
            )
            issues = await _invoke_stage_1_batch_llm(
                retry_llm,
                invocation_prompt,
                batch_file_paths,
                label="direct-output recovery",
                force_unstructured=True,
                event_callback=event_callback,
                direct_fallback_prompt=direct_invocation_prompt,
            )
        if issues is None:
            logger.debug(
                "Batch review parse failure for %s evidence shard %d/%d. "
                "The batch will fail so missing results cannot be published "
                "as a clean review.",
                batch_file_paths,
                invocation_index,
                len(invocations),
            )
            raise RuntimeError(
                "Stage 1 batch produced no valid result after all configured "
                "attempts: " + ", ".join(batch_file_paths)
            )
        _merge_stage1_visible_evidence(
            rag_state,
            invocation_visible_evidence,
        )
        _register_stage_1_candidates(
            issues,
            batch_items,
            candidate_ledger,
            invocation_prompt,
            invocation_visible_evidence,
        )
        all_issues.extend(issues)

    return all_issues


def _rag_context_has_chunks(rag_context: Optional[Dict[str, Any]]) -> bool:
    context = _unwrap_rag_context(rag_context)
    chunks = context.get("relevant_code") or context.get("chunks") or []
    return bool(chunks)


def _rag_context_chunks(
    rag_context: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    context = _unwrap_rag_context(rag_context)
    chunks = context.get("relevant_code") or context.get("chunks") or []
    return [chunk for chunk in chunks if isinstance(chunk, dict)]


def _rag_context_coverage_note(
    rag_context: Optional[Dict[str, Any]],
) -> str:
    """Describe non-exhaustive retrieval without turning it into evidence."""
    context = _unwrap_rag_context(rag_context)
    metadata = context.get("_metadata") if isinstance(context, dict) else None
    if not isinstance(metadata, dict):
        return ""

    retrieval_state = str(metadata.get("retrieval_state") or "").casefold()
    coverage_state = str(metadata.get("coverage_state") or "").casefold()
    retrieval_scope = str(metadata.get("retrieval_scope") or "").casefold()
    raw_reasons = metadata.get("partial_reasons")
    reasons = tuple(
        dict.fromkeys(
            str(reason).strip()
            for reason in raw_reasons or ()
            if str(reason).strip()
        )
    ) if isinstance(raw_reasons, (list, tuple, set)) else ()

    if retrieval_state == "partial" or coverage_state == "partial":
        reason_text = f" Reasons: {', '.join(reasons[:3])}." if reasons else ""
        return (
            "[Structural retrieval coverage: PARTIAL. Returned exact evidence "
            "is usable, but absence from it is not negative evidence."
            f"{reason_text}]"
        )
    if coverage_state == "bounded_complete" or retrieval_scope == "bounded":
        return (
            "[Structural retrieval coverage: BOUNDED COMPLETE. The requested "
            "exact evidence bound completed; absence outside the returned "
            "evidence is not negative evidence.]"
        )
    return ""


def _rag_chunk_repository_paths(chunk: Dict[str, Any]) -> tuple[str, ...]:
    """Return exact normalized source/provenance paths carried by one chunk."""
    metadata = chunk.get("metadata") or {}
    candidates: List[Any] = [
        metadata.get("path"),
        chunk.get("path"),
        chunk.get("file_path"),
    ]
    matched_on = chunk.get("_matched_on")
    if isinstance(matched_on, str):
        candidates.extend(part.strip() for part in matched_on.split(","))
    return tuple(dict.fromkeys(
        normalized
        for candidate in candidates
        for normalized in (normalize_repository_path(candidate),)
        if normalized
    ))


def _architecture_relation_paths(chunk: Dict[str, Any]) -> tuple[str, ...]:
    """Return exact paths governed by one deterministic relation packet."""
    metadata = chunk.get("metadata") or {}
    candidates: List[Any] = list(metadata.get("architecture_paths") or ())
    for fact in metadata.get("plugin_graph_facts") or ():
        if not isinstance(fact, dict):
            continue
        candidates.append(fact.get("path"))
        candidates.extend(fact.get("related_paths") or ())
    return tuple(dict.fromkeys(
        normalized
        for candidate in candidates
        for normalized in (normalize_repository_path(candidate),)
        if normalized
    ))


def _stage1_rag_evidence_bundles(
    rag_context: Optional[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Pair exact relation authority with source bodies by repository path.

    Deterministic RAG relation packets carry ``architecture_key`` and
    ``architecture_paths``. Hydrated source chunks intentionally remain neutral
    source records and usually carry no architecture key, so key-only grouping
    would orphan the implementation body from its authority. Exact path
    provenance is the contract shared by both payloads.
    """
    chunks = _rag_context_chunks(rag_context)
    relations = [
        chunk
        for chunk in chunks
        if chunk.get("_match_type") == "architecture_relation"
    ]
    sources = [
        chunk
        for chunk in chunks
        if chunk.get("_match_type") == "architecture_related"
    ]
    relation_paths = {
        id(relation): set(_architecture_relation_paths(relation))
        for relation in relations
    }
    used_relation_ids: set[int] = set()
    bundles: List[List[Dict[str, Any]]] = []
    for source in sources:
        source_paths = set(_rag_chunk_repository_paths(source))
        authority = [
            relation
            for relation in relations
            if source_paths.intersection(relation_paths[id(relation)])
        ]
        used_relation_ids.update(id(relation) for relation in authority)
        bundles.append(authority + [source])

    # A relation can still be useful when its implementation body is absent
    # from the exact response. Preserve it once rather than silently dropping it.
    bundles.extend(
        [relation]
        for relation in relations
        if id(relation) not in used_relation_ids
    )
    bundles.extend(
        [chunk]
        for chunk in chunks
        if chunk not in relations and chunk not in sources
    )
    return bundles


def _clone_rag_chunk_with_text(
    chunk: Dict[str, Any],
    text: str,
    shard_index: int,
    shard_total: int,
) -> Dict[str, Any]:
    clone = dict(chunk)
    clone["text"] = text
    clone["content"] = text
    metadata = dict(clone.get("metadata") or {})
    metadata["source_shard_index"] = shard_index
    metadata["source_shard_total"] = shard_total
    clone["metadata"] = metadata
    return clone


def _format_stage1_rag_chunks(
    material: Stage1PromptMaterial,
    chunks: Sequence[Dict[str, Any]],
    coverage_note: str = "",
) -> tuple[str, Dict[str, tuple[Dict[str, Any], ...]]]:
    visible: Dict[str, tuple[Dict[str, Any], ...]] = {}
    text = format_rag_context(
        {"relevant_code": list(chunks)},
        set(material.batch_file_paths),
        pr_changed_files=getattr(material.request, "changedFiles", None),
        deleted_files=getattr(material.request, "deletedFiles", None),
        current_file_complete_paths=material.complete_current_file_paths,
        visible_evidence_by_id=visible,
    )
    if text and coverage_note:
        text = f"{coverage_note}\n\n{text}"
    return text, visible


def _expand_oversized_rag_bundle(
    material: Stage1PromptMaterial,
    bundle: List[Dict[str, Any]],
    token_budget: int,
    coverage_note: str = "",
    use_mcp_tools: Optional[bool] = None,
) -> List[List[Dict[str, Any]]]:
    """Losslessly split a source body while repeating its relation authority."""
    rag_text, visible = _format_stage1_rag_chunks(
        material,
        bundle,
        coverage_note,
    )
    prompt, _ = _render_stage1_prompt(
        material,
        rag_text,
        visible_evidence_by_id=visible,
        use_mcp_tools=use_mcp_tools,
    )
    if _estimated_prompt_tokens(prompt) <= token_budget:
        return [bundle]

    source_candidates = [
        chunk
        for chunk in bundle
        if chunk.get("_match_type") in {
            "architecture_related",
            "changed_file",
            "definition",
            "transitive_parent",
        }
    ]
    if len(source_candidates) != 1:
        return [bundle]
    source_chunk = source_candidates[0]
    source_text = str(
        source_chunk.get("text", source_chunk.get("content", ""))
    )
    relation_chunks = [chunk for chunk in bundle if chunk is not source_chunk]
    relation_text, relation_visible = _format_stage1_rag_chunks(
        material,
        relation_chunks,
        coverage_note,
    )
    relation_prompt, _ = _render_stage1_prompt(
        material,
        relation_text,
        visible_evidence_by_id=relation_visible,
        use_mcp_tools=use_mcp_tools,
    )
    available_tokens = max(
        1,
        token_budget
        - _estimated_prompt_tokens(relation_prompt)
        - 256,
    )
    segments = _split_source_at_semantic_line_boundaries(
        source_text,
        _text_character_budget(source_text, available_tokens),
    )
    if len(segments) <= 1:
        return [bundle]
    return [
        relation_chunks
        + [_clone_rag_chunk_with_text(
            source_chunk,
            segment,
            shard_index,
            len(segments),
        )]
        for shard_index, segment in enumerate(segments, start=1)
    ]


def _build_stage1_rag_invocations(
    material: Stage1PromptMaterial,
    rag_context: Optional[Dict[str, Any]],
    token_budget: int,
    max_invocations: int = 3,
    use_mcp_tools: Optional[bool] = None,
) -> List[
    tuple[
        str,
        str,
        str,
        Dict[str, tuple[Dict[str, Any], ...]],
        int,
        int,
    ]
]:
    """Pack prioritized RAG bundles under one concrete invocation quota."""
    max_invocations = max(1, max_invocations)
    coverage_note = _rag_context_coverage_note(rag_context)
    bundles: List[List[Dict[str, Any]]] = []
    for bundle in _stage1_rag_evidence_bundles(rag_context):
        bundles.extend(_expand_oversized_rag_bundle(
            material,
            bundle,
            token_budget,
            coverage_note,
            use_mcp_tools,
        ))

    if not bundles:
        prompt, plugin_context = _render_stage1_prompt(
            material,
            "",
            use_mcp_tools=use_mcp_tools,
        )
        return [(prompt, "", plugin_context, {}, 0, 0)]

    packed: List[List[Dict[str, Any]]] = []
    packing_budget = max(1, token_budget - 128)
    current: List[Dict[str, Any]] = []
    for bundle in bundles:
        candidate = current + bundle
        rag_text, visible = _format_stage1_rag_chunks(
            material,
            candidate,
            coverage_note,
        )
        prompt, _ = _render_stage1_prompt(
            material,
            rag_text,
            visible_evidence_by_id=visible,
            use_mcp_tools=use_mcp_tools,
        )
        if current and _estimated_prompt_tokens(prompt) > packing_budget:
            packed.append(current)
            current = list(bundle)
        else:
            current = candidate
    if current:
        packed.append(current)

    omitted_packed = packed[max_invocations:]
    selected_packed = packed[:max_invocations]
    omitted_rag_shards = len(omitted_packed)
    omitted_rag_chunks = sum(len(chunks) for chunks in omitted_packed)
    if omitted_rag_shards:
        logger.warning(
            "Stage 1 RAG invocation ceiling reached: paths=%s admitted=%d "
            "omitted_shards=%d omitted_chunks=%d",
            material.batch_file_paths,
            len(selected_packed),
            omitted_rag_shards,
            omitted_rag_chunks,
        )

    invocations = []
    for shard_index, chunks in enumerate(selected_packed, start=1):
        rag_text, visible = _format_stage1_rag_chunks(
            material,
            chunks,
            coverage_note,
        )
        if len(selected_packed) > 1:
            rag_text = (
                f"[Bounded exact-evidence shard {shard_index}/"
                f"{len(selected_packed)}.]\n\n"
                + rag_text
            )
        if omitted_rag_shards and shard_index == len(selected_packed):
            rag_text += (
                "\n\n[CodeCrow RAG invocation ceiling reached: "
                f"omitted_shards={omitted_rag_shards}, "
                f"omitted_chunks={omitted_rag_chunks}. Absence from the "
                "admitted evidence is not negative evidence.]"
            )
        prompt, plugin_context = _render_stage1_prompt(
            material,
            rag_text,
            visible_evidence_by_id=visible,
            use_mcp_tools=use_mcp_tools,
        )
        if _estimated_prompt_tokens(prompt) > token_budget:
            logger.warning(
                "Stage 1 retained one indivisible semantic evidence bundle "
                "above the packing target: paths=%s estimated_tokens=%d "
                "target_tokens=%d",
                material.batch_file_paths,
                _estimated_prompt_tokens(prompt),
                token_budget,
            )
        invocations.append((
            prompt,
            rag_text,
            plugin_context,
            visible,
            omitted_rag_shards,
            omitted_rag_chunks,
        ))
    return invocations


def _merge_stage1_visible_evidence(
    rag_state: Optional[Stage1RagState],
    visible_evidence_by_id: Dict[str, tuple[Dict[str, Any], ...]],
) -> None:
    if not rag_state:
        return
    for evidence_id in sorted(visible_evidence_by_id):
        combined = list(rag_state.exact_evidence_by_id.get(evidence_id, ()))
        for fact in visible_evidence_by_id[evidence_id]:
            if fact not in combined:
                combined.append(fact)
        rag_state.exact_evidence_by_id[evidence_id] = tuple(sorted(
            combined,
            key=lambda fact: json.dumps(
                fact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        ))


async def _invoke_stage_1_batch_llm(
    llm,
    prompt: str,
    batch_file_paths: List[str],
    label: str = "primary",
    force_unstructured: bool = False,
    agent_service: Optional["AgentExecutionService"] = None,
    event_callback: Optional[Callable[[Dict], None]] = None,
    direct_fallback_prompt: Optional[str] = None,
) -> Optional[List[CodeReviewIssue]]:
    if agent_service is not None:
        from service.agent import AgentExecutionRequest

        try:
            execution = await agent_service.execute(AgentExecutionRequest(
                prompt=prompt,
                allowed_tool_names=STAGE1_AGENT_TOOL_NAMES,
                max_steps=STAGE1_AGENT_MAX_STEPS,
                # The Stage 1 prompt already requires schema-shaped JSON. Keep
                # the tool-aware final answer intact and validate it here;
                # mcp-use's optional post-formatting call does not retain the
                # batch prompt and can discard exact file coverage.
                output_schema=None,
                additional_instructions=(
                    "Use repository tools only to fill concrete gaps in the "
                    "supplied diff, source, and RAG evidence. Then return the "
                    "complete structured file-review response required by the "
                    "prompt for every requested file."
                ),
                metadata={
                    "stage": "stage_1",
                    "label": label,
                    "batchPaths": tuple(batch_file_paths),
                },
            ))
            output = execution.output
            if isinstance(output, FileReviewBatchOutput):
                data = output
            elif isinstance(output, dict):
                data = FileReviewBatchOutput.model_validate(output)
            else:
                content = extract_llm_response_text(output)
                if not content.strip():
                    raise ValueError("Stage 1 agent returned no review content")
                data = await parse_llm_response(
                    content,
                    FileReviewBatchOutput,
                    llm,
                    max_provider_repairs=0,
                )
            _validate_batch_review_coverage(data, batch_file_paths)
            return _extract_calibrated_issues(data)
        except Exception as agent_error:
            # Agentic repository exploration is optional enrichment. Continue
            # with the same assembled evidence when its final review is absent,
            # malformed, incomplete, or the tool transport itself failed.
            logger.warning(
                "Stage 1 agent did not produce a complete review for batch %s "
                "(%s); continuing with the assembled prompt without tools: %s",
                batch_file_paths,
                label,
                agent_error,
            )
            emit_status(
                event_callback,
                "stage_1_agent_degraded",
                "Agentic repository analysis did not produce a complete result "
                "for one file-review batch; analysis continued with its diff "
                "and prepared context",
            )

    direct_prompt = direct_fallback_prompt or prompt

    if _supports_structured_output(llm) and not force_unstructured:
        try:
            invocation = await invoke_structured_output(
                llm,
                direct_prompt,
                FileReviewBatchOutput,
                effort=ReasoningEffort.LOW,
                label=f"stage-1-{label}",
            )
            result = await resolve_structured_output(
                invocation,
                FileReviewBatchOutput,
                llm,
            )
            if result:
                _validate_batch_review_coverage(result, batch_file_paths)
                return _extract_calibrated_issues(result)
            logger.debug(
                "Structured output returned empty Stage 1 result for %s (%s)",
                batch_file_paths,
                label,
            )
        except Exception as e:
            # The batch owner emits the single exhausted-attempt warning. Keep
            # attempt detail at DEBUG to avoid duplicating one failure for every
            # nested batch call; raw-response shape failures are already logged
            # by the structured-output adapter.
            logger.debug(
                "Structured output failed for Stage 1 batch %s (%s): "
                "error_type=%s",
                batch_file_paths,
                label,
                type(e).__name__,
            )
        # A direct-output parse is the one configured recovery call, not an
        # implicit second primary call. The caller owns that recovery so total
        # provider attempts remain primary + one finite recovery attempt.
        return None
    else:
        logger.info(
            "Structured output skipped for Stage 1 batch %s (%s); using prompt JSON parsing",
            batch_file_paths,
            label,
        )

    try:
        response = await llm.ainvoke(
            direct_prompt,
            **reasoning_request_kwargs(
                llm,
                ReasoningEffort.NONE
                if force_unstructured
                else ReasoningEffort.LOW,
            ),
        )
        content = extract_llm_response_text(response)
        if not content.strip():
            logger.warning(
                "Stage 1 raw fallback returned no content for %s (%s): %s",
                batch_file_paths,
                label,
                format_response_diagnostics(response),
            )
        data = await parse_llm_response(
            content,
            FileReviewBatchOutput,
            llm,
            max_provider_repairs=0,
        )
        _validate_batch_review_coverage(data, batch_file_paths)
        return _extract_calibrated_issues(data)
    except Exception as parse_err:
        logger.debug(
            "Stage 1 batch parse failed for %s (%s): %s",
            batch_file_paths,
            label,
            parse_err,
        )
        return None


def _validate_batch_review_coverage(
    batch_output: FileReviewBatchOutput,
    batch_file_paths: List[str],
) -> None:
    """Require one unambiguous review result for every requested batch file."""
    expected = [normalize_repository_path(path) for path in batch_file_paths]
    observed = [
        normalize_repository_path(review.file)
        for review in batch_output.reviews
    ]
    expected_counts = Counter(expected)
    observed_counts = Counter(observed)
    missing = sorted(
        path
        for path in expected_counts
        if not observed_counts[path]
    )
    unexpected = sorted(
        path for path in observed_counts if path not in expected_counts
    )
    duplicates = sorted(
        path for path, count in observed_counts.items() if path and count > 1
    )
    empty_count = observed_counts.get("", 0)

    if (
        not expected
        or "" in expected_counts
        or len(observed) != len(expected)
        or missing
        or unexpected
        or duplicates
        or empty_count
    ):
        details = [f"expected={len(expected)}", f"received={len(observed)}"]
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        if duplicates:
            details.append("duplicates=" + ",".join(duplicates))
        if empty_count:
            details.append(f"empty_paths={empty_count}")
        raise ValueError(
            "Stage 1 batch review coverage mismatch (" + "; ".join(details) + ")"
        )


def _extract_calibrated_issues(batch_output: FileReviewBatchOutput) -> List[CodeReviewIssue]:
    all_batch_issues: List[CodeReviewIssue] = []
    for review in batch_output.reviews:
        review_confidence = (review.confidence or "MEDIUM").upper()
        for issue in review.issues:
            if review_confidence == "LOW" and issue.severity.upper() == "HIGH":
                logger.info(
                    f"Downgrading issue in {review.file} from HIGH to MEDIUM "
                    f"(batch confidence: LOW): {issue.reason[:80]}"
                )
                issue.severity = "MEDIUM"
        all_batch_issues.extend(review.issues)
    return all_batch_issues


def _candidate_owner_item(
    issue: CodeReviewIssue,
    batch_items: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    issue_path = normalize_repository_path(getattr(issue, "file", "") or "")
    if not issue_path:
        return None
    exact = [
        item
        for item in batch_items
        if normalize_repository_path(getattr(item.get("file"), "path", ""))
        == issue_path
    ]
    if len(exact) == 1:
        return exact[0]
    matches = [
        item
        for item in batch_items
        if repository_paths_match(
            issue_path,
            getattr(item.get("file"), "path", ""),
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _register_stage_1_candidates(
    issues: List[CodeReviewIssue],
    batch_items: List[Dict[str, Any]],
    candidate_ledger: Optional[CandidateEvidenceLedger],
    generation_prompt: str,
    visible_evidence_by_id: Optional[
        Dict[str, tuple[Dict[str, Any], ...]]
    ] = None,
) -> None:
    if candidate_ledger is None:
        return
    batch_identity = ",".join(sorted(
        str(item.get("_review_unit_id") or "")
        for item in batch_items
    ))
    for index, issue in enumerate(issues):
        owner = _candidate_owner_item(issue, batch_items)
        candidate_ledger.register(
            issue,
            stage="stage_1",
            source_key=f"{batch_identity}:{index}",
            review_unit_ids=(
                (str(owner.get("_review_unit_id")),)
                if owner is not None and owner.get("_review_unit_id")
                else ()
            ),
            prompt_hunk_ids=(
                tuple(owner.get("_hunk_ids", ()) or ())
                if owner is not None
                else ()
            ),
            generation_prompt=generation_prompt,
            visible_evidence_by_id=visible_evidence_by_id,
        )
