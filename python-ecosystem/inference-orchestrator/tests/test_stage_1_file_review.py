"""
Tests for pure helper functions in stage_1_file_review.py.

Covers deterministic Stage 1 batching, context, scheduling, and issue extraction.
"""
import pytest
import asyncio
import json
import logging
import re
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

from service.review.orchestrator.stage_1_file_review import (
    chunk_files,
    Stage1PreparedContext,
    _build_stage_1_prepared_context,
    _bounded_current_file_context,
    _diff_contains_complete_added_source,
    _find_diff_file_for_path,
    _split_hunk_by_lines,
    _chunk_diff_preserving_hunks,
    _expand_oversized_diff_batches,
    _expand_oversized_current_source_batches,
    _expand_oversized_stage1_evidence_batches,
    _repack_stage1_batches_by_rendered_input,
    _prepare_stage1_prompt_material,
    _render_stage1_prompt,
    _estimated_prompt_tokens,
    _build_stage1_rag_invocations,
    _stage1_rag_evidence_bundles,
    _format_batch_metadata_json,
    _iter_batch_enrichment_metadata,
    _extract_metadata_identifiers,
    _flatten_deterministic_context,
    _rag_context_has_chunks,
    Stage1RagState,
    Stage1ReviewUnitState,
    fetch_batch_rag_context,
    execute_stage_1_file_reviews,
    review_file_batch,
    _supports_structured_output,
    _deduplicate_pr_stale_chunks,
    _extract_calibrated_issues,
    _invoke_stage_1_batch_llm,
    _validate_batch_review_coverage,
    create_smart_batches_wrapper,
    DETERMINISTIC_RAG_MAX_CHUNKS,
    STAGE1_CURRENT_SOURCE_BATCH_CHAR_BUDGET,
    STAGE1_METADATA_CHAR_BUDGET,
    STAGE1_AGENT_TOOL_NAMES,
)
from service.review.orchestrator.context_helpers import format_rag_context
from service.review.candidate_ledger import CandidateEvidenceLedger
from model.multi_stage import (
    FileGroup,
    ReviewFile,
    FileReviewBatchOutput,
    FileReviewOutput,
    ReviewPlan,
)
from model.output_schemas import CodeReviewIssue
from utils.diff_processor import DiffChangeType, DiffFile, DiffProcessor, ProcessedDiff


# ── chunk_files ──────────────────────────────────────────────────

@pytest.mark.asyncio(loop_scope="function")
async def test_batch_llm_attempt_details_do_not_duplicate_owner_warning(caplog):
    class FailingLlm:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _prompt):
            raise RuntimeError("provider unavailable")

    with caplog.at_level(logging.DEBUG):
        result = await _invoke_stage_1_batch_llm(
            FailingLlm(), "prompt", ["src/a.py"]
        )

    assert result is None
    assert not [
        record for record in caplog.records
        if record.levelno >= logging.WARNING
    ]


def _clean_file_review(path):
    return FileReviewOutput(
        file=path,
        analysis_summary="No defect found",
        issues=[],
        confidence="HIGH",
    )


def _packing_request(paths, enrichment):
    request = MagicMock(
        deltaDiff=None,
        rawDiff="",
        taskContext=None,
        enrichmentData=enrichment,
        projectRules=[],
        previousCodeAnalysisIssues=[],
        changedFiles=paths,
        deletedFiles=[],
        currentCommitHash="a" * 40,
        commitHash="a" * 40,
        maxAllowedTokens=200000,
    )
    request.get_rag_branch.return_value = "feature"
    request.get_rag_base_branch.return_value = "main"
    return request


class TestBatchReviewCoverage:
    def test_exact_clean_coverage_is_valid(self):
        output = FileReviewBatchOutput(
            reviews=[_clean_file_review("src/a.py"), _clean_file_review("src/b.py")]
        )

        _validate_batch_review_coverage(output, ["src/a.py", "src/b.py"])

    @pytest.mark.parametrize(
        ("review_paths", "match"),
        [
            ([], "missing=src/a.py,src/b.py"),
            (["src/a.py"], "missing=src/b.py"),
            (["src/a.py", "src/a.py"], "duplicates=src/a.py"),
            (["src/a.py", "src/c.py"], "unexpected=src/c.py"),
            (["src/a.py", ""], "empty_paths=1"),
        ],
    )
    def test_incomplete_or_ambiguous_coverage_is_rejected(
        self,
        review_paths,
        match,
    ):
        output = FileReviewBatchOutput(
            reviews=[_clean_file_review(path) for path in review_paths]
        )

        with pytest.raises(ValueError, match=match):
            _validate_batch_review_coverage(
                output,
                ["src/a.py", "src/b.py"],
            )

    @pytest.mark.asyncio(loop_scope="function")
    async def test_invalid_structured_coverage_leaves_retry_to_outer_loop(self):
        class StructuredAttempt:
            async def ainvoke(self, _prompt):
                return FileReviewBatchOutput(reviews=[])

        class Llm:
            def __init__(self):
                self.raw_calls = 0
                self.structured_calls = 0

            def with_structured_output(self, _schema):
                self.structured_calls += 1
                return StructuredAttempt()

            async def ainvoke(self, _prompt):
                self.raw_calls += 1
                return (
                    '{"reviews":[{"file":"src/a.py",'
                    '"analysis_summary":"No defect found","issues":[],'
                    '"confidence":"HIGH","note":""}]}'
                )

        llm = Llm()

        result = await _invoke_stage_1_batch_llm(
            llm,
            "complete review prompt",
            ["src/a.py"],
        )

        assert result is None
        assert llm.structured_calls == 1
        assert llm.raw_calls == 0

        retry_result = await _invoke_stage_1_batch_llm(
            llm,
            "complete review prompt",
            ["src/a.py"],
            label="retry",
            force_unstructured=True,
        )

        assert retry_result == []
        assert llm.structured_calls == 1
        assert llm.raw_calls == 1

    @pytest.mark.asyncio(loop_scope="function")
    async def test_agent_executes_complete_batch_prompt_with_read_tools(self):
        class AgentService:
            def __init__(self):
                self.requests = []

            async def execute(self, request):
                self.requests.append(request)
                return SimpleNamespace(output=FileReviewBatchOutput(
                    reviews=[_clean_file_review("src/a.py")]
                ))

        class DirectLlmMustNotRun:
            def with_structured_output(self, _schema):
                raise AssertionError("direct model path must not run")

        agent_service = AgentService()
        result = await _invoke_stage_1_batch_llm(
            DirectLlmMustNotRun(),
            "diff plus prepared RAG evidence",
            ["src/a.py"],
            agent_service=agent_service,
        )

        assert result == []
        assert len(agent_service.requests) == 1
        request = agent_service.requests[0]
        assert request.prompt == "diff plus prepared RAG evidence"
        assert request.allowed_tool_names == STAGE1_AGENT_TOOL_NAMES
        assert request.output_schema is None
        assert request.metadata["batchPaths"] == ("src/a.py",)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_agent_fenced_json_is_parsed_without_post_formatting_call(self):
        class AgentService:
            def __init__(self):
                self.requests = []

            async def execute(self, request):
                self.requests.append(request)
                return SimpleNamespace(output=(
                    "```json\n"
                    '{"reviews":[{"file":"src/a.py",'
                    '"analysis_summary":"No defect found","issues":[],'
                    '"confidence":"HIGH","note":""}]}\n'
                    "```"
                ))

        class DirectLlmMustNotRun:
            def with_structured_output(self, _schema):
                raise AssertionError("direct model path must not run")

        agent_service = AgentService()
        result = await _invoke_stage_1_batch_llm(
            DirectLlmMustNotRun(),
            "diff plus prepared RAG evidence",
            ["src/a.py"],
            agent_service=agent_service,
        )

        assert result == []
        assert len(agent_service.requests) == 1
        assert agent_service.requests[0].output_schema is None

    @pytest.mark.asyncio(loop_scope="function")
    async def test_agent_transport_failure_falls_back_to_prepared_prompt(self):
        class AgentService:
            async def execute(self, _request):
                raise RuntimeError("MCP subprocess unavailable")

        class StructuredAttempt:
            async def ainvoke(self, prompt):
                assert prompt == "prepared prompt without tool instructions"
                return FileReviewBatchOutput(
                    reviews=[_clean_file_review("src/a.py")]
                )

        class Llm:
            def with_structured_output(self, _schema):
                return StructuredAttempt()

        events = []
        result = await _invoke_stage_1_batch_llm(
            Llm(),
            "prepared prompt remains intact",
            ["src/a.py"],
            agent_service=AgentService(),
            event_callback=events.append,
            direct_fallback_prompt=(
                "prepared prompt without tool instructions"
            ),
        )

        assert result == []
        assert events[-1]["state"] == "stage_1_agent_degraded"
        assert "did not produce a complete result" in events[-1]["message"]
        assert "tools were unavailable" not in events[-1]["message"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_empty_agent_output_uses_mcp_free_direct_fallback(self):
        class AgentService:
            async def execute(self, _request):
                return SimpleNamespace(output="")

        class StructuredAttempt:
            async def ainvoke(self, prompt):
                assert prompt == "direct evidence-only prompt"
                return FileReviewBatchOutput(
                    reviews=[_clean_file_review("src/a.py")]
                )

        class Llm:
            def with_structured_output(self, _schema):
                return StructuredAttempt()

        result = await _invoke_stage_1_batch_llm(
            Llm(),
            "agent prompt with repository tools",
            ["src/a.py"],
            agent_service=AgentService(),
            direct_fallback_prompt="direct evidence-only prompt",
        )

        assert result == []

    @pytest.mark.asyncio(loop_scope="function")
    async def test_all_coverage_invalid_attempts_do_not_publish_clean_result(self):
        class Llm:
            def with_structured_output(self, _schema):
                return self

            async def ainvoke(self, _prompt):
                return '{"reviews":[]}'

        result = await _invoke_stage_1_batch_llm(
            Llm(),
            "complete review prompt",
            ["src/a.py"],
        )

        assert result is None

    @staticmethod
    def _length_exhaustion_review_case(recovery_content):
        class ChatOpenRouter:
            model_name = "openai/reasoning-model"
            extra_body = {}

            def __init__(self):
                self.calls = []

            def with_structured_output(self, _schema, **_kwargs):
                return self

            async def ainvoke(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                if len(self.calls) == 1:
                    return {
                        "raw": SimpleNamespace(
                            content="",
                            tool_calls=[],
                            response_metadata={
                                "finish_reason": "max_tokens",
                                "token_usage": {
                                    "completion_tokens": 18_000,
                                    "completion_tokens_details": {
                                        "reasoning_tokens": 18_000,
                                    },
                                },
                            },
                        ),
                        "parsed": None,
                        "parsing_error": None,
                    }
                return SimpleNamespace(
                    content=recovery_content,
                    response_metadata={"finish_reason": "stop"},
                )

        path = "src/a.py"
        request = MagicMock(
            deltaDiff=None,
            rawDiff="",
            taskContext=None,
            taskHistoryContext=None,
            enrichmentData=None,
            projectRules=[],
            projectCapabilities=None,
            previousCodeAnalysisIssues=[],
            changedFiles=[path],
            deletedFiles=[],
            currentCommitHash=None,
            commitHash=None,
            baseCommitHash=None,
            pullRequestId=None,
            ragCollectionTarget=None,
            ragBaseGenerationManifestSha256=None,
            maxAllowedTokens=200_000,
        )
        prepared = _build_stage_1_prepared_context(
            request,
            None,
            is_incremental=False,
        )
        batch = [{
            "file": ReviewFile(
                path=path,
                focus_areas=["general"],
                risk_level="LOW",
            ),
            "priority": "LOW",
        }]
        return ChatOpenRouter(), request, prepared, batch

    @pytest.mark.asyncio(loop_scope="function")
    async def test_empty_length_primary_recovers_with_reasoning_disabled(self):
        recovery = json.dumps({
            "reviews": [{
                "file": "src/a.py",
                "analysis_summary": "No defect found",
                "issues": [],
                "confidence": "HIGH",
                "note": "",
            }],
        })
        llm, request, prepared, batch = self._length_exhaustion_review_case(
            recovery
        )

        result = await review_file_batch(
            llm,
            request,
            batch,
            rag_client=None,
            prepared_context=prepared,
        )

        assert result == []
        assert len(llm.calls) == 2
        assert llm.calls[0][1]["extra_body"]["reasoning"] == {
            "effort": "low"
        }
        assert llm.calls[1][1]["extra_body"]["reasoning"] == {
            "effort": "none"
        }

    @pytest.mark.asyncio(loop_scope="function")
    async def test_direct_recovery_does_not_rerun_degraded_agent(self):
        recovery = json.dumps({
            "reviews": [{
                "file": "src/a.py",
                "analysis_summary": "No defect found",
                "issues": [],
                "confidence": "HIGH",
                "note": "",
            }],
        })
        llm, request, prepared, batch = self._length_exhaustion_review_case(
            recovery
        )

        class AgentService:
            def __init__(self):
                self.calls = 0

            async def execute(self, _request):
                self.calls += 1
                return SimpleNamespace(output="")

        agent_service = AgentService()
        events = []

        result = await review_file_batch(
            llm,
            request,
            batch,
            rag_client=None,
            prepared_context=prepared,
            agent_service=agent_service,
            event_callback=events.append,
        )

        assert result == []
        assert agent_service.calls == 1
        assert [
            event["state"]
            for event in events
            if event.get("state") == "stage_1_agent_degraded"
        ] == ["stage_1_agent_degraded"]
        assert len(llm.calls) == 2
        assert llm.calls[1][1]["extra_body"]["reasoning"] == {
            "effort": "none"
        }

    @pytest.mark.asyncio(loop_scope="function")
    async def test_length_recovery_does_not_accept_truncated_json(self):
        llm, request, prepared, batch = self._length_exhaustion_review_case(
            '{"reviews":[{"file":"src/a.py"'
        )

        with pytest.raises(RuntimeError, match="produced no valid result"):
            await review_file_batch(
                llm,
                request,
                batch,
                rag_client=None,
                prepared_context=prepared,
            )

        assert len(llm.calls) == 2
        assert llm.calls[1][1]["extra_body"]["reasoning"] == {
            "effort": "none"
        }


class TestChunkFiles:
    def _make_groups(self, paths_per_group):
        groups = []
        for gid, paths in enumerate(paths_per_group):
            files = [ReviewFile(path=p, focus_areas=[], risk_level="MEDIUM") for p in paths]
            groups.append(
                FileGroup(group_id=f"g{gid}", priority="MEDIUM", rationale="test", files=files)
            )
        return groups

    def test_single_small_group(self):
        groups = self._make_groups([["a.py", "b.py"]])
        batches = chunk_files(groups, max_files_per_batch=5)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_group_exceeds_batch_size(self):
        groups = self._make_groups([["a.py", "b.py", "c.py", "d.py", "e.py", "f.py"]])
        batches = chunk_files(groups, max_files_per_batch=3)
        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3

    def test_multiple_groups_fit(self):
        groups = self._make_groups([["a.py"], ["b.py"]])
        batches = chunk_files(groups, max_files_per_batch=5)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_empty_groups(self):
        batches = chunk_files([], max_files_per_batch=5)
        assert batches == []

    def test_groups_split_across_batches(self):
        groups = self._make_groups([["a.py", "b.py", "c.py"], ["d.py", "e.py", "f.py"]])
        batches = chunk_files(groups, max_files_per_batch=3)
        assert len(batches) == 2

    def test_batch_size_one(self):
        groups = self._make_groups([["a.py", "b.py"]])
        batches = chunk_files(groups, max_files_per_batch=1)
        assert len(batches) == 2
        assert len(batches[0]) == 1
        assert len(batches[1]) == 1


# ── Stage 1 prepared context ────────────────────────────────────

class TestStage1PreparedContext:
    def test_diff_lookup_uses_suffix_index(self):
        request = MagicMock(deltaDiff=None, taskContext=None, enrichmentData=None)
        processed = ProcessedDiff(files=[
            DiffFile(
                path="repo/services/api/src/Foo.py",
                change_type=DiffChangeType.MODIFIED,
                content="diff --git a/repo/services/api/src/Foo.py b/repo/services/api/src/Foo.py",
            )
        ])

        context = _build_stage_1_prepared_context(request, processed, is_incremental=False)

        assert _find_diff_file_for_path(context, "services/api/src/Foo.py").path == "repo/services/api/src/Foo.py"
        assert _find_diff_file_for_path(context, "src/Foo.py").path == "repo/services/api/src/Foo.py"

    def test_current_file_content_is_indexed_for_direct_stage_1_evidence(self):
        file_content = MagicMock(
            path="repo/templates/ratings.phtml",
            content="use SwatchHelper;\n$this->helper(SwatchHelper::class);",
            skipped=False,
        )
        enrichment = MagicMock(fileContents=[file_content], fileMetadata=[])
        request = MagicMock(
            deltaDiff=None,
            taskContext=None,
            enrichmentData=enrichment,
        )

        context = _build_stage_1_prepared_context(request, None, is_incremental=False)

        assert context.file_content_by_path["templates/ratings.phtml"] == file_content.content

    def test_large_current_source_is_never_reduced_to_hunk_windows(self):
        source = "\n".join(
            f"line_{line_number}" for line_number in range(1, 401)
        )
        diff = """\
diff --git a/src/large.py b/src/large.py
--- a/src/large.py
+++ b/src/large.py
@@ -198,3 +198,3 @@
-old
+new
"""

        rendered = _bounded_current_file_context(
            source,
            diff,
            context_lines=3,
        )

        assert rendered == source
        assert "line_1\n" in rendered
        assert "line_400" in rendered

    def test_large_current_source_is_complete_when_diff_has_no_hunk(self):
        source = "start\n" + ("middle\n" * 100) + "end\n"

        rendered = _bounded_current_file_context(source, "metadata only")

        assert rendered == source

    def test_complete_added_source_requires_contiguous_lossless_diff(self):
        source = "first()\nsecond()\n"
        complete_diff = """\
diff --git a/src/new.py b/src/new.py
new file mode 100644
--- /dev/null
+++ b/src/new.py
@@ -0,0 +1,2 @@
+first()
+second()
"""
        partial_diff = """\
diff --git a/src/new.py b/src/new.py
new file mode 100644
--- /dev/null
+++ b/src/new.py
@@ -0,0 +2,1 @@
+second()
"""
        modified_diff = """\
diff --git a/src/new.py b/src/new.py
--- a/src/new.py
+++ b/src/new.py
@@ -1 +1 @@
-first()
+second()
"""

        assert _diff_contains_complete_added_source(source, complete_diff)
        assert not _diff_contains_complete_added_source(source, partial_diff)
        assert not _diff_contains_complete_added_source(source, modified_diff)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_prompt_receives_current_file_content_without_rag(self):
        path = "templates/ratings.phtml"
        source = "use SwatchHelper;\n$this->helper(SwatchHelper::class);"
        file_content = MagicMock(path=path, content=source, skipped=False)
        enrichment = MagicMock(fileContents=[file_content], fileMetadata=[])
        request = MagicMock(
            deltaDiff=None,
            rawDiff="",
            taskContext=None,
            enrichmentData=enrichment,
            projectRules=[],
            previousCodeAnalysisIssues=[],
            changedFiles=[path],
            deletedFiles=[],
            currentCommitHash="a" * 40,
        )
        prepared = _build_stage_1_prepared_context(request, None, is_incremental=False)
        batch = [{
            "file": ReviewFile(path=path, focus_areas=["general"], risk_level="LOW"),
            "priority": "LOW",
        }]

        with patch(
            "service.review.orchestrator.stage_1_file_review._invoke_stage_1_batch_llm",
            new_callable=AsyncMock,
            return_value=[],
        ) as invoke:
            result = await review_file_batch(
                MagicMock(),
                request,
                batch,
                rag_client=None,
                prepared_context=prepared,
            )

        assert result == []
        prompt = invoke.await_args.args[1]
        assert "Current File Content (post-change" in prompt
        assert source in prompt

    @pytest.mark.asyncio(loop_scope="function")
    async def test_added_file_source_is_not_duplicated_when_diff_is_complete(self):
        path = "src/new.py"
        source = "first()\nsecond()\n"
        raw_diff = """\
diff --git a/src/new.py b/src/new.py
new file mode 100644
--- /dev/null
+++ b/src/new.py
@@ -0,0 +1,2 @@
+first()
+second()
"""
        file_content = MagicMock(
            path=path,
            content=source,
            skipped=False,
        )
        enrichment = MagicMock(
            fileContents=[file_content],
            fileMetadata=[],
        )
        request = MagicMock(
            deltaDiff=None,
            rawDiff=raw_diff,
            taskContext=None,
            enrichmentData=enrichment,
            projectRules=[],
            previousCodeAnalysisIssues=[],
            changedFiles=[path],
            deletedFiles=[],
            currentCommitHash="a" * 40,
        )
        processed = DiffProcessor().process(raw_diff)
        prepared = _build_stage_1_prepared_context(
            request,
            processed,
            is_incremental=False,
        )
        batch = [{
            "file": ReviewFile(
                path=path,
                focus_areas=["general"],
                risk_level="LOW",
            ),
            "priority": "LOW",
        }]

        with patch(
            "service.review.orchestrator.stage_1_file_review."
            "_invoke_stage_1_batch_llm",
            new_callable=AsyncMock,
            return_value=[],
        ) as invoke:
            await review_file_batch(
                MagicMock(),
                request,
                batch,
                rag_client=None,
                prepared_context=prepared,
            )

        prompt = invoke.await_args.args[1]
        assert "Type: ADDED" in prompt
        assert (
            "[Complete post-change source is present once as the added side "
            "of the diff below"
        ) in prompt
        assert "\nfirst()\nsecond()\n\nDiff:" not in prompt
        assert "+first()\n+second()" in prompt

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_current_source_bounds_hunkless_files_without_losing_ends(self):
        paths = ["src/first.py", "src/second.py"]
        source = "start\n" + ("middle\n" * 2_000) + "end\n"
        enrichment = MagicMock(
            fileContents=[
                MagicMock(path=path, content=source, skipped=False)
                for path in paths
            ],
            fileMetadata=[],
        )
        request = MagicMock(
            deltaDiff=None,
            rawDiff="",
            taskContext=None,
            enrichmentData=enrichment,
            projectRules=[],
            previousCodeAnalysisIssues=[],
            changedFiles=paths,
            deletedFiles=[],
            currentCommitHash="a" * 40,
        )
        prepared = _build_stage_1_prepared_context(
            request,
            None,
            is_incremental=False,
        )
        batch = [
            {
                "file": ReviewFile(
                    path=path,
                    focus_areas=["general"],
                    risk_level="LOW",
                ),
                "priority": "LOW",
            }
            for path in paths
        ]

        with patch(
            "service.review.orchestrator.stage_1_file_review."
            "_invoke_stage_1_batch_llm",
            new_callable=AsyncMock,
            return_value=[],
        ) as invoke:
            await review_file_batch(
                MagicMock(),
                request,
                batch,
                rag_client=None,
                prepared_context=prepared,
            )

        prompt = invoke.await_args.args[1]
        source_marker = "Current File Content (post-change):\n"
        current_source_sections = prompt.split(source_marker)[1:]
        assert len(current_source_sections) == 2
        bounded_sections = [
            section.split("\n\nDiff:\n", 1)[0]
            for section in current_source_sections
        ]
        assert all(source not in section for section in bounded_sections)
        assert all(section.startswith("start\n") for section in bounded_sections)
        assert all(section.endswith("end\n") for section in bounded_sections)
        assert all("Current file context truncated" in section for section in bounded_sections)
        assert all(
            len(section) <= STAGE1_CURRENT_SOURCE_BATCH_CHAR_BUDGET // len(paths)
            for section in bounded_sections
        )

    def test_cloudflare_structured_output_disabled_by_default(self):
        ChatCloudflareOpenAI = type("ChatCloudflareOpenAI", (), {})

        assert _supports_structured_output(ChatCloudflareOpenAI()) is False

    def test_oversized_processed_diff_does_not_reload_full_raw(self):
        raw_diff = """\
diff --git a/src/big.py b/src/big.py
--- a/src/big.py
+++ b/src/big.py
@@ -1 +1,3 @@
+first_changed_line()
+second_changed_line()
"""
        summarized = DiffFile(
            path="src/big.py",
            change_type=DiffChangeType.MODIFIED,
            content="[summary only]",
            is_skipped=False,
            skip_reason="File too large: 999999 bytes > 1",
        )
        request = MagicMock(rawDiff=raw_diff, deltaDiff=None, enrichmentData=None, taskContext=None)

        prepared = _build_stage_1_prepared_context(
            request,
            ProcessedDiff(files=[summarized]),
            is_incremental=False,
        )
        diff_file = _find_diff_file_for_path(prepared, "src/big.py")

        assert diff_file is summarized
        assert diff_file.content == "[summary only]"
        assert "first_changed_line" not in diff_file.content
        assert prepared.full_diff_index_loaded is False
        assert prepared.full_diff_by_path == {}

        full_diff_file = _find_diff_file_for_path(
            prepared,
            "src/big.py",
            use_full_diff=True,
        )

        assert full_diff_file is summarized
        assert full_diff_file.content == "[summary only]"
        assert prepared.full_diff_index_loaded is False
        assert prepared.full_diff_by_path == {}

    def test_globally_compacted_diff_does_not_reload_full_raw(self):
        raw_diff = """\
diff --git a/src/after_limit.py b/src/after_limit.py
--- a/src/after_limit.py
+++ b/src/after_limit.py
@@ -1 +1,3 @@
+first_changed_line()
+second_changed_line()
"""
        summarized = DiffFile(
            path="src/after_limit.py",
            change_type=DiffChangeType.MODIFIED,
            content="[summary only]",
            is_skipped=False,
            skip_reason="Would exceed total size limit: 120000",
        )
        request = MagicMock(rawDiff=raw_diff, deltaDiff=None, enrichmentData=None, taskContext=None)

        prepared = _build_stage_1_prepared_context(
            request,
            ProcessedDiff(files=[summarized]),
            is_incremental=False,
        )
        diff_file = _find_diff_file_for_path(prepared, "src/after_limit.py")

        assert diff_file is summarized
        assert diff_file.content == "[summary only]"
        assert "first_changed_line" not in diff_file.content
        assert prepared.full_diff_index_loaded is False
        assert prepared.full_diff_by_path == {}

        full_diff_file = _find_diff_file_for_path(
            prepared,
            "src/after_limit.py",
            use_full_diff=True,
        )

        assert full_diff_file is summarized
        assert full_diff_file.content == "[summary only]"
        assert "second_changed_line" not in full_diff_file.content
        assert prepared.full_diff_index_loaded is False
        assert prepared.full_diff_by_path == {}


class TestLargeDiffSegmentation:
    def test_chunk_diff_preserves_file_header_and_hunk_headers(self):
        diff = """\
diff --git a/src/big.py b/src/big.py
--- a/src/big.py
+++ b/src/big.py
@@ -1 +1,2 @@
+aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
@@ -10 +11,2 @@
+bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
"""

        chunks = _chunk_diff_preserving_hunks(diff, max_input_tokens=20)

        assert len(chunks) > 1
        assert all("diff --git a/src/big.py b/src/big.py" in chunk for chunk in chunks)
        assert any("@@ -1,0 +1,1 @@" in chunk for chunk in chunks)
        assert any("@@ -10,0 +11,1 @@" in chunk for chunk in chunks)

    def test_split_hunk_recomputes_each_fragment_coordinates(self):
        hunk = (
            "@@ -100,3 +200,3 @@ def changed():\n"
            " context_one_xxxxxxxxx\n"
            "-removed_two_xxxxxxxxx\n"
            "+added_two_xxxxxxxxxxx\n"
            " context_three_xxxxxxx\n"
        )

        chunks = _split_hunk_by_lines(hunk, max_chars=55)

        assert len(chunks) == 4
        assert chunks[0].startswith("@@ -100,1 +200,1 @@ def changed():")
        assert chunks[1].startswith("@@ -101,1 +201,0 @@ def changed():")
        assert chunks[2].startswith("@@ -102,0 +201,1 @@ def changed():")
        assert chunks[3].startswith("@@ -102,1 +202,1 @@ def changed():")

    def test_owned_segments_require_every_fragment_before_hunk_is_reviewed(self):
        diff = """\
diff --git a/src/big.py b/src/big.py
--- a/src/big.py
+++ b/src/big.py
@@ -10,4 +10,4 @@ def changed():
 context_one_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
-removed_two_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
+added_two_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
 context_three_xxxxxxxxxxxxxxxxxxxxxxxxxx
"""
        processed = DiffProcessor().process(diff)
        diff_file = processed.files[0]
        file_info = ReviewFile(
            path="src/big.py",
            focus_areas=[],
            risk_level="MEDIUM",
        )
        prepared = Stage1PreparedContext(
            diff_source=processed,
            diff_by_path={"src/big.py": diff_file},
        )

        expanded = _expand_oversized_diff_batches(
            [[{"file": file_info, "priority": "MEDIUM"}]],
            prepared,
            diff_chunk_token_budget=24,
        )
        unit_ids = tuple(
            batch[0]["_review_unit_id"]
            for batch in expanded
        )
        state = Stage1ReviewUnitState()
        state.register_batches(expanded)

        assert len(expanded) > 1
        assert len(set(unit_ids)) == len(unit_ids)
        assert {
            hunk_id
            for batch in expanded
            for hunk_id in batch[0]["_hunk_ids"]
        } == {diff_file.hunks[0].id}

        state.mark_completed(unit_ids[:-1])
        assert state.reviewed_hunk_ids == ()
        with pytest.raises(RuntimeError, match="coverage is incomplete"):
            state.assert_complete()

        state.mark_completed(unit_ids[-1:])
        state.assert_complete()
        assert state.reviewed_hunk_ids == (diff_file.hunks[0].id,)

    def test_duplicate_review_unit_assignment_fails_closed(self):
        file_info = ReviewFile(
            path="src/a.py",
            focus_areas=[],
            risk_level="MEDIUM",
        )
        item = {
            "file": file_info,
            "_review_unit_id": "sha256:unit",
            "_hunk_ids": ("sha256:hunk",),
        }

        with pytest.raises(RuntimeError, match="assigned more than once"):
            Stage1ReviewUnitState().register_batches([[item], [dict(item)]])

    def test_expand_oversized_batches_creates_segment_batches(self):
        file_info = ReviewFile(path="src/big.py", focus_areas=[], risk_level="MEDIUM")
        diff = """\
diff --git a/src/big.py b/src/big.py
--- a/src/big.py
+++ b/src/big.py
@@ -1 +1,2 @@
+aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
@@ -10 +11,2 @@
+bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
"""
        diff_file = DiffFile(path="src/big.py", change_type=DiffChangeType.MODIFIED, content=diff)
        prepared = Stage1PreparedContext(
            diff_source=ProcessedDiff(files=[diff_file]),
            diff_by_path={"src/big.py": diff_file},
        )

        expanded = _expand_oversized_diff_batches(
            [[{"file": file_info, "priority": "MEDIUM"}]],
            prepared,
            diff_chunk_token_budget=20,
        )

        assert len(expanded) > 1
        assert all(len(batch) == 1 for batch in expanded)
        assert expanded[0][0]["_diff_chunk_total"] == len(expanded)

    def test_size_limited_diff_retains_bounded_summary_without_focus_flag(self):
        raw_diff = """\
diff --git a/src/big.py b/src/big.py
--- a/src/big.py
+++ b/src/big.py
@@ -1 +1,5 @@
+aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
+bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
+cccccccccccccccccccccccccccccccccccccccccccccc
+dddddddddddddddddddddddddddddddddddddddddddddd
"""
        summarized = DiffFile(
            path="src/big.py",
            change_type=DiffChangeType.MODIFIED,
            content="[summary only]",
            is_skipped=False,
            skip_reason="File too large: 999999 bytes > 1",
        )
        request = MagicMock(rawDiff=raw_diff, deltaDiff=None, enrichmentData=None, taskContext=None)
        prepared = _build_stage_1_prepared_context(
            request,
            ProcessedDiff(files=[summarized]),
            is_incremental=False,
        )
        file_info = ReviewFile(path="src/big.py", focus_areas=[], risk_level="MEDIUM")

        expanded = _expand_oversized_diff_batches(
            [[{"file": file_info, "priority": "MEDIUM"}]],
            prepared,
            diff_chunk_token_budget=20,
        )

        assert len(expanded) == 1
        assert expanded[0][0]["_hunk_ids"] == ()
        assert "_diff_chunk_total" not in expanded[0][0]
        assert _find_diff_file_for_path(prepared, "src/big.py") is summarized
        assert "first_changed_line" not in summarized.content
        assert prepared.full_diff_index_loaded is False

    def test_full_diff_focus_cannot_bypass_bounded_diff_admission(self):
        raw_diff = """\
diff --git a/src/big.py b/src/big.py
--- a/src/big.py
+++ b/src/big.py
@@ -1 +1,5 @@
+aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
+bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
+cccccccccccccccccccccccccccccccccccccccccccccc
+dddddddddddddddddddddddddddddddddddddddddddddd
"""
        summarized = DiffFile(
            path="src/big.py",
            change_type=DiffChangeType.MODIFIED,
            content="[summary only]",
            is_skipped=False,
            skip_reason="File too large: 999999 bytes > 1",
        )
        request = MagicMock(rawDiff=raw_diff, deltaDiff=None, enrichmentData=None, taskContext=None)
        prepared = _build_stage_1_prepared_context(
            request,
            ProcessedDiff(files=[summarized]),
            is_incremental=False,
        )
        file_info = ReviewFile(
            path="src/big.py",
            focus_areas=["FULL_DIFF_REVIEW"],
            risk_level="MEDIUM",
        )

        expanded = _expand_oversized_diff_batches(
            [[{"file": file_info, "priority": "MEDIUM"}]],
            prepared,
            diff_chunk_token_budget=20,
        )

        assert len(expanded) == 1
        assert expanded[0][0]["_hunk_ids"] == ()
        assert "_diff_chunk_total" not in expanded[0][0]
        assert _find_diff_file_for_path(
            prepared,
            "src/big.py",
            use_full_diff=True,
        ) is summarized
        assert "first_changed_line" not in summarized.content
        assert prepared.full_diff_index_loaded is False


# ── Structured metadata formatting ───────────────────────────────

class TestBatchEnrichmentMetadataScoping:
    def test_same_basename_in_another_module_is_not_selected(self):
        checkout = MagicMock(path="app/code/Acme/Checkout/etc/di.xml")
        cart = MagicMock(path="app/code/Acme/Cart/etc/di.xml")
        request = MagicMock()
        request.enrichmentData.fileMetadata = [cart, checkout]

        result = _iter_batch_enrichment_metadata(
            request,
            ["app/code/Acme/Checkout/etc/di.xml"],
            prepared_context=None,
        )

        assert result == [checkout]

    def test_absolute_prefix_metadata_matches_repository_path(self):
        checkout = MagicMock(
            path="/tmp/checkout/app/code/Acme/Checkout/etc/di.xml"
        )
        request = MagicMock()
        request.enrichmentData.fileMetadata = [checkout]

        result = _iter_batch_enrichment_metadata(
            request,
            ["app/code/Acme/Checkout/etc/di.xml"],
            prepared_context=None,
        )

        assert result == [checkout]


class TestStructuredMetadataFormatting:
    def test_metadata_is_serialized_as_json_without_outline_truncation(self):
        meta = MagicMock()
        meta.model_dump.return_value = {
            "path": "src/Foo.py",
            "imports": [f"pkg{i}" for i in range(25)],
            "symbolNames": [f"symbol{i}" for i in range(35)],
            "calls": [f"call{i}" for i in range(20)],
        }

        result = _format_batch_metadata_json([meta])

        assert '"path":"src/Foo.py"' in result
        assert "pkg24" in result
        assert "symbol34" in result
        assert "call19" in result

    def test_large_plugin_metadata_is_bounded_with_omission_marker(self):
        meta = {
            "path": "app/code/Acme/Checkout/Model/Cart.php",
            "language": "php",
            "pluginSpecificFacts": [
                {
                    "relation": f"relation-{index}",
                    "target": "x" * 200,
                }
                for index in range(500)
            ],
        }

        result = _format_batch_metadata_json([meta])

        assert "app/code/Acme/Checkout/Model/Cart.php" in result
        assert "relation-0" in result
        assert "relation-499" not in result
        assert "_codecrowOmittedItems" in result
        assert len(result) <= STAGE1_METADATA_CHAR_BUDGET

    def test_metadata_projection_is_deterministic_and_schema_neutral(self):
        first = {
            "path": "src/Foo.php",
            "frameworkExtension": {
                "zeta": ["z2", "z1"],
                "alpha": "value",
            },
        }
        second = {
            "frameworkExtension": {
                "alpha": "value",
                "zeta": ["z2", "z1"],
            },
            "path": "src/Foo.php",
        }

        assert _format_batch_metadata_json([first]) == _format_batch_metadata_json([second])

    def test_metadata_identifiers_only_use_structural_symbol_fields(self):
        meta = {
            "path": "src/Foo.py",
            "imports": ["KnownDependency"],
            "symbolNames": ["KnownSymbol"],
            "unknownParserField": {
                "frameworkSpecificName": "FrameworkThing",
                "nested": ["NestedValue"],
            },
        }

        result = _extract_metadata_identifiers([meta])

        assert result == ["KnownDependency"]
        assert "KnownSymbol" not in result
        assert "src/Foo.py" not in result
        assert "FrameworkThing" not in result
        assert "NestedValue" not in result

    def test_metadata_identifier_expansion_has_finite_structural_name_cap(self):
        imports = [f"Dependency{index}" for index in range(350)]

        result = _extract_metadata_identifiers([{"imports": imports}])

        assert result == imports[:200]
        assert result[-1] == "Dependency199"


# ── Deterministic RAG normalization ──────────────────────────────

class TestDeterministicRagNormalization:
    def test_default_flattening_uses_deterministic_chunk_cap(self):
        response = {
            "context": {
                "changed_files": {
                    "src/a.py": [
                        {
                            "text": f"exact chunk {index}",
                            "metadata": {"path": f"src/related-{index}.py"},
                        }
                        for index in range(125)
                    ],
                },
            },
        }

        chunks = _flatten_deterministic_context(response)

        assert 0 < len(chunks) <= DETERMINISTIC_RAG_MAX_CHUNKS
        assert chunks[0]["text"] == "exact chunk 0"
        assert not any(chunk["text"] == "exact chunk 124" for chunk in chunks)

    def test_flattens_all_deterministic_groups(self):
        response = {
            "context": {
                "chunks": [
                    {"text": "all chunk", "metadata": {"path": "src/all.py"}},
                ],
                "changed_files": {
                    "src/a.py": [
                        {"text": "changed", "metadata": {"path": "src/a.py"}},
                    ],
                },
                "related_definitions": {
                    "Thing": [
                        {"text": "definition", "metadata": {"path": "src/thing.py"}},
                    ],
                },
            }
        }

        chunks = _flatten_deterministic_context(response)
        texts = {chunk["text"] for chunk in chunks}

        assert {"changed", "definition"} <= texts
        assert "all chunk" not in texts
        assert all(chunk["_source"] == "deterministic" for chunk in chunks)

    def test_pr_architecture_packet_retains_pr_indexed_freshness(self):
        response = {
            "context": {
                "architecture_context": {
                    "plugin": [{
                        "text": "fresh effective plugin relation",
                        "metadata": {
                            "path": "__analysis_architecture__/magento/plugin.context",
                            "pr": True,
                            "pr_number": 42,
                            "architecture_kind": "magento-interception",
                        },
                    }],
                },
            },
        }

        chunks = _flatten_deterministic_context(response)

        assert len(chunks) == 1
        assert chunks[0]["_source"] == "pr_indexed"
        assert chunks[0]["_match_type"] == "architecture_relation"

    def test_distinct_chunks_with_equal_long_prefix_are_not_collapsed(self):
        shared_prefix = "Deterministic repository architecture context\n" + (
            "same-prefix " * 60
        )
        response = {
            "context": {
                "architecture_context": {
                    "first": [{
                        "text": shared_prefix + "first-late-fact",
                        "metadata": {
                            "path": "__analysis_architecture__/shared.context",
                            "architecture_kind": "magento-layout",
                        },
                    }],
                    "second": [{
                        "text": shared_prefix + "second-late-fact",
                        "metadata": {
                            "path": "__analysis_architecture__/shared.context",
                            "architecture_kind": "magento-layout",
                        },
                    }],
                },
            },
        }

        chunks = _flatten_deterministic_context(response)

        assert len(chunks) == 2
        assert {
            chunk["text"].rsplit(" ", 1)[-1]
            for chunk in chunks
        } == {
            "first-late-fact",
            "second-late-fact",
        }

    def test_exact_duplicate_chunk_across_grouped_views_is_collapsed(self):
        duplicate = {
            "text": "same exact deterministic fact",
            "metadata": {
                "path": "__analysis_architecture__/same.context",
                "architecture_kind": "magento-layout",
                "architecture_key": "magento-layout:same:0",
            },
        }
        response = {
            "context": {
                "architecture_context": {"packet": [duplicate]},
                "architecture_related": {"packet": [dict(duplicate)]},
                "chunks": [dict(duplicate)],
            },
        }

        chunks = _flatten_deterministic_context(response)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "same exact deterministic fact"

    def test_repeated_point_id_across_grouped_views_is_collapsed(self):
        duplicate = {
            "id": "qdrant-point-β",
            "text": "точний повтор 🧭",
            "metadata": {
                "path": "src/навігація.py",
                "start_line": 12,
                "end_line": 12,
            },
        }
        response = {
            "context": {
                "changed_files": {"src/навігація.py": [duplicate]},
                "related_definitions": {"Навігація": [dict(duplicate)]},
            },
        }

        chunks = _flatten_deterministic_context(response)

        assert len(chunks) == 1
        assert chunks[0]["id"] == "qdrant-point-β"

    def test_equal_unicode_text_at_distinct_ranges_is_preserved(self):
        text = "return 'однаковий результат ✨'"
        response = {
            "context": {
                "related_definitions": {
                    "Перший": [{
                        "text": text,
                        "metadata": {
                            "path": "src/приклад.py",
                            "start_line": 5,
                            "end_line": 5,
                        },
                    }],
                    "Другий": [{
                        "text": text,
                        "metadata": {
                            "path": "src/приклад.py",
                            "start_line": 25,
                            "end_line": 25,
                        },
                    }],
                },
            },
        }

        chunks = _flatten_deterministic_context(response)

        assert len(chunks) == 2
        assert {
            chunk["metadata"]["start_line"] for chunk in chunks
        } == {5, 25}

    def test_architecture_relation_limit_round_robins_neutral_kinds(self):
        response = {
            "context": {
                "architecture_context": {
                    f"packet-{index}": [{
                        "text": f"relation {index}",
                        "metadata": {
                            "path": f"__analysis_architecture__/packet-{index}.context",
                            "architecture_kind": f"kind-{index % 3}",
                            "architecture_key": f"packet-{index}",
                        },
                    }]
                    for index in range(9)
                },
                "related_definitions": {
                    "RequiredType": [{
                        "text": "class RequiredType {}",
                        "metadata": {"path": "src/RequiredType.php"},
                    }],
                },
            },
        }

        chunks = _flatten_deterministic_context(response, max_chunks=4)

        relations = [
            chunk for chunk in chunks
            if chunk["_match_type"] == "architecture_relation"
        ]
        assert {
            chunk["metadata"]["architecture_kind"] for chunk in relations
        } == {"kind-0", "kind-1", "kind-2"}
        assert any(
            chunk["_match_type"] == "definition" for chunk in chunks
        )

    def test_architecture_relation_limit_covers_distinct_review_paths_first(self):
        architecture_context = {
            f"dominant-{index}": [{
                "text": f"dominant relation {index}",
                "metadata": {
                    "path": (
                        "__analysis_architecture__/"
                        f"dominant-{index}.context"
                    ),
                    "architecture_kind": f"kind-{index}",
                    "architecture_key": f"dominant-{index}",
                },
                "_matched_on": "src/dominant.py",
            }]
            for index in range(8)
        }
        architecture_context["quiet"] = [{
            "text": "quiet file exact relation",
            "metadata": {
                "path": "__analysis_architecture__/quiet.context",
                "architecture_kind": "kind-quiet",
                "architecture_key": "quiet",
            },
            "_matched_on": "src/quiet.py",
        }]
        response = {
            "context": {
                "architecture_context": architecture_context,
                "related_definitions": {
                    "RequiredType": [{
                        "text": "class RequiredType {}",
                        "metadata": {"path": "src/RequiredType.php"},
                    }],
                },
            },
        }

        chunks = _flatten_deterministic_context(response, max_chunks=4)

        assert "quiet file exact relation" in {
            chunk["text"] for chunk in chunks
        }
        assert {
            chunk["_matched_on"]
            for chunk in chunks
            if chunk["_match_type"] == "architecture_relation"
        } == {"src/dominant.py", "src/quiet.py"}

    def test_related_bodies_precede_surplus_relations_for_same_paths(self):
        related_paths = ["src/related-a.py", "src/related-b.py"]

        def relation(index):
            return {
                "text": f"relation {index}",
                "metadata": {
                    "path": (
                        "__analysis_architecture__/"
                        f"relation-{index}.context"
                    ),
                    "architecture_kind": "python-import",
                    "architecture_key": f"relation-{index}",
                    "architecture_paths": ["src/review.py", *related_paths],
                },
                "_matched_on": "src/review.py",
            }

        response = {
            "context": {
                "architecture_context": {
                    "relation-0": [relation(0)],
                    "relation-1": [relation(1)],
                },
                "architecture_related": {
                    path: [{
                        "text": f"implementation for {path}",
                        "metadata": {
                            "path": path,
                            "structural_record_type": "source_chunk",
                        },
                    }]
                    for path in related_paths
                },
            },
        }

        chunks = _flatten_deterministic_context(response, max_chunks=4)

        assert [chunk["_match_type"] for chunk in chunks] == [
            "architecture_relation",
            "architecture_related",
            "architecture_related",
            "architecture_relation",
        ]
        assert [
            chunk["metadata"]["path"] for chunk in chunks[1:3]
        ] == related_paths

    def test_related_body_survives_related_path_in_match_provenance(self):
        relation = {
            "text": "exact relation",
            "metadata": {
                "path": "__analysis_architecture__/relation.context",
                "architecture_kind": "data-contract-reference",
                "architecture_key": "relation",
                "architecture_paths": [
                    "src/review.py",
                    "src/related.py",
                ],
            },
            "_matched_on": "src/review.py,src/related.py",
        }
        response = {
            "context": {
                "architecture_context": {"relation": [relation]},
                "architecture_related": {
                    "src/related.py": [{
                        "text": "RELATED_IMPLEMENTATION_BODY",
                        "metadata": {"path": "src/related.py"},
                    }],
                },
            },
        }

        chunks = _flatten_deterministic_context(
            response,
            max_chunks=2,
            reviewed_paths=["src/review.py"],
        )

        assert [chunk["_match_type"] for chunk in chunks] == [
            "architecture_relation",
            "architecture_related",
        ]
        assert chunks[1]["text"] == "RELATED_IMPLEMENTATION_BODY"

    def test_dense_relations_keep_bounded_relation_source_pairs(self):
        relation_count = 60
        architecture_context = {}
        architecture_related = {}
        for index in range(relation_count):
            related_path = f"src/related-{index:02}.py"
            architecture_context[f"relation-{index:02}"] = [{
                "text": (
                    f"RELATION_FACT_{index:02} "
                    + "relation detail " * 38
                ),
                "metadata": {
                    "path": (
                        "__analysis_architecture__/"
                        f"relation-{index:02}.context"
                    ),
                    "architecture_kind": f"kind-{index:02}",
                    "architecture_key": f"relation-{index:02}",
                    "architecture_paths": [
                        "src/review.py",
                        related_path,
                    ],
                },
                "_matched_on": "src/review.py",
            }]
            architecture_related[related_path] = [{
                "text": (
                    f"IMPLEMENTATION_BODY_{index:02}\n"
                    + "implementation statement\n" * 20
                ),
                "metadata": {
                    "path": related_path,
                    "structural_record_type": "source_chunk",
                },
            }]

        chunks = _flatten_deterministic_context({
            "context": {
                "architecture_context": architecture_context,
                "architecture_related": architecture_related,
            },
        })

        assert len(chunks) == DETERMINISTIC_RAG_MAX_CHUNKS
        assert [chunk["_match_type"] for chunk in chunks[::2]] == [
            "architecture_relation"
        ] * (DETERMINISTIC_RAG_MAX_CHUNKS // 2)
        assert [chunk["_match_type"] for chunk in chunks[1::2]] == [
            "architecture_related"
        ] * (DETERMINISTIC_RAG_MAX_CHUNKS // 2)
        relation_paths = {
            path
            for relation in chunks[::2]
            for path in relation["metadata"]["architecture_paths"]
        }
        assert {
            body["metadata"]["path"] for body in chunks[1::2]
        } <= relation_paths

        formatted = format_rag_context({"relevant_code": chunks})

        assert "RELATION_FACT_" in formatted
        assert "IMPLEMENTATION_BODY_" in formatted

    def test_architecture_kind_prefers_diagnostic_then_resolved_fact(self):
        def relation(name, fact):
            return {
                "text": name,
                "metadata": {
                    "path": f"__analysis_architecture__/{name}.context",
                    "architecture_kind": "magento-di",
                    "architecture_key": name,
                    "plugin_graph_facts": [fact],
                },
            }

        response = {
            "context": {
                "architecture_context": {
                    "coarse": [relation("coarse", {
                        "kind": "php-inheritance",
                        "source": "Child",
                        "relation": "extends",
                        "target": "Parent",
                        "attributes": {},
                        "related_paths": [],
                    })],
                    "resolved": [relation("resolved", {
                        "kind": "php-inheritance",
                        "source": "Acme\\Child",
                        "relation": "extends",
                        "target": "Acme\\Parent",
                        "attributes": {"targetKind": "class"},
                        "related_paths": ["src/Parent.php"],
                    })],
                    "diagnostic": [relation("diagnostic", {
                        "kind": "magento-interceptor-inapplicable",
                        "source": "Acme\\Audit",
                        "relation": "cannot-intercept",
                        "target": "Acme\\FinalCart::save",
                        "attributes": {"semanticRole": "diagnostic"},
                        "related_paths": ["src/FinalCart.php"],
                    })],
                },
                "related_definitions": {
                    "RequiredType": [{
                        "text": "class RequiredType {}",
                        "metadata": {"path": "src/RequiredType.php"},
                    }],
                },
            },
        }

        first = _flatten_deterministic_context(response, max_chunks=2)
        second = _flatten_deterministic_context(response, max_chunks=3)

        assert [chunk["text"] for chunk in first] == [
            "diagnostic",
            "class RequiredType {}",
        ]
        assert [chunk["text"] for chunk in second] == [
            "diagnostic",
            "resolved",
            "class RequiredType {}",
        ]

    def test_source_relation_precedes_duplicate_area_projections_at_chunk_cap(self):
        def relation(name, relation_name, attributes):
            fact = {
                "kind": "php-constructor-dependency",
                "source": "Acme\\Checkout\\Model\\Service",
                "relation": relation_name,
                "target": "Acme\\Checkout\\Api\\CartInterface",
                "path": "app/code/Acme/Checkout/Model/Service.php",
                "line": 1,
                "attributes": attributes,
                "related_paths": [
                    "app/code/Acme/Checkout/Api/CartInterface.php",
                ],
            }
            return {
                "text": (
                    f"[{fact['kind']}] {fact['source']} "
                    f"{fact['relation']} {fact['target']}"
                ),
                "metadata": {
                    "path": fact["path"],
                    "architecture_kind": fact["kind"],
                    "architecture_key": name,
                    "plugin_graph_facts": [fact],
                },
                "_matched_on": fact["path"],
            }

        area_projections = {
            f"area-{index:02}": [relation(
                f"area-{index:02}",
                "requests",
                {"area": f"area-{index:02}"},
            )]
            for index in range(20)
        }
        source_relation = relation(
            "zz-source-relation",
            "constructor-requires",
            {"sourceKind": "class", "targetKind": "interface"},
        )
        response = {
            "context": {
                "architecture_context": {
                    **area_projections,
                    "zz-source-relation": [source_relation],
                },
            },
        }

        chunks = _flatten_deterministic_context(response, max_chunks=4)

        assert len(chunks) == 4
        assert source_relation["text"] in {
            chunk["text"] for chunk in chunks
        }
        assert chunks[0]["text"] == source_relation["text"]

    def test_empty_context_has_no_chunks(self):
        assert _rag_context_has_chunks({"relevant_code": []}) is False
        assert _rag_context_has_chunks({"context": {"relevant_code": []}}) is False
        assert _rag_context_has_chunks({"relevant_code": [{"text": "x"}]}) is True



class TestFetchBatchRagContext:
    @staticmethod
    def _request():
        request = MagicMock()
        request.get_rag_branch.return_value = "feature"
        request.get_rag_base_branch.return_value = "main"
        request.projectWorkspace = "ws"
        request.projectNamespace = "proj"
        request.pullRequestId = 12
        request.changedFiles = ["src/a.py"]
        request.currentCommitHash = "a" * 40
        request.commitHash = "a" * 40
        request.baseCommitHash = "b" * 40
        request.targetHeadCommitHash = "t" * 40
        request.get_target_head_commit_hash.return_value = "t" * 40
        request.ragCollectionTarget = "cc_ws_proj_main_generation"
        request.ragBaseGenerationManifestSha256 = "c" * 64
        request.ragPrGenerationFingerprint = "d" * 64
        request.ragPrOverlayGenerationManifestSha256 = "e" * 64
        return request

    @pytest.mark.asyncio(loop_scope="function")
    async def test_exact_structural_context_uses_revision_binding(self):
        client = MagicMock()
        client.get_deterministic_context = AsyncMock(return_value={
            "context": {
                "chunks": [{
                    "text": "class Dependency: pass",
                    "metadata": {"path": "src/dependency.py"},
                    "_match_type": "definition",
                }],
                "_metadata": {"retrieval_state": "complete"},
            },
        })
        state = Stage1RagState()

        result = await fetch_batch_rag_context(
            client,
            self._request(),
            ["src/a.py"],
            pr_indexed=True,
            enrichment_identifiers=["Dependency"],
            rag_state=state,
        )

        assert result["relevant_code"][0]["_match_type"] == "definition"
        request_payload = client.get_deterministic_context.await_args.kwargs
        assert request_payload["source_revision"] == "a" * 40
        assert request_payload["base_revision"] == "t" * 40
        assert request_payload["collection_target"] == "cc_ws_proj_main_generation"
        assert request_payload["additional_identifiers"] == ["Dependency"]
        assert state.deterministic_retrieval_states == ["complete"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_partial_structural_context_keeps_exact_chunks_and_later_batches(self):
        client = MagicMock()
        client.get_deterministic_context = AsyncMock(return_value={
            "context": {
                "chunks": [{
                    "text": "class ExactDependency: pass",
                    "metadata": {"path": "src/exact_dependency.py"},
                    "_match_type": "definition",
                }],
                "_metadata": {
                    "retrieval_state": "partial",
                    "coverage_state": "partial",
                    "context_usable": True,
                    "partial_reasons": ["global_matching_point_limit"],
                },
            },
        })
        state = Stage1RagState()

        first = await fetch_batch_rag_context(
            client,
            self._request(),
            ["src/a.py"],
            pr_indexed=True,
            rag_state=state,
        )
        second = await fetch_batch_rag_context(
            client,
            self._request(),
            ["src/b.py"],
            pr_indexed=True,
            rag_state=state,
        )

        assert first["relevant_code"][0]["text"] == "class ExactDependency: pass"
        assert first["_metadata"]["retrieval_state"] == "partial"
        assert first["_metadata"]["partial_reasons"] == [
            "global_matching_point_limit"
        ]
        formatted = format_rag_context(
            first,
            {"src/a.py"},
            pr_changed_files=["src/a.py"],
        )
        assert formatted
        assert "class ExactDependency: pass" in formatted
        assert second is not None
        assert client.get_deterministic_context.await_count == 2
        assert state.context_disabled is False
        assert state.deterministic_retrieval_states == ["partial", "partial"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_unavailable_structural_context_opens_optional_context_circuit(self):
        client = MagicMock()
        client.get_deterministic_context = AsyncMock(return_value={
            "context": {
                "chunks": [],
                "_metadata": {
                    "retrieval_state": "unavailable",
                    "coverage_state": "unavailable",
                    "context_usable": False,
                    "partial_reasons": ["collection_not_found"],
                },
            },
        })
        state = Stage1RagState()

        first = await fetch_batch_rag_context(
            client,
            self._request(),
            ["src/a.py"],
            pr_indexed=True,
            rag_state=state,
        )
        second = await fetch_batch_rag_context(
            client,
            self._request(),
            ["src/b.py"],
            pr_indexed=True,
            rag_state=state,
        )

        assert first is None
        assert second is None
        assert client.get_deterministic_context.await_count == 1
        assert state.context_disabled is True
        assert state.deterministic_retrieval_states == ["unavailable"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_missing_branch_disables_optional_context_without_query(self):
        request = self._request()
        request.get_rag_branch.return_value = None
        client = MagicMock()
        client.get_deterministic_context = AsyncMock()
        state = Stage1RagState()

        result = await fetch_batch_rag_context(
            client,
            request,
            ["src/a.py"],
            pr_indexed=True,
            rag_state=state,
        )

        assert result is None
        assert state.context_disabled is True
        assert state.deterministic_retrieval_states == ["failed"]
        client.get_deterministic_context.assert_not_awaited()


class TestDeduplicatePrStaleChunks:
    def test_empty_chunks(self):
        assert _deduplicate_pr_stale_chunks([], ["a.py"], ["a.py"]) == []

    def test_empty_pr_files(self):
        chunks = [{"text": "code", "metadata": {"path": "a.py"}}]
        result = _deduplicate_pr_stale_chunks(chunks, [], ["a.py"])
        assert result == chunks

    def test_non_pr_file_kept(self):
        chunks = [{"text": "code", "metadata": {"path": "lib.py"}}]
        result = _deduplicate_pr_stale_chunks(chunks, ["a.py"], ["a.py"])
        assert len(result) == 1

    def test_pr_file_in_batch_kept(self):
        chunks = [
            {"text": "stale", "metadata": {"path": "a.py"}, "_source": "branch"},
            {"text": "fresh", "metadata": {"path": "a.py"}, "_source": "pr_indexed"},
        ]
        result = _deduplicate_pr_stale_chunks(chunks, ["a.py"], ["a.py"])
        assert len(result) == 2  # Both kept because file is in batch

    def test_pr_file_not_in_batch_prefers_pr_indexed(self):
        chunks = [
            {"text": "stale", "metadata": {"path": "a.py"}, "_source": "branch"},
            {"text": "fresh", "metadata": {"path": "a.py"}, "_source": "pr_indexed"},
        ]
        result = _deduplicate_pr_stale_chunks(chunks, ["a.py"], ["other.py"])
        assert len(result) == 1
        assert result[0]["_source"] == "pr_indexed"

    def test_no_pr_indexed_marks_stale(self):
        chunks = [
            {"text": "stale", "metadata": {"path": "a.py"}, "_source": "branch"},
        ]
        result = _deduplicate_pr_stale_chunks(chunks, ["a.py"], ["other.py"])
        assert len(result) == 1
        assert result[0].get("_potentially_stale") is True

    def test_no_metadata_path_uses_unknown(self):
        chunks = [{"text": "code"}]
        result = _deduplicate_pr_stale_chunks(chunks, ["a.py"], ["a.py"])
        assert len(result) == 1

    def test_basename_matching(self):
        chunks = [
            {"text": "code", "metadata": {"path": "src/a.py"}, "_source": "pr_indexed"},
            {"text": "stale", "metadata": {"path": "src/a.py"}, "_source": "branch"},
        ]
        result = _deduplicate_pr_stale_chunks(chunks, ["src/a.py"], ["other.py"])
        assert len(result) == 1
        assert result[0]["_source"] == "pr_indexed"

    def test_same_basename_in_different_module_is_not_a_pr_file(self):
        chunks = [{
            "text": "Cart module branch configuration",
            "metadata": {"path": "app/code/Acme/Cart/etc/di.xml"},
            "_source": "branch",
        }]

        result = _deduplicate_pr_stale_chunks(
            chunks,
            ["app/code/Acme/Checkout/etc/di.xml"],
            ["app/code/Acme/Checkout/etc/di.xml"],
        )

        assert result == chunks
        assert "_potentially_stale" not in result[0]


# ── _extract_calibrated_issues ───────────────────────────────────

class TestExtractCalibratedIssues:
    def _make_issue(self, severity="MEDIUM"):
        return CodeReviewIssue(
            id="i1",
            severity=severity,
            category="BUG",
            file="a.py",
            line=10,
            title="Test issue",
            reason="Test reason",
            suggestedFixDescription="Fix it",
        )

    def test_empty_batch(self):
        batch_output = FileReviewBatchOutput(reviews=[])
        result = _extract_calibrated_issues(batch_output)
        assert result == []

    def test_issues_returned(self):
        batch_output = FileReviewBatchOutput(reviews=[
            FileReviewOutput(
                file="a.py",
                analysis_summary="ok",
                issues=[self._make_issue()],
                confidence="HIGH",
            )
        ])
        result = _extract_calibrated_issues(batch_output)
        assert len(result) == 1

    def test_low_confidence_downgrades_high_to_medium(self):
        issue = self._make_issue(severity="HIGH")
        batch_output = FileReviewBatchOutput(reviews=[
            FileReviewOutput(
                file="a.py",
                analysis_summary="uncertain",
                issues=[issue],
                confidence="LOW",
            )
        ])
        result = _extract_calibrated_issues(batch_output)
        assert len(result) == 1
        assert result[0].severity == "MEDIUM"

    def test_low_confidence_does_not_downgrade_medium(self):
        issue = self._make_issue(severity="MEDIUM")
        batch_output = FileReviewBatchOutput(reviews=[
            FileReviewOutput(
                file="a.py",
                analysis_summary="ok",
                issues=[issue],
                confidence="LOW",
            )
        ])
        result = _extract_calibrated_issues(batch_output)
        assert result[0].severity == "MEDIUM"

    def test_high_confidence_keeps_high_severity(self):
        issue = self._make_issue(severity="HIGH")
        batch_output = FileReviewBatchOutput(reviews=[
            FileReviewOutput(
                file="a.py",
                analysis_summary="ok",
                issues=[issue],
                confidence="HIGH",
            )
        ])
        result = _extract_calibrated_issues(batch_output)
        assert result[0].severity == "HIGH"

    def test_multiple_reviews_aggregated(self):
        batch_output = FileReviewBatchOutput(reviews=[
            FileReviewOutput(
                file="a.py",
                analysis_summary="ok",
                issues=[self._make_issue(), self._make_issue()],
                confidence="HIGH",
            ),
            FileReviewOutput(
                file="b.py",
                analysis_summary="ok",
                issues=[self._make_issue()],
                confidence="MEDIUM",
            ),
        ])
        result = _extract_calibrated_issues(batch_output)
        assert len(result) == 3


# ── bounded rendered-input packing ───────────────────────────────


class TestRenderedInputPacking:
    def _batch(self, paths):
        return [{
            "file": ReviewFile(
                path=path,
                focus_areas=["general"],
                risk_level="MEDIUM",
            ),
            "priority": "MEDIUM",
            "_review_unit_id": f"unit:{path}",
            "_hunk_ids": (f"hunk:{path}",),
        } for path in paths]

    def test_agent_repository_tools_use_the_pinned_target_revision(self):
        request = _packing_request(["src/a.py"], None)
        request.useMcpTools = True
        request.localRepoRevision = "target-head-sha"
        request.baseCommitHash = "base-sha"
        request.targetBranchName = "main"
        request.projectVcsWorkspace = "team"
        request.projectVcsRepoSlug = "repo"
        prepared = _build_stage_1_prepared_context(request, None, False)
        material = _prepare_stage1_prompt_material(
            request,
            self._batch(["src/a.py"]),
            prepared,
            False,
        )

        prompt, _ = _render_stage1_prompt(
            material,
            "",
            use_mcp_tools=True,
        )

        assert "TARGET BRANCH/REVISION REF: target-head-sha" in prompt
        assert "TARGET BRANCH/REVISION REF: main" not in prompt

    def test_agent_repository_tools_fall_back_to_request_target_head(self):
        request = _packing_request(["src/a.py"], None)
        request.useMcpTools = True
        request.localRepoRevision = None
        request.targetHeadCommitHash = "target-head-sha"
        request.baseCommitHash = "merge-base-sha"
        request.get_target_head_commit_hash.return_value = "target-head-sha"
        request.targetBranchName = "main"
        prepared = _build_stage_1_prepared_context(request, None, False)
        material = _prepare_stage1_prompt_material(
            request,
            self._batch(["src/a.py"]),
            prepared,
            False,
        )

        prompt, _ = _render_stage1_prompt(
            material,
            "",
            use_mcp_tools=True,
        )

        assert "TARGET BRANCH/REVISION REF: target-head-sha" in prompt
        assert "TARGET BRANCH/REVISION REF: merge-base-sha" not in prompt

    def test_rendered_local_prompt_uses_bounded_source_projections(self):
        paths = ["src/a.py", "src/b.py"]
        sources = {
            path: (f"# {path}\n" + ("value = 1\n" * 8_000))
            for path in paths
        }
        enrichment = MagicMock(
            fileContents=[
                MagicMock(path=path, content=source, skipped=False)
                for path, source in sources.items()
            ],
            fileMetadata=[],
            relationships=[],
        )
        request = _packing_request(paths, enrichment)
        prepared = _build_stage_1_prepared_context(request, None, False)

        repacked = _repack_stage1_batches_by_rendered_input(
            [self._batch(paths)],
            request,
            prepared,
            False,
            token_budget=30_000,
        )

        assert [[item["file"].path for item in batch] for batch in repacked] == [
            paths,
        ]
        prompts = [
            _render_stage1_prompt(
                _prepare_stage1_prompt_material(
                    request, batch, prepared, False
                ),
                "",
            )[0]
            for batch in repacked
        ]
        assert all(_estimated_prompt_tokens(prompt) <= 30_000 for prompt in prompts)
        assert all(sources[path] not in prompts[0] for path in paths)
        assert prompts[0].count("Current file context truncated") == len(paths)
        assert "# src/a.py" in prompts[0]
        assert "# src/b.py" in prompts[0]

    def test_related_files_stay_together_when_rendered_prompt_fits(self):
        paths = ["src/a.py", "src/b.py"]
        relationship = SimpleNamespace(
            sourceFile=paths[0],
            targetFile=paths[1],
            relationshipType=SimpleNamespace(value="IMPORTS"),
            matchedOn="B",
        )
        enrichment = MagicMock(
            fileContents=[
                MagicMock(path=path, content="value = 1\n", skipped=False)
                for path in paths
            ],
            fileMetadata=[],
            relationships=[relationship],
        )
        request = _packing_request(paths, enrichment)
        prepared = _build_stage_1_prepared_context(request, None, False)
        batch = self._batch(paths)

        repacked = _repack_stage1_batches_by_rendered_input(
            [batch], request, prepared, False, token_budget=60_000
        )

        assert repacked == [batch]

    def test_bounded_source_projection_keeps_exact_dependency_capsule(self):
        paths = ["src/a.py", "src/b.py"]
        relationship = SimpleNamespace(
            sourceFile=paths[0],
            targetFile=paths[1],
            relationshipType=SimpleNamespace(value="IMPORTS"),
            matchedOn="B",
        )
        sources = {path: "value = 1\n" * 7_000 for path in paths}
        enrichment = MagicMock(
            fileContents=[
                MagicMock(path=path, content=source, skipped=False)
                for path, source in sources.items()
            ],
            fileMetadata=[],
            relationships=[relationship],
        )
        request = _packing_request(paths, enrichment)
        prepared = _build_stage_1_prepared_context(request, None, False)

        repacked = _repack_stage1_batches_by_rendered_input(
            [self._batch(paths)],
            request,
            prepared,
            False,
            token_budget=8_000,
        )

        assert len(repacked) == 2
        for batch in repacked:
            material = _prepare_stage1_prompt_material(
                request, batch, prepared, False
            )
            prompt, _ = _render_stage1_prompt(material, "")
            assert '"source": "src/a.py"' in prompt
            assert '"target": "src/b.py"' in prompt
            assert '"type": "IMPORTS"' in prompt
            assert '"matchedOn": "B"' in prompt

    def test_individual_source_is_bounded_without_new_invocations(self):
        path = "src/large.py"
        source = "".join(f"line_{index} = {index}\n" for index in range(12_000))
        enrichment = MagicMock(
            fileContents=[MagicMock(path=path, content=source, skipped=False)],
            fileMetadata=[],
            relationships=[],
        )
        request = _packing_request([path], enrichment)
        prepared = _build_stage_1_prepared_context(request, None, False)
        batches = [self._batch([path])]

        expanded = _expand_oversized_current_source_batches(
            batches,
            request,
            prepared,
            False,
            token_budget=18_000,
        )

        assert len(expanded) == 1
        material = _prepare_stage1_prompt_material(
            request,
            expanded[0],
            prepared,
            False,
        )
        bounded_source = material.batch_files_data[0]["current_code"]
        assert bounded_source != source
        assert bounded_source.startswith("line_0 = 0\n")
        assert bounded_source.endswith("line_11999 = 11999\n")
        assert "Current file context truncated" in bounded_source
        assert len(bounded_source) <= STAGE1_CURRENT_SOURCE_BATCH_CHAR_BUDGET
        assert expanded[0][0]["_hunk_ids"] == (f"hunk:{path}",)

    def test_oversized_shared_scaffold_does_not_spawn_review_calls(self):
        path = "src/shared.py"
        task_lines = [f"TASK_FACT_{index:04d} " + "t" * 80 for index in range(350)]
        project_lines = [
            f"PROJECT_RULE_{index:04d} " + "p" * 80
            for index in range(350)
        ]
        plugin_lines = [
            f"PLUGIN_RULE_{index:04d} " + "g" * 80
            for index in range(350)
        ]
        previous_lines = [
            f"PREVIOUS_ISSUE_FACT_{index:04d} " + "v" * 80
            for index in range(350)
        ]
        metadata_facts = [
            f"METADATA_FACT_{index:04d}_" + "m" * 80
            for index in range(350)
        ]
        metadata = SimpleNamespace(
            path=path,
            pluginSpecificFacts=metadata_facts,
        )
        enrichment = MagicMock(
            fileContents=[MagicMock(path=path, content="value = 1\n", skipped=False)],
            fileMetadata=[metadata],
            relationships=[],
        )
        request = _packing_request([path], enrichment)
        request.projectRules = json.dumps([{
            "title": "Lossless rules",
            "description": "\n".join(project_lines),
            "filePatterns": ["*.py"],
            "ruleType": "ENFORCE",
        }])
        request.previousCodeAnalysisIssues = [{
            "id": "previous-1",
            "status": "OPEN",
            "severity": "MEDIUM",
            "file": path,
            "line": 1,
            "reason": "\n".join(previous_lines),
        }]
        prepared = _build_stage_1_prepared_context(request, None, False)
        prepared.task_context = "\n".join(task_lines)
        batch = self._batch([path])
        token_budget = 8_000

        with patch(
            "service.review.orchestrator.stage_1_file_review.review_plugin_context",
            return_value="\n".join(plugin_lines),
        ):
            expanded = _expand_oversized_stage1_evidence_batches(
                [batch],
                request,
                prepared,
                False,
                token_budget,
            )

        assert len(expanded) == 1
        prompts = [
            _render_stage1_prompt(
                _prepare_stage1_prompt_material(
                    request,
                    batch,
                    prepared,
                    False,
                ),
                "",
            )[0]
            for batch in expanded
        ]
        assert all(
            _estimated_prompt_tokens(prompt) <= token_budget
            for prompt in prompts
        )

        prompt = prompts[0]
        assert "CodeCrow bounded optional Stage 1 context" in prompt
        assert "omitted character counts" in prompt
        assert "value = 1" in prompt
        assert task_lines[-1] not in prompt
        assert project_lines[-1] not in prompt
        assert plugin_lines[-1] not in prompt
        assert previous_lines[-1] not in prompt
        assert metadata_facts[-1] not in prompt

    def test_compacted_diff_and_source_have_bounded_truthful_coverage(self):
        path = "src/joint.py"
        source = "".join(
            f"SOURCE_LINE_{index:05d} = '{'s' * 64}'\n"
            for index in range(5_000)
        )
        diff_parts = [
            f"diff --git a/{path} b/{path}\n",
            f"--- a/{path}\n",
            f"+++ b/{path}\n",
        ]
        diff_sentinels = []
        for index in range(180):
            sentinel = f"DIFF_SENTINEL_{index:04d}_" + "d" * 96
            diff_sentinels.append(sentinel)
            line_number = index + 1
            diff_parts.extend([
                f"@@ -{line_number},1 +{line_number},1 @@\n",
                f"-old_{index:04d}\n",
                f"+{sentinel}\n",
            ])
        diff = "".join(diff_parts)
        processed = DiffProcessor().process(diff)
        assert len(processed.files) == 1
        enrichment = MagicMock(
            fileContents=[MagicMock(path=path, content=source, skipped=False)],
            fileMetadata=[],
            relationships=[],
        )
        request = _packing_request([path], enrichment)
        prepared = _build_stage_1_prepared_context(request, processed, False)
        item = {
            "file": ReviewFile(
                path=path,
                focus_areas=["general"],
                risk_level="MEDIUM",
            ),
            "priority": "MEDIUM",
        }
        token_budget = 9_000

        expanded = _expand_oversized_stage1_evidence_batches(
            [[item]],
            request,
            prepared,
            False,
            token_budget,
            max_units_per_item=3,
            max_total_batches=3,
        )
        units = [batch[0] for batch in expanded]

        source_parts = {}
        source_total = 0
        diff_parts_by_index = {}
        diff_total = 0
        rendered_prompts = []
        for unit in units:
            source_override = unit.get("_current_source_override") or ""
            source_match = re.match(
                r"\[Lossless Stage 1 source slice (\d+)/(\d+) [^\n]*\]\n",
                source_override,
            )
            if source_match:
                source_index, source_total = map(int, source_match.groups())
                source_parts[source_index] = source_override[source_match.end():]
            diff_override = unit.get("_diff_override") or ""
            diff_match = re.match(
                r"\[Lossless Stage 1 diff slice (\d+)/(\d+) [^\n]*\]\n",
                diff_override,
            )
            if diff_match:
                diff_index, diff_total = map(int, diff_match.groups())
                diff_parts_by_index[diff_index] = diff_override[diff_match.end():]

            prompt, _ = _render_stage1_prompt(
                _prepare_stage1_prompt_material(
                    request,
                    [unit],
                    prepared,
                    False,
                ),
                "",
            )
            rendered_prompts.append(prompt)
            assert _estimated_prompt_tokens(prompt) <= token_budget
            assert _prepare_stage1_prompt_material(
                request,
                [unit],
                prepared,
                False,
            ).batch_file_paths == [path]

        assert processed.files[0].skip_reason.startswith("File too large:")
        assert "[CodeCrow Summary:" in processed.files[0].content
        assert 1 <= len(units) <= 3
        assert diff_total == 1
        assert len(diff_parts_by_index) == 1
        packed_source = "\n".join(
            unit.get("_current_source_override") or "" for unit in units
        )
        assert "SOURCE_LINE_" in packed_source
        assert source not in packed_source
        packed_diff = "".join(
            diff_parts_by_index[index]
            for index in sorted(diff_parts_by_index)
        )
        assert "[CodeCrow Summary:" in packed_diff
        assert diff_sentinels[0] in packed_diff
        assert diff_sentinels[-1] not in packed_diff
        assert len(units) <= source_total + diff_total
        assert len(units) == max(source_total, diff_total)
        assert all(
            "CodeCrow bounded diff compaction" in prompt
            for prompt in rendered_prompts
        )

        expected_hunks = tuple(sorted(hunk.id for hunk in processed.files[0].hunks))
        assert expected_hunks
        omitted_hunks = tuple(units[-1]["_omitted_hunk_ids"])
        admitted_hunks = tuple(sorted({
            hunk_id for unit in units for hunk_id in unit["_hunk_ids"]
        }))
        assert units[-1]["_omitted_stage1_unit_count"] >= 1
        assert admitted_hunks == ()
        assert omitted_hunks == expected_hunks
        state = Stage1ReviewUnitState()
        state.register_batches(expanded)
        unit_ids = tuple(unit["_review_unit_id"] for unit in units)
        state.mark_completed(unit_ids)
        state.assert_complete()
        assert state.reviewed_hunk_ids == ()
        assert tuple(sorted(state.omitted_hunk_ids)) == omitted_hunks

    def test_compacted_added_file_uses_bounded_source_and_omits_hunk(self):
        path = "src/new-large.py"
        source_lines = [
            f"ADDED_SOURCE_{index:05d} = '{'a' * 64}'"
            for index in range(2_000)
        ]
        source = "\n".join(source_lines) + "\n"
        diff = (
            f"diff --git a/{path} b/{path}\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            f"+++ b/{path}\n"
            f"@@ -0,0 +1,{len(source_lines)} @@\n"
            + "".join(f"+{line}\n" for line in source_lines)
        )
        processed = DiffProcessor().process(diff)
        enrichment = MagicMock(
            fileContents=[MagicMock(path=path, content=source, skipped=False)],
            fileMetadata=[],
            relationships=[],
        )
        request = _packing_request([path], enrichment)
        prepared = _build_stage_1_prepared_context(request, processed, False)
        item = {
            "file": ReviewFile(
                path=path,
                focus_areas=["general"],
                risk_level="MEDIUM",
            ),
            "priority": "MEDIUM",
        }

        expanded = _expand_oversized_stage1_evidence_batches(
            [[item]],
            request,
            prepared,
            False,
            token_budget=8_000,
            max_units_per_item=3,
            max_total_batches=3,
        )

        assert 1 <= len(expanded) <= 3
        assert processed.files[0].skip_reason.startswith("File too large:")
        assert "[CodeCrow Summary:" in processed.files[0].content
        packed_source = "\n".join(
            batch[0].get("_current_source_override") or ""
            for batch in expanded
        )
        packed_diff = "\n".join(
            batch[0]["_diff_override"] for batch in expanded
        )
        assert source_lines[0] in packed_source
        assert source_lines[-1] not in packed_source
        assert source_lines[0] in packed_diff
        assert source_lines[-1] not in packed_diff
        assert expanded[-1][0]["_omitted_stage1_unit_count"] > 0
        expected_hunks = tuple(
            sorted(hunk.id for hunk in processed.files[0].hunks)
        )
        assert expected_hunks
        assert all(
            not (batch[0].get("_hunk_ids") or ()) for batch in expanded
        )
        assert tuple(expanded[-1][0]["_omitted_hunk_ids"]) == expected_hunks
        assert all(
            "CodeCrow bounded diff compaction" in _render_stage1_prompt(
                _prepare_stage1_prompt_material(
                    request,
                    batch,
                    prepared,
                    False,
                ),
                "",
            )[0]
            for batch in expanded
        )

    def test_realistic_architecture_source_bundles_by_exact_path(self):
        owner_path = "src/owner.py"
        related_path = "config/routes.xml"
        relation_text = "ROUTE_RELATION_AUTHORITY checkout configured-by RouteTarget"
        source_lines = [
            f"ROUTE_SOURCE_{index:04d} " + "x" * 180
            for index in range(500)
        ]
        source_text = "\n".join(source_lines) + "\n"
        relation = {
            "text": relation_text,
            "metadata": {
                "path": "__analysis_architecture__/routes.context",
                "architecture_key": "route-relation:routes.xml:0",
                "architecture_paths": [owner_path, related_path],
                "plugin_graph_facts": [{
                    "kind": "route-relation",
                    "source": "Owner",
                    "relation": "configured-by",
                    "target": "RouteTarget",
                    "path": related_path,
                    "related_paths": [owner_path],
                }],
                "pr": True,
            },
            "_match_type": "architecture_relation",
            "_matched_on": owner_path,
            "_source": "pr_indexed",
        }
        # This is the production deterministic-RAG source shape: exact path and
        # provenance, deliberately no architecture_key.
        source = {
            "text": source_text,
            "metadata": {
                "path": related_path,
                "structural_record_type": "source_chunk",
                "start_line": 1,
                "end_line": len(source_lines),
            },
            "_match_type": "architecture_related",
            "_matched_on": related_path,
            "_source": "deterministic",
        }
        rag_context = {
            "relevant_code": [relation, source],
            "_metadata": {
                "retrieval_state": "partial",
                "coverage_state": "partial",
                "context_usable": True,
                "partial_reasons": ["global_matching_point_limit"],
            },
        }

        bundles = _stage1_rag_evidence_bundles(rag_context)

        assert bundles == [[relation, source]]

        enrichment = MagicMock(
            fileContents=[
                MagicMock(path=owner_path, content="owner = True\n", skipped=False)
            ],
            fileMetadata=[],
            relationships=[],
        )
        request = _packing_request([owner_path], enrichment)
        prepared = _build_stage_1_prepared_context(request, None, False)
        material = _prepare_stage1_prompt_material(
            request,
            self._batch([owner_path]),
            prepared,
            False,
        )

        invocations = _build_stage1_rag_invocations(
            material,
            rag_context,
            token_budget=12_000,
        )

        assert len(invocations) == 1
        rag_texts = [invocation[1] for invocation in invocations]
        assert all(relation_text in text for text in rag_texts)
        assert all("Structural retrieval coverage: PARTIAL" in text for text in rag_texts)
        assert all("absence from it is not negative evidence" in text for text in rag_texts)
        combined = "\n".join(rag_texts)
        observed_source_lines = [
            line for line in combined.splitlines()
            if line.startswith("ROUTE_SOURCE_")
        ]
        assert observed_source_lines
        assert len(observed_source_lines) < len(source_lines)
        assert "Context chunk truncated by deterministic prompt budget" in combined
        assert _estimated_prompt_tokens(invocations[0][0]) <= 12_000

    @pytest.mark.asyncio(loop_scope="function")
    async def test_huge_local_and_rag_evidence_form_one_linear_sequence(self):
        path = "src/linear.py"
        related_path = "src/dependency.py"
        local_source = "".join(
            f"LOCAL_SLICE_{index:05d} = '{'l' * 72}'\n"
            for index in range(8_000)
        )
        diff = (
            f"diff --git a/{path} b/{path}\n"
            f"--- a/{path}\n"
            f"+++ b/{path}\n"
            "@@ -1,1 +1,1 @@\n"
            "-old_value = 0\n"
            "+new_value = 1\n"
        )
        processed = DiffProcessor().process(diff)
        enrichment = MagicMock(
            fileContents=[MagicMock(path=path, content=local_source, skipped=False)],
            fileMetadata=[],
            relationships=[],
        )
        request = _packing_request([path], enrichment)
        prepared = _build_stage_1_prepared_context(request, processed, False)
        item = {
            "file": ReviewFile(
                path=path,
                focus_areas=["general"],
                risk_level="MEDIUM",
            ),
            "priority": "MEDIUM",
        }
        token_budget = 60_000
        local_batches = _expand_oversized_stage1_evidence_batches(
            [[item]],
            request,
            prepared,
            False,
            token_budget,
        )
        assert len(local_batches) == 1

        relation_text = "LINEAR_RELATION_AUTHORITY owner imports dependency"
        rag_lines = [
            f"RAG_SLICE_{index:05d} = '{'r' * 72}'"
            for index in range(8_000)
        ]
        rag_source = "\n".join(rag_lines) + "\n"
        rag_context = {"relevant_code": [{
            "text": relation_text,
            "metadata": {
                "path": "__analysis_architecture__/linear.context",
                "architecture_key": "linear-relation:0",
                "architecture_paths": [path, related_path],
                "pr": True,
            },
            "_match_type": "architecture_relation",
            "_matched_on": path,
            "_source": "pr_indexed",
        }, {
            "text": rag_source,
            "metadata": {
                "path": related_path,
                "structural_record_type": "source_chunk",
            },
            "_match_type": "architecture_related",
            "_matched_on": related_path,
            "_source": "deterministic",
        }]}
        owner_material = _prepare_stage1_prompt_material(
            request,
            local_batches[0],
            prepared,
            False,
        )
        rag_invocations = _build_stage1_rag_invocations(
            owner_material,
            rag_context,
            token_budget,
        )
        assert len(rag_invocations) == 1

        class CapturingLlm:
            def __init__(self):
                self.prompts = []

            def with_structured_output(self, _schema):
                return self

            async def ainvoke(self, prompt, **_kwargs):
                self.prompts.append(prompt)
                issue = CodeReviewIssue(
                    id=f"issue-{len(self.prompts)}",
                    severity="MEDIUM",
                    category="BUG",
                    file=path,
                    line=1,
                    title="Synthetic provenance candidate",
                    reason="Synthetic issue used only to verify prompt provenance.",
                    suggestedFixDescription="Synthetic fix.",
                )
                return FileReviewBatchOutput(reviews=[FileReviewOutput(
                    file=path,
                    analysis_summary="synthetic",
                    issues=[issue],
                    confidence="HIGH",
                )])

        llm = CapturingLlm()
        ledger = CandidateEvidenceLedger()
        state = Stage1ReviewUnitState()
        state.register_batches(local_batches)
        fetch_mock = AsyncMock(return_value=rag_context)
        all_issues = []
        with patch(
            "service.review.orchestrator.stage_1_file_review.fetch_batch_rag_context",
            fetch_mock,
        ):
            for batch_number, batch in enumerate(local_batches, start=1):
                all_issues.extend(await review_file_batch(
                    llm,
                    request,
                    batch,
                    rag_client=object(),
                    prepared_context=prepared,
                    candidate_ledger=ledger,
                ))
                state.mark_completed(
                    state.unit_ids_for_batch(batch_number, batch)
                )

        state.assert_complete()
        expected_calls = 1
        assert len(llm.prompts) == expected_calls
        assert fetch_mock.await_count == 1
        assert len(llm.prompts) <= 1

        prompt_corpus = "\n".join(llm.prompts)
        observed_rag_lines = [
            line for line in prompt_corpus.splitlines()
            if line.startswith("RAG_SLICE_")
        ]
        assert observed_rag_lines
        assert len(observed_rag_lines) < len(rag_lines)
        assert prompt_corpus.count(relation_text) == 1
        assert "Context chunk truncated by deterministic prompt budget" in prompt_corpus

        records = list(ledger._records.values())
        assert len(records) == len(all_issues) == expected_calls
        rag_records = [record for record in records if record.visible_evidence_by_id]
        assert len(rag_records) == 1
        owner_unit_id = local_batches[0][0]["_review_unit_id"]
        assert all(record.review_unit_ids == (owner_unit_id,) for record in rag_records)

    def test_rag_evidence_chunks_are_bounded_without_extra_invocations(self):
        path = "src/owner.py"
        enrichment = MagicMock(
            fileContents=[MagicMock(path=path, content="owner = True\n", skipped=False)],
            fileMetadata=[],
            relationships=[],
        )
        request = _packing_request([path], enrichment)
        prepared = _build_stage_1_prepared_context(request, None, False)
        material = _prepare_stage1_prompt_material(
            request, self._batch([path]), prepared, False
        )
        evidence_texts = [
            "def dependency_a():\n" + ("    return 1\n" * 5_000),
            "def dependency_b():\n" + ("    return 2\n" * 5_000),
        ]
        rag_context = {"relevant_code": [
            {
                "text": text,
                "metadata": {"path": f"src/dependency_{index}.py"},
                "_match_type": "definition",
            }
            for index, text in enumerate(evidence_texts, start=1)
        ]}

        invocations = _build_stage1_rag_invocations(
            material,
            rag_context,
            token_budget=24_000,
        )

        assert len(invocations) == 1
        prompts = [invocation[0] for invocation in invocations]
        assert all(_estimated_prompt_tokens(prompt) <= 24_000 for prompt in prompts)
        assert all(
            text not in prompts[0]
            for text in evidence_texts
        )
        assert "def dependency_a():" in prompts[0]
        assert "def dependency_b():" in prompts[0]
        assert prompts[0].count(
            "Context chunk truncated by deterministic prompt budget"
        ) == 2
        assert all("owner = True" in prompt for prompt in prompts)


# ── create_smart_batches_wrapper ─────────────────────────────────

class TestCreateSmartBatchesWrapper:
    def _make_plan(self, paths):
        files = [ReviewFile(path=p, focus_areas=[], risk_level="MEDIUM") for p in paths]
        return [FileGroup(group_id="g0", priority="HIGH", rationale="test", files=files)]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_fallback_when_no_processed_diff(self):
        groups = self._make_plan(["a.py", "b.py"])
        result = await create_smart_batches_wrapper(
            file_groups=groups,
            processed_diff=None,
            request=MagicMock(),
            rag_client=None,
        )
        assert len(result) >= 1
        # Each item is a dict with 'file' key
        for batch in result:
            for item in batch:
                assert "file" in item

    @pytest.mark.asyncio(loop_scope="function")
    async def test_single_file(self):
        groups = self._make_plan(["a.py"])
        result = await create_smart_batches_wrapper(
            file_groups=groups,
            processed_diff=None,
            request=MagicMock(),
            rag_client=None,
        )
        assert len(result) == 1
        assert len(result[0]) == 1

    @patch("service.review.orchestrator.stage_1_file_review.create_smart_batches_async")
    @pytest.mark.asyncio(loop_scope="function")
    async def test_uses_smart_batches_when_available(self, mock_smart):
        mock_smart.return_value = None  # Force fallback
        groups = self._make_plan(["a.py", "b.py"])
        result = await create_smart_batches_wrapper(
            file_groups=groups,
            processed_diff=MagicMock(),
            request=MagicMock(enrichmentData=None),
            rag_client=None,
        )
        # Should still return valid batches from fallback
        assert len(result) >= 1

    @patch("service.review.orchestrator.stage_1_file_review.create_smart_batches_async")
    @pytest.mark.asyncio(loop_scope="function")
    async def test_caps_stage_1_batch_token_budget_for_latency(self, mock_smart):
        groups = self._make_plan(["a.py", "b.py"])
        mock_smart.return_value = [[{"file": groups[0].files[0], "priority": "MEDIUM"}]]
        request = MagicMock(
            enrichmentData=None,
            maxAllowedTokens=200000,
            projectWorkspace="ws",
            projectNamespace="proj",
        )
        request.get_rag_branch.return_value = "feature"
        request.get_rag_base_branch.return_value = "main"

        result = await create_smart_batches_wrapper(
            file_groups=groups,
            processed_diff=MagicMock(),
            request=request,
            rag_client=None,
        )

        assert result == mock_smart.return_value
        assert mock_smart.call_args.kwargs["max_allowed_tokens"] == 60000

    @patch("service.review.orchestrator.stage_1_file_review.create_smart_batches_async")
    @pytest.mark.asyncio(loop_scope="function")
    async def test_missing_target_branch_uses_local_grouping_without_rag(self, mock_smart):
        groups = self._make_plan(["a.py"])
        mock_smart.return_value = [[{"file": groups[0].files[0], "priority": "MEDIUM"}]]
        request = MagicMock(
            enrichmentData=None,
            maxAllowedTokens=200000,
            projectWorkspace="ws",
            projectNamespace="proj",
        )
        request.get_rag_branch.return_value = None
        request.get_rag_base_branch.return_value = None

        result = await create_smart_batches_wrapper(
            file_groups=groups,
            processed_diff=MagicMock(),
            request=request,
            rag_client=MagicMock(),
        )

        assert result == mock_smart.return_value
        assert mock_smart.call_args.kwargs["branches"] == []
        assert mock_smart.call_args.kwargs["rag_client"] is None

    @patch("service.review.orchestrator.stage_1_file_review.create_smart_batches_async")
    @pytest.mark.asyncio(loop_scope="function")
    async def test_exact_receipts_disable_unbound_batching_rag(self, mock_smart):
        groups = self._make_plan(["a.py"])
        mock_smart.return_value = [[{
            "file": groups[0].files[0],
            "priority": "MEDIUM",
        }]]
        request = MagicMock(
            enrichmentData=None,
            maxAllowedTokens=200000,
            projectWorkspace="ws",
            projectNamespace="proj",
            ragBaseGenerationManifestSha256="a" * 64,
            ragPrGenerationFingerprint="sha256:" + "b" * 64,
            ragPrOverlayGenerationManifestSha256="c" * 64,
        )
        request.get_rag_branch.return_value = "main"
        request.get_rag_base_branch.return_value = "main"

        result = await create_smart_batches_wrapper(
            file_groups=groups,
            processed_diff=MagicMock(),
            request=request,
            rag_client=MagicMock(),
        )

        assert result == mock_smart.return_value
        assert mock_smart.call_args.kwargs["rag_client"] is None


class TestStage1Scheduling:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_batches_run_with_bounded_concurrency(self):
        files = [ReviewFile(path=f"src/f{i}.py", focus_areas=[], risk_level="MEDIUM") for i in range(3)]
        batches = [[{"file": f, "priority": "MEDIUM"}] for f in files]
        request = MagicMock()
        request.deltaDiff = None
        request.rawDiff = ""
        request.taskContext = None
        request.enrichmentData = None
        request.changedFiles = [f.path for f in files]

        async def fake_batches(**kwargs):
            return batches

        async def fake_review(*args, **kwargs):
            await asyncio.sleep(0.05)
            return []

        with patch(
            "service.review.orchestrator.stage_1_file_review.create_smart_batches_wrapper",
            side_effect=fake_batches,
        ), patch(
            "service.review.orchestrator.stage_1_file_review.review_file_batch",
            side_effect=fake_review,
        ):
            started = time.perf_counter()
            issues = await execute_stage_1_file_reviews(
                llm=MagicMock(),
                request=request,
                plan=ReviewPlan(analysis_summary="x", file_groups=[], cross_file_concerns=[]),
                rag_client=None,
                max_parallel=3,
            )
            elapsed = time.perf_counter() - started

        assert issues == []
        assert elapsed < 0.12

    @pytest.mark.asyncio(loop_scope="function")
    async def test_reverse_completion_keeps_batch_order_and_completes_units(self):
        files = [
            ReviewFile(
                path=f"src/f{i}.py",
                focus_areas=[],
                risk_level="MEDIUM",
            )
            for i in range(3)
        ]
        batches = [[{"file": file, "priority": "MEDIUM"}] for file in files]
        request = MagicMock(
            deltaDiff=None,
            rawDiff="",
            taskContext=None,
            enrichmentData=None,
            changedFiles=[file.path for file in files],
        )
        state = Stage1ReviewUnitState()

        async def fake_batches(**kwargs):
            return batches

        async def fake_review(_llm, _request, batch, *_args, **_kwargs):
            index = int(batch[0]["file"].path.removesuffix(".py")[-1])
            await asyncio.sleep((2 - index) * 0.02)
            return [batch[0]["file"].path]

        with patch(
            "service.review.orchestrator.stage_1_file_review.create_smart_batches_wrapper",
            side_effect=fake_batches,
        ), patch(
            "service.review.orchestrator.stage_1_file_review.review_file_batch",
            side_effect=fake_review,
        ):
            issues = await execute_stage_1_file_reviews(
                llm=MagicMock(),
                request=request,
                plan=ReviewPlan(
                    analysis_summary="x",
                    file_groups=[],
                    cross_file_concerns=[],
                ),
                rag_client=None,
                max_parallel=3,
                review_unit_state=state,
            )

        assert issues == [file.path for file in files]
        state.assert_complete()
        assert len(state.completed_unit_ids) == 3

    @pytest.mark.asyncio(loop_scope="function")
    async def test_any_failed_batch_fails_the_whole_stage(self):
        files = [ReviewFile(path=f"src/f{i}.py", focus_areas=[], risk_level="MEDIUM") for i in range(2)]
        batches = [[{"file": file, "priority": "MEDIUM"}] for file in files]
        request = MagicMock(
            deltaDiff=None,
            rawDiff="",
            taskContext=None,
            enrichmentData=None,
            changedFiles=[file.path for file in files],
        )

        async def fake_batches(**kwargs):
            return batches

        async def fake_review(_llm, _request, batch, *_args, **_kwargs):
            if batch[0]["file"].path.endswith("f0.py"):
                raise RuntimeError("provider timeout")
            await asyncio.sleep(0.1)
            return []

        with patch(
            "service.review.orchestrator.stage_1_file_review.create_smart_batches_wrapper",
            side_effect=fake_batches,
        ), patch(
            "service.review.orchestrator.stage_1_file_review.review_file_batch",
            side_effect=fake_review,
        ):
            with pytest.raises(RuntimeError, match="Stage 1 review is incomplete"):
                await execute_stage_1_file_reviews(
                    llm=MagicMock(),
                    request=request,
                    plan=ReviewPlan(analysis_summary="x", file_groups=[], cross_file_concerns=[]),
                    rag_client=None,
                    max_parallel=2,
                )
