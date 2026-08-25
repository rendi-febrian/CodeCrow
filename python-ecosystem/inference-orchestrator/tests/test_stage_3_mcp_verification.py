"""Focused boundary tests for the extracted Stage 3 MCP subsystem."""

from types import SimpleNamespace

import pytest

from llm.reasoning_policy import ReasoningEffort
from service.review.orchestrator import stage_3_aggregation
from service.review.orchestrator import stage_3_mcp_verification
from service.review.orchestrator.stage_3_mcp_verification import (
    Stage3McpRuntime,
    execute_stage_3_mcp_verification,
)


def test_aggregation_preserves_legacy_helper_imports() -> None:
    """Existing callers can keep importing the historical private names."""
    assert (
        stage_3_aggregation._extract_dismissed_issues
        is stage_3_mcp_verification.extract_dismissed_issues
    )
    assert (
        stage_3_aggregation._stage_3_verification_issue_map
        is stage_3_mcp_verification.verification_issue_map
    )
    assert (
        stage_3_aggregation._validated_mcp_dismissals
        is stage_3_mcp_verification.validated_mcp_dismissals
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_mcp_subsystem_runs_through_the_typed_runtime_boundary() -> None:
    response = SimpleNamespace(
        content="bounded verification report",
        tool_calls=[],
        response_metadata={},
    )

    class BoundLlm:
        async def ainvoke(self, _messages, **_kwargs):
            return response

    class Llm:
        def bind_tools(self, tool_definitions):
            assert tool_definitions
            return BoundLlm()

    async def unexpected_report_fallback(*_args, **_kwargs):
        raise AssertionError("the plain report fallback should not be used")

    runtime = Stage3McpRuntime(
        input_token_target=lambda _request: 10_000,
        estimate_messages_tokens=lambda _messages, **_kwargs: 1,
        continuation_messages=lambda _prompt, _records: [],
        invoke_report=unexpected_report_fallback,
        response_finished_by_length=lambda _response: False,
    )
    request = SimpleNamespace(
        projectVcsWorkspace="workspace",
        projectVcsRepoSlug="repository",
    )

    result = await execute_stage_3_mcp_verification(
        Llm(),
        request,
        "prompt",
        SimpleNamespace(),
        "commit-sha",
        {},
        runtime,
    )

    assert result == {
        "report": "bounded verification report",
        "dismissed_issue_ids": [],
        "dismissed_issue_keys": [],
        "dismissed_issue_object_ids": [],
    }


@pytest.mark.asyncio(loop_scope="function")
async def test_length_recovery_requests_reasoning_free_plain_report() -> None:
    response = SimpleNamespace(
        content="",
        tool_calls=[],
        response_metadata={"finish_reason": "max_tokens"},
    )

    class BoundLlm:
        async def ainvoke(self, _messages, **_kwargs):
            return response

    class Llm:
        def bind_tools(self, _tool_definitions):
            return BoundLlm()

    captured = {}

    async def invoke_report(
        llm,
        prompt,
        fallback_llm=None,
        allow_retry=True,
        reasoning_effort=ReasoningEffort.LOW,
    ):
        captured.update({
            "llm": llm,
            "prompt": prompt,
            "fallback_llm": fallback_llm,
            "allow_retry": allow_retry,
            "reasoning_effort": reasoning_effort,
        })
        return {"report": "recovered report"}

    runtime = Stage3McpRuntime(
        input_token_target=lambda _request: 10_000,
        estimate_messages_tokens=lambda _messages, **_kwargs: 1,
        continuation_messages=lambda _prompt, _records: [],
        invoke_report=invoke_report,
        response_finished_by_length=lambda candidate: (
            candidate.response_metadata.get("finish_reason") == "max_tokens"
        ),
    )
    request = SimpleNamespace(
        projectVcsWorkspace="workspace",
        projectVcsRepoSlug="repository",
    )
    llm = Llm()

    result = await execute_stage_3_mcp_verification(
        llm,
        request,
        "prompt",
        SimpleNamespace(),
        "commit-sha",
        {},
        runtime,
        fallback_llm=llm,
    )

    assert result == {"report": "recovered report"}
    assert captured == {
        "llm": llm,
        "prompt": "prompt",
        "fallback_llm": None,
        "allow_retry": False,
        "reasoning_effort": ReasoningEffort.NONE,
    }
