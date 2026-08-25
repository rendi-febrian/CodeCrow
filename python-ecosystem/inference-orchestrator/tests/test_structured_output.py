from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from llm.reasoning_policy import ReasoningEffort
from service.review.orchestrator.json_utils import resolve_structured_output
from service.review.orchestrator.structured_output import (
    StructuredOutputInvocation,
    invoke_structured_output,
    response_diagnostics,
)


class _Payload(BaseModel):
    value: str


class ChatOpenRouter:
    def __init__(self, response, *, model_name, extra_body=None):
        self.response = response
        self.model_name = model_name
        self.extra_body = extra_body
        self.binding = None
        self.calls = []

    def with_structured_output(
        self,
        schema,
        *,
        include_raw=False,
        method="json_schema",
    ):
        self.binding = {
            "schema": schema,
            "include_raw": include_raw,
            "method": method,
        }
        return self

    async def ainvoke(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response


@pytest.mark.asyncio(loop_scope="function")
async def test_deepseek_openrouter_uses_function_calling_and_compatible_route():
    parsed = _Payload(value="ok")
    llm = ChatOpenRouter(
        {"raw": None, "parsed": parsed, "parsing_error": None},
        model_name="~DeepSeek/DeepSeek-V4-Flash-0731:nitro",
        extra_body={"provider": {"order": ["DeepInfra"]}},
    )

    invocation = await invoke_structured_output(
        llm,
        "prompt",
        _Payload,
        effort=ReasoningEffort.HIGH,
        label="test",
    )

    assert invocation.parsed == parsed
    assert invocation.method == "function_calling"
    assert invocation.raw_included is True
    assert llm.binding == {
        "schema": _Payload,
        "include_raw": True,
        "method": "function_calling",
    }
    assert llm.calls[0][1] == {
        "extra_body": {
            "provider": {
                "order": ["DeepInfra"],
                "require_parameters": True,
            },
            "reasoning": {"effort": "high"},
        }
    }


@pytest.mark.asyncio(loop_scope="function")
async def test_other_openrouter_models_keep_json_schema_transport():
    llm = ChatOpenRouter(
        _Payload(value="ok"),
        model_name="other/model",
    )

    invocation = await invoke_structured_output(
        llm,
        "prompt",
        _Payload,
        effort=ReasoningEffort.LOW,
        label="test",
    )

    assert invocation.method == "json_schema"
    assert llm.binding["method"] == "json_schema"


@pytest.mark.asyncio(loop_scope="function")
async def test_one_argument_legacy_delegate_remains_compatible_without_output_cap():
    class ChatOpenRouter:
        model_name = "deepseek/deepseek-v4-flash-0731"

        def __init__(self):
            self.calls = []

        def with_structured_output(self, schema):
            self.schema = schema
            return self

        async def ainvoke(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return _Payload(value="legacy")

    delegate = ChatOpenRouter()

    invocation = await invoke_structured_output(
        delegate,
        "prompt",
        _Payload,
        effort=ReasoningEffort.NONE,
        label="legacy",
    )

    assert invocation.parsed == _Payload(value="legacy")
    assert invocation.method is None
    assert invocation.raw_included is False
    assert delegate.schema is _Payload
    assert "max_tokens" not in delegate.calls[0][1]
    assert "max_completion_tokens" not in delegate.calls[0][1]
    assert delegate.calls[0][1]["extra_body"]["provider"] == {
        "require_parameters": True,
    }


@pytest.mark.asyncio(loop_scope="function")
async def test_raw_tool_arguments_are_recovered_without_another_provider_call():
    raw = SimpleNamespace(
        content="",
        tool_calls=[{"args": {"value": "from-tool"}}],
    )
    invocation = StructuredOutputInvocation(
        parsed=None,
        raw=raw,
        parsing_error=ValueError("initial validation failed"),
        method="function_calling",
        raw_included=True,
    )

    result = await resolve_structured_output(
        invocation,
        _Payload,
        SimpleNamespace(),
    )

    assert result == _Payload(value="from-tool")


@pytest.mark.asyncio(loop_scope="function")
async def test_direct_mapping_result_is_normalized_through_schema():
    result = await resolve_structured_output(
        StructuredOutputInvocation(parsed={"value": "mapping"}),
        _Payload,
        SimpleNamespace(),
    )

    assert result == _Payload(value="mapping")


def test_response_diagnostics_are_content_free_and_provider_neutral():
    response = SimpleNamespace(
        content=None,
        tool_calls=[],
        response_metadata={
            "stop_reason": "max_tokens",
            "token_usage": {
                "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": 40},
            },
        },
    )

    assert response_diagnostics(response) == {
        "content_chars": 0,
        "finish_reason": "max_tokens",
        "output_tokens": 50,
        "reasoning_tokens": 40,
        "tool_calls": 0,
    }
