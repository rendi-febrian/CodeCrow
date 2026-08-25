"""Focused tests for the shared MCP agent execution lifecycle."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from service.agent import (
    AgentExecutionRequest,
    AgentExecutionService,
    AgentModelCallLimitError,
    AgentOutputEvent,
    AgentToolEvent,
    RecursiveMCPAgent,
)


class _Output(BaseModel):
    value: str


class _FakeSession:
    def __init__(self, *tool_names: str):
        self._tools = [SimpleNamespace(name=name) for name in tool_names]
        self.list_tools_calls = 0
        self.initialize_calls = 0

    async def initialize(self):
        self.initialize_calls += 1

    async def list_tools(self):
        self.list_tools_calls += 1
        return self._tools


class _FakeClient:
    def __init__(self, sessions=None):
        self.sessions = dict(sessions or {})
        self.create_calls = 0
        self.close_all_sessions = AsyncMock()

    def get_all_active_sessions(self):
        return self.sessions

    async def create_all_sessions(self):
        self.create_calls += 1
        await asyncio.sleep(0)
        self.sessions = {
            "vcs": _FakeSession("allowedTool", "blockedTool"),
        }
        return self.sessions


class _FakeAgent:
    stream_items = []
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.stream_call = None
        type(self).instances.append(self)

    async def stream(self, prompt, **kwargs):
        self.stream_call = (prompt, kwargs)
        await asyncio.sleep(0)
        for item in type(self).stream_items:
            yield item


@pytest.fixture(autouse=True)
def _reset_fake_agent():
    _FakeAgent.instances = []
    _FakeAgent.stream_items = []


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_filters_tools_and_collects_the_stream_without_closing_client():
    client = _FakeClient()
    action = SimpleNamespace(tool="allowedTool", tool_input={"path": "a.py"})
    output = _Output(value="done")
    _FakeAgent.stream_items = [(action, "file contents"), output]
    service = AgentExecutionService(llm=object(), client=client)
    request = AgentExecutionRequest(
        prompt="review this",
        allowed_tool_names=frozenset({"allowedTool", "unknownTool"}),
        max_steps=7,
        output_schema=_Output,
        additional_instructions="Return structured output.",
        metadata={"batch": 3},
        recursion_limit=61,
    )

    with patch(
        "service.agent.agent_execution_service.RecursiveMCPAgent",
        _FakeAgent,
    ):
        result = await service.execute(request)

    assert client.create_calls == 1
    assert client.close_all_sessions.await_count == 0
    assert service.available_tool_names == frozenset({
        "allowedTool",
        "blockedTool",
    })
    assert result.output is output
    assert len(result.tool_events) == 1
    assert result.tool_events[0].action is action
    assert result.tool_events[0].observation == "file contents"
    assert result.metadata == {"batch": 3}

    agent = _FakeAgent.instances[0]
    assert agent.kwargs["disallowed_tools"] == ["blockedTool"]
    assert agent.kwargs["memory_enabled"] is False
    assert agent.kwargs["max_steps"] == 7
    assert agent.kwargs["recursion_limit"] == 61
    assert agent.kwargs["additional_instructions"] == "Return structured output."
    assert agent.stream_call == (
        "review this",
        {
            "max_steps": 7,
            "manage_connector": False,
            "output_schema": _Output,
        },
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_concurrent_prompts_initialize_once_and_use_fresh_agents():
    client = _FakeClient()
    _FakeAgent.stream_items = ["done"]
    service = AgentExecutionService(llm=object(), client=client)

    with patch(
        "service.agent.agent_execution_service.RecursiveMCPAgent",
        _FakeAgent,
    ):
        results = await asyncio.gather(*(
            service.execute(AgentExecutionRequest(
                prompt=f"prompt-{index}",
                allowed_tool_names=frozenset({"allowedTool"}),
                max_steps=5,
            ))
            for index in range(8)
        ))

    assert client.create_calls == 1
    assert len(_FakeAgent.instances) == 8
    assert len({id(agent) for agent in _FakeAgent.instances}) == 8
    assert all(agent.kwargs["memory_enabled"] is False for agent in _FakeAgent.instances)
    assert [result.output for result in results] == ["done"] * 8


@pytest.mark.asyncio(loop_scope="function")
async def test_stream_exposes_typed_events_and_reuses_active_sessions():
    session = _FakeSession("allowedTool", "otherTool")
    client = _FakeClient({"vcs": session})
    action = SimpleNamespace(tool="allowedTool")
    _FakeAgent.stream_items = [(action, "observation"), "answer"]
    service = AgentExecutionService(llm=object(), client=client)
    request = AgentExecutionRequest(
        prompt="prompt",
        allowed_tool_names=frozenset({"allowedTool"}),
        max_steps=3,
        metadata={"flow": "test"},
    )

    with patch(
        "service.agent.agent_execution_service.RecursiveMCPAgent",
        _FakeAgent,
    ):
        events = [event async for event in service.stream(request)]

    assert client.create_calls == 0
    assert session.list_tools_calls == 1
    assert isinstance(events[0], AgentToolEvent)
    assert isinstance(events[1], AgentOutputEvent)
    assert events[0].metadata == {"flow": "test"}
    assert events[1].output == "answer"


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_rejects_mcp_use_model_call_limit_as_final_output():
    client = _FakeClient()
    _FakeAgent.stream_items = [
        "Model call limits exceeded: run limit (12/12)",
    ]
    service = AgentExecutionService(llm=object(), client=client)

    with (
        patch(
            "service.agent.agent_execution_service.RecursiveMCPAgent",
            _FakeAgent,
        ),
        pytest.raises(
            AgentModelCallLimitError,
            match="before producing a final response",
        ),
    ):
        await service.execute(AgentExecutionRequest(
            prompt="prompt",
            allowed_tool_names=frozenset({"allowedTool"}),
            max_steps=12,
        ))


@pytest.mark.asyncio(loop_scope="function")
async def test_similar_user_text_is_not_treated_as_model_call_limit():
    client = _FakeClient()
    output = "Model call limits exceeded: this is quoted documentation"
    _FakeAgent.stream_items = [output]
    service = AgentExecutionService(llm=object(), client=client)

    with patch(
        "service.agent.agent_execution_service.RecursiveMCPAgent",
        _FakeAgent,
    ):
        result = await service.execute(AgentExecutionRequest(
            prompt="prompt",
            allowed_tool_names=frozenset({"allowedTool"}),
            max_steps=12,
        ))

    assert result.output == output


@pytest.mark.asyncio(loop_scope="function")
async def test_optional_server_failure_keeps_required_server_tools_available():
    class SelectiveClient(_FakeClient):
        def __init__(self):
            super().__init__()
            self.session_calls = []

        async def create_session(self, server_name, auto_initialize=True):
            self.session_calls.append(server_name)
            if server_name == "rag":
                class FailingSession(_FakeSession):
                    async def initialize(self):
                        raise RuntimeError("RAG startup failed")

                session = FailingSession("searchRepositoryCode")
            else:
                session = _FakeSession("getBranchFileContent")
            self.sessions[server_name] = session
            if auto_initialize:
                await session.initialize()
            return session

        async def close_session(self, server_name):
            self.sessions.pop(server_name, None)

    client = SelectiveClient()
    service = AgentExecutionService(llm=object(), client=client)

    optional_errors = await service.initialize(
        required_server_names=("vcs",),
        optional_server_names=("rag",),
        session_timeout_seconds=1,
    )

    assert client.session_calls == ["vcs", "rag"]
    assert set(optional_errors) == {"rag"}
    assert isinstance(optional_errors["rag"], RuntimeError)
    assert "rag" not in client.sessions
    assert service.available_tool_names == frozenset({
        "getBranchFileContent",
    })


def test_review_local_agent_module_remains_a_compatibility_export():
    from service.review.orchestrator.agents import (
        RecursiveMCPAgent as ReviewRecursiveMCPAgent,
    )

    assert ReviewRecursiveMCPAgent is RecursiveMCPAgent
