"""Shared construction and execution lifecycle for prompt-scoped MCP agents."""

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Generic

from mcp_use import MCPClient

from service.agent.models import (
    AgentExecutionEvent,
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentOutputEvent,
    AgentOutputT,
    AgentToolEvent,
    reject_mcp_use_model_call_limit_output,
)
from service.agent.recursive_mcp_agent import RecursiveMCPAgent


class AgentExecutionService(Generic[AgentOutputT]):
    """
    Run independent prompt agents over one caller-owned MCP client.

    The service initializes the client's shared sessions once, but deliberately
    does not close them. The command/review job that created the client remains
    responsible for its lifecycle.
    """

    def __init__(self, *, llm: Any, client: MCPClient):
        self._llm = llm
        self._client = client
        self._initialization_lock = asyncio.Lock()
        self._initialized = False
        self._available_tool_names: frozenset[str] = frozenset()

    @property
    def available_tool_names(self) -> frozenset[str]:
        return self._available_tool_names

    async def initialize(
            self,
            *,
            required_server_names: Sequence[str] | None = None,
            optional_server_names: Sequence[str] = (),
            session_timeout_seconds: float | None = None,
    ) -> Mapping[str, Exception]:
        """Create shared MCP sessions and inventory their tools exactly once.

        ``required_server_names`` lets a caller initialize a known-good core
        server before optional enrichment servers. Optional startup errors are
        returned to the caller so they can be reported without disabling the
        sessions that did start.
        """
        if self._initialized:
            return {}

        async with self._initialization_lock:
            if self._initialized:
                return {}

            sessions = self._client.get_all_active_sessions()
            optional_errors: dict[str, Exception] = {}
            if required_server_names is None and not sessions:
                sessions = await self._client.create_all_sessions()
            elif required_server_names is not None:
                for server_name in required_server_names:
                    if server_name in sessions:
                        continue
                    await self._create_session(
                        server_name,
                        session_timeout_seconds,
                    )
                    sessions = self._client.get_all_active_sessions()

                for server_name in optional_server_names:
                    if server_name in sessions:
                        continue
                    try:
                        await self._create_session(
                            server_name,
                            session_timeout_seconds,
                        )
                    except Exception as exception:
                        optional_errors[server_name] = exception
                    sessions = self._client.get_all_active_sessions()

            tool_names: set[str] = set()
            optional_names = set(optional_server_names)
            for server_name, session in list(sessions.items()):
                try:
                    tools = await self._await_with_timeout(
                        session.list_tools(),
                        session_timeout_seconds,
                    )
                except Exception as exception:
                    if server_name not in optional_names:
                        raise
                    optional_errors[server_name] = exception
                    await self._close_optional_session(
                        server_name,
                        session_timeout_seconds,
                    )
                    continue
                tool_names.update(
                    tool.name
                    for tool in tools
                    if isinstance(getattr(tool, "name", None), str)
                )

            self._available_tool_names = frozenset(tool_names)
            self._initialized = True
            return optional_errors

    async def _create_session(
            self,
            server_name: str,
            timeout_seconds: float | None,
    ) -> Any:
        # Register the connector with the client before initialization so a
        # timeout or cancellation still has a concrete session to disconnect.
        session = await self._client.create_session(
            server_name,
            auto_initialize=False,
        )
        if session is None:
            return None
        try:
            await self._await_with_timeout(
                session.initialize(),
                timeout_seconds,
            )
            return session
        except BaseException:
            await self._close_optional_session(
                server_name,
                timeout_seconds,
            )
            raise

    @staticmethod
    async def _await_with_timeout(
            awaitable: Any,
            timeout_seconds: float | None,
    ) -> Any:
        if timeout_seconds is None:
            return await awaitable
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)

    async def _close_optional_session(
            self,
            server_name: str,
            timeout_seconds: float | None,
    ) -> None:
        try:
            await self._await_with_timeout(
                self._client.close_session(server_name),
                timeout_seconds,
            )
        except Exception:
            # The startup error remains the useful diagnostic. Client teardown
            # at the owning request boundary gets another chance to close it.
            pass

    async def stream(
            self,
            request: AgentExecutionRequest[AgentOutputT],
    ) -> AsyncIterator[AgentExecutionEvent[AgentOutputT]]:
        """Stream typed tool/output events from a fresh prompt-scoped agent."""
        await self.initialize()

        allowed_tool_names = set(request.allowed_tool_names)
        disallowed_tools = sorted(
            self._available_tool_names.difference(allowed_tool_names)
        )
        metadata: Mapping[str, Any] = dict(request.metadata)

        agent = RecursiveMCPAgent(
            llm=self._llm,
            client=self._client,
            max_steps=request.max_steps,
            recursion_limit=request.recursion_limit,
            additional_instructions=request.additional_instructions,
            disallowed_tools=disallowed_tools,
            memory_enabled=False,
        )

        async for item in agent.stream(
            request.prompt,
            max_steps=request.max_steps,
            manage_connector=False,
            output_schema=request.output_schema,
        ):
            # Keep the shared execution boundary safe even if a future agent
            # implementation bypasses RecursiveMCPAgent's upstream workaround.
            reject_mcp_use_model_call_limit_output(item)
            if isinstance(item, tuple) and len(item) == 2:
                action, observation = item
                yield AgentToolEvent(
                    action=action,
                    observation=observation,
                    metadata=metadata,
                )
            else:
                yield AgentOutputEvent(output=item, metadata=metadata)

    async def execute(
            self,
            request: AgentExecutionRequest[AgentOutputT],
    ) -> AgentExecutionResult[AgentOutputT]:
        """Collect ``stream`` without introducing a second execution path."""
        final_output: Any = None
        tool_events: list[AgentToolEvent] = []

        async for event in self.stream(request):
            if isinstance(event, AgentToolEvent):
                tool_events.append(event)
            else:
                final_output = event.output

        return AgentExecutionResult(
            output=final_output,
            tool_events=tuple(tool_events),
            metadata=dict(request.metadata),
        )
