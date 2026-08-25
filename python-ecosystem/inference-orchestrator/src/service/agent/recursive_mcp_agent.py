"""MCP agent variant that applies the requested graph recursion limit."""

import logging
from typing import Any

from mcp_use import MCPAgent

from service.agent.models import reject_mcp_use_model_call_limit_output


logger = logging.getLogger(__name__)


# Avoid duplicate mcp_use log propagation when several prompt agents share a
# client. Keep the existing log format used by the inference orchestrator.
mcp_logger = logging.getLogger("mcp_use")
mcp_logger.propagate = False
if not mcp_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    mcp_logger.addHandler(handler)


class RecursiveMCPAgent(MCPAgent):
    """Apply a caller-selected recursion limit to the internal agent graph."""

    def __init__(self, *args: Any, recursion_limit: int = 50, **kwargs: Any):
        self._custom_recursion_limit = recursion_limit
        super().__init__(*args, **kwargs)

    async def _attempt_structured_output(
            self,
            raw_result: str,
            *args: Any,
            **kwargs: Any,
    ):
        # mcp-use treats LangChain's artificial model-call-limit message as a
        # normal final AI message and otherwise sends it through another model
        # call for schema conversion. Reject it before it can become fabricated
        # structured output.
        reject_mcp_use_model_call_limit_output(raw_result)
        return await super()._attempt_structured_output(
            raw_result,
            *args,
            **kwargs,
        )

    async def stream(self, *args: Any, **kwargs: Any):
        if self._agent_executor is None:
            await self.initialize()

        executor = self._agent_executor
        if executor and not getattr(executor, "_is_patched_recursion", False):
            original_astream = executor.astream
            limit = self._custom_recursion_limit

            async def patched_astream(
                    input_data: Any,
                    config: dict[str, Any] | None = None,
                    **astream_kwargs: Any,
            ):
                graph_config = dict(config or {})
                graph_config["recursion_limit"] = limit
                async for chunk in original_astream(
                    input_data,
                    config=graph_config,
                    **astream_kwargs,
                ):
                    yield chunk

            executor.astream = patched_astream
            executor._is_patched_recursion = True
            logger.info(
                "RecursiveMCPAgent: patched recursion limit to %s",
                limit,
            )

        async for item in super().stream(*args, **kwargs):
            reject_mcp_use_model_call_limit_output(item)
            yield item
