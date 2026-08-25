"""Provider-safe reasoning controls for individual model invocations."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from utils.llm_delegate import llm_class_names, unwrap_llm_delegate


class ReasoningEffort(str, Enum):
    """OpenRouter's normalized reasoning-effort levels used by review calls."""

    NONE = "none"
    LOW = "low"
    HIGH = "high"


def reasoning_request_kwargs(llm: Any, effort: ReasoningEffort) -> dict[str, Any]:
    """Return per-invocation OpenRouter reasoning parameters.

    LangChain's ``reasoning`` model field selects the Responses API. CodeCrow's
    OpenRouter integration uses Chat Completions, so the normalized OpenRouter
    object must travel through ``extra_body`` on the individual invocation.
    Other providers are left untouched.
    """
    if "ChatOpenRouter" not in llm_class_names(llm):
        return {}

    delegate = unwrap_llm_delegate(llm)
    configured_body = getattr(delegate, "extra_body", None)
    extra_body = (
        dict(configured_body)
        if isinstance(configured_body, Mapping)
        else {}
    )
    # The semantic call site owns effort. Replace a configured reasoning object
    # so incompatible settings such as reasoning.max_tokens cannot travel next
    # to reasoning.effort. Other OpenRouter fields remain intact.
    extra_body["reasoning"] = {"effort": effort.value}
    return {"extra_body": extra_body}
