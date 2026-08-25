"""Request-scoped MCP adapter for optional repository code search."""

from __future__ import annotations

import json
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from service.rag.rag_client import RagClient


server = FastMCP(
    "CodeCrow repository search",
    instructions=(
        "Search the exact repository generation for source or symbols that are "
        "not already present in the review prompt. Returned results are additional "
        "context; absence from a bounded result set is not negative evidence."
    ),
    log_level="WARNING",
)


def _context(name: str) -> str | None:
    value = os.environ.get(f"CODECROW_RAG_MCP_{name}")
    return value.strip() if value and value.strip() else None


@server.tool(
    name="searchRepositoryCode",
    description=(
        "Search the review's exact repository generation by source text, symbol, "
        "type, or concept. Use this only for context not already supplied."
    ),
    structured_output=True,
)
async def search_repository_code(query: str, top_k: int = 8) -> dict[str, Any]:
    """Return deterministic repository matches for the request-bound generation."""
    client = RagClient()
    try:
        result = await client.search_code(
            query=query,
            workspace=_context("WORKSPACE") or "",
            project=_context("PROJECT") or "",
            branch=_context("BRANCH") or "",
            top_k=max(1, min(int(top_k), 20)),
            repository_revision=_context("REVISION"),
            repository_generation_manifest_sha256=_context("MANIFEST"),
            collection_target=_context("COLLECTION_TARGET"),
        )
        # FastMCP serializes the mapping as structured content. Round-trip the
        # value to ensure provider-specific scalar types cannot leak onto stdio.
        return json.loads(json.dumps(result, ensure_ascii=False, default=str))
    finally:
        await client.close()


if __name__ == "__main__":
    server.run("stdio")
