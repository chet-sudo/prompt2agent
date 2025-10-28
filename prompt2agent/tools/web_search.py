"""Web search tool implementation using DuckDuckGo instant answers."""
from __future__ import annotations

import httpx

from prompt2agent.tools.base import tool_definition
from prompt2agent.utils.logging import get_logger

logger = get_logger(__name__)


@tool_definition(
    name="web_search",
    description="Search the web for a query using DuckDuckGo's instant answer API.",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    output_schema={
        "type": "object",
        "properties": {
            "heading": {"type": "string"},
            "abstract": {"type": "string"},
            "related_topics": {"type": "array"},
        },
    },
)
async def web_search(*, query: str) -> dict[str, object]:
    """Perform a web search and return structured results."""
    logger.debug("Executing web search for query: %s", query)
    params = {"q": query, "format": "json", "no_html": 1}
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get("https://api.duckduckgo.com/", params=params)
        response.raise_for_status()
        data = response.json()
    logger.debug("Web search response: %s", data)
    result = {
        "heading": data.get("Heading"),
        "abstract": data.get("AbstractText"),
        "related_topics": data.get("RelatedTopics", []),
    }
    return result
