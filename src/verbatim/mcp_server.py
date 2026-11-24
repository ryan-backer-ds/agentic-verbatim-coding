"""The analyst tools as an MCP server, so any MCP client can use them.

A desktop assistant, an IDE, or another team's agent can call the same read-only
`theme_stats` and `search_answers` tools as the LangGraph and ADK agents:

    verbatim mcp            # serves over stdio
"""

from functools import cache
from typing import Any

from mcp.server.mcpserver import MCPServer

from verbatim.analysis import AnalystToolkit, GroupBy
from verbatim.data import SegmentFilter

server = MCPServer(
    name="verbatim",
    instructions=(
        "Read-only tools over coded NHS Friends and Family Test feedback. "
        "Use theme_stats for numbers and search_answers for quotes."
    ),
)


@server.tool()
def theme_stats(
    segment: SegmentFilter, group_by: GroupBy | None = None, top_n: int = 8
) -> dict[str, Any]:
    """Most common themes in a segment of survey answers, with shares and 95% intervals."""
    return _toolkit().theme_stats(segment, group_by=group_by, top_n=min(top_n, 20))


@server.tool()
def search_answers(query: str, segment: SegmentFilter, k: int = 8) -> list[dict[str, Any]]:
    """Survey answers that best match a topic, to quote as evidence (at most 20)."""
    return _toolkit().search_answers(query, segment, k=k)


@cache
def _toolkit() -> AnalystToolkit:
    return AnalystToolkit.load()


def serve() -> None:
    server.run("stdio")
