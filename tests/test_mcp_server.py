import asyncio

from verbatim.mcp_server import server


def test_the_mcp_server_exposes_the_analyst_tools():
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}

    assert set(tools) == {"theme_stats", "search_answers"}
    assert "segment" in tools["search_answers"].input_schema["properties"]
