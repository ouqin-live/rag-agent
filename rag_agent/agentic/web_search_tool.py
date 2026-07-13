"""Web search tool backed by an MCP server.

The default implementation uses the official Brave Search MCP server:
https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search
"""

from __future__ import annotations

import logging
import os

from rag_agent.agentic.base import BaseTool
from rag_agent.agentic.mcp_client import McpClient, McpServerParams

logger = logging.getLogger(__name__)


class WebSearchMcpTool(BaseTool):
    """Search the web via an MCP search server.

    Requires a running Node/npx installation and a Brave API key for the
    default Brave Search MCP server. The key can be provided via the
    ``BRAVE_API_KEY`` environment variable or passed directly.
    """

    name = "web_search"

    def __init__(
        self,
        api_key: str | None = None,
        server_command: str = "npx",
        server_package: str = "@modelcontextprotocol/server-brave-search",
    ):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not self.api_key:
            logger.warning(
                "WebSearchMcpTool initialized without BRAVE_API_KEY. "
                "Web search calls will fail until the key is configured."
            )

        params = McpServerParams(
            command=server_command,
            args=["-y", server_package],
            env={"BRAVE_API_KEY": self.api_key or ""},
        )
        self.client = McpClient(params)

    def invoke(self, query: str) -> str:
        """Search the web for the given query and return result snippets."""
        if not self.api_key:
            return "Web search is not configured: missing BRAVE_API_KEY."

        try:
            result = self.client.call_tool(
                "brave_web_search",
                {"query": query, "count": 5},
            )
            return result
        except Exception as exc:
            logger.warning("Web search MCP call failed: %s", exc)
            return f"Web search failed: {exc}"
