"""MCP client wrapper for invoking external MCP servers via stdio.

This module provides a thin synchronous wrapper around the asynchronous
``mcp`` SDK so that MCP-based tools can be plugged into the existing
``BaseTool`` interface without rewriting the agentic loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


@dataclass
class McpServerParams:
    """Parameters describing how to launch an MCP server process."""

    command: str
    args: list[str]
    env: dict[str, str] | None = None


class McpClient:
    """Lightweight MCP client that starts a server process and calls a tool.

    Each invocation spins up a fresh stdio server process. This keeps the
    implementation simple and avoids lifecycle issues, at the cost of a small
    per-call overhead. For high-traffic deployments, consider maintaining a
    persistent session pool.
    """

    def __init__(self, server_params: McpServerParams):
        self.server_params = server_params

    async def _call_tool_async(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Call a tool on the MCP server asynchronously."""
        env = os.environ.copy()
        if self.server_params.env:
            env.update(self.server_params.env)

        params = StdioServerParameters(
            command=self.server_params.command,
            args=self.server_params.args,
            env=env,
        )

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)

                # Flatten tool result content into a single text block.
                parts: list[str] = []
                for content in result.content:
                    if hasattr(content, "text"):
                        parts.append(content.text)
                    else:
                        parts.append(str(content))
                return "\n".join(parts) if parts else "(no content)"

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool synchronously.

        Handles both "no event loop" and "already inside an event loop" cases
        so that the tool can be used from sync scripts as well as from async
        servers like FastAPI.
        """
        coro = self._call_tool_async(tool_name, arguments)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running: safe to use asyncio.run.
            return asyncio.run(coro)

        # Already inside an event loop (e.g. FastAPI). Submit the coroutine to
        # the loop and wait for the result. This blocks the current task, which
        # is acceptable because the agentic loop currently calls tools from a
        # synchronous context.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
