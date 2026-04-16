"""MCP server exposing the core operations over stdio.

Wraps the core ops as MCP tools so any MCP-capable client (Claude Desktop,
Cursor, custom agents) can query the same local store the Python library
accesses directly.

Public surface:

- :func:`create_server` — build a FastMCP app with a given context.
- :func:`build_context` — load store + encoder/detector from the environment.
- :func:`run` — blocking entry point used by ``python -m vemem.mcp_server``.
"""

from vemem.mcp_server.server import build_context, create_server, run

__all__ = ["build_context", "create_server", "run"]
