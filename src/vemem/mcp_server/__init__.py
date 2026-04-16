"""MCP server exposing the core operations over stdio.

Wraps the core ops as MCP tools so any MCP-capable client (Claude Desktop,
Cursor, custom agents) can query the same local store the Python library
accesses directly.
"""
