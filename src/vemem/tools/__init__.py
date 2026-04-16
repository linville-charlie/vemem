"""OpenAI-compatible tool schemas.

Generated JSON schemas for every vemem core op, suitable for function-calling
LLMs that aren't MCP-aware (OpenAI, Anthropic tool use, Gemini, Ollama,
openclaw). The MCP server exposes the same ops over the MCP wire format; these
schemas are the "stateless paste-into-your-prompt" surface that describes the
same contract.

Quick start::

    from vemem.tools import all_tools
    tools = all_tools()  # list[dict] ready for the OpenAI Chat Completions API

Or drop a pre-generated ``tools.json`` alongside your agent config::

    python -m vemem.tools.export > tools.json
"""

from vemem.tools.export import all_tools, write_tools_json
from vemem.tools.schemas import TOOL_NAMES, schema_for

__all__ = ["TOOL_NAMES", "all_tools", "schema_for", "write_tools_json"]
