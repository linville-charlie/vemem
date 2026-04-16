"""Entry point for ``python -m vemem.mcp_server``.

Delegates to :func:`vemem.mcp_server.server.run`. Kept tiny so the operator
can reason about what this does without reading a second module.
"""

from __future__ import annotations

from vemem.mcp_server.server import run

if __name__ == "__main__":
    run()
