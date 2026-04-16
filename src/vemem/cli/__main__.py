"""Entry point for ``python -m vemem.cli``.

Mirrors the ``vm`` console script so operators without a ``pip install``
on the shell path can still invoke the CLI through the Python module system.
Keeps the body tiny — everything interesting lives in :mod:`vemem.cli.app`.
"""

from __future__ import annotations

from vemem.cli.app import app, main

# Re-export so tests can assert `python -m vemem.cli` exposes the same Typer app.
__all__ = ["app", "main"]

if __name__ == "__main__":  # pragma: no cover - exercised by subprocess invocation
    raise SystemExit(main())
