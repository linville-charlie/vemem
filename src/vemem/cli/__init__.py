"""Command-line interface for the vemem library.

Exposes the ``vm`` console script (see ``pyproject.toml [project.scripts]``)
plus the ``python -m vemem.cli`` entry point. Subcommands wrap the core ops
in :mod:`vemem.core.ops`, read/write a LanceDB store at ``$VEMEM_HOME``, and
render results via Rich tables or ``--format json``.
"""

from __future__ import annotations

from vemem.cli.app import app, main

__all__ = ["app", "main"]
