"""Programmatic export of the OpenAI tool schemas (Wave 3E).

Three public surfaces:

- :func:`all_tools` — returns the canonical list of schemas (a fresh copy per
  call) for programmatic use.
- :func:`write_tools_json` — writes the canonical JSON to a given path. Used
  by the snapshot test and by operators who want to drop a pre-generated
  ``tools.json`` into their agent framework.
- :func:`main` / ``python -m vemem.tools.export`` — CLI entry point. Prints
  the JSON to stdout by default; ``--output PATH`` writes to a file.

The JSON format is pretty-printed (2-space indent, sorted-key-free to preserve
the builders' declaration order) so diffs against the committed snapshot stay
readable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from vemem.tools.schemas import build_all_schemas

_INDENT = 2


def all_tools() -> list[dict[str, Any]]:
    """Return every OpenAI function-calling tool schema for the vemem ops."""
    return build_all_schemas()


def _to_json(tools: list[dict[str, Any]]) -> str:
    # Trailing newline keeps POSIX-friendly files and keeps the snapshot stable
    # after editors that auto-append newlines on save.
    return json.dumps(tools, indent=_INDENT) + "\n"


def write_tools_json(path: str | Path) -> Path:
    """Write the canonical tool JSON to ``path`` and return the written path.

    Creates parent directories if missing. Overwrites any existing file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_to_json(all_tools()))
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vemem.tools.export",
        description=(
            "Emit the OpenAI-compatible function-calling JSON schemas for every "
            "vemem operation. Writes to stdout by default, or to --output PATH."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write to PATH instead of stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — return the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = _to_json(all_tools())
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    raise SystemExit(main())
