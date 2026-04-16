"""CLI runtime context — store + encoder + detector + clock + actor.

Mirrors the pattern in ``vemem.mcp_server.config``: each invocation builds a
context that owns the heavy components, and the Typer command handlers use it
without caring how the components were wired. Shared concerns:

- ``VEMEM_HOME`` environment variable (or ``--home`` flag) selects the store
  path; the default is ``~/.vemem``.
- InsightFace is loaded best-effort. If weights are missing the context still
  builds — commands that don't need an encoder (``label``, ``recall``,
  ``remember``, ``merge``, ``undo``, …) keep working, and image-dependent
  commands print the install hint and exit with code 2 on invocation.
- Actor defaults to ``f"cli:{getpass.getuser()}"`` but is overridable via the
  ``--actor`` global flag so scripted pipelines can attribute writes.

Tests install a pre-built :class:`CliContext` via :func:`set_test_context` so
the Typer CliRunner path never touches LanceDB or InsightFace. This is the
same escape hatch the MCP server exposes via ``VEMEM_MCP_TEST_MODE``.
"""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


class _HasNow(Protocol):
    def now(self) -> datetime: ...


class UTCClock:
    """Real-time clock producing UTC-aware timestamps."""

    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass
class CliContext:
    """Shared dependencies passed to every CLI command handler.

    ``encoder`` and ``detector`` are ``None`` when weights are missing;
    ``encoder_error`` carries the failure message so image-dependent commands
    surface an actionable error. ``actor`` is already formatted as
    ``"cli:{user}"`` (or whatever the ``--actor`` flag supplied).
    """

    store: Any
    clock: _HasNow
    encoder: Any | None
    detector: Any | None
    encoder_error: str | None
    actor: str


# Process-global override for tests. Using a module-level singleton (rather
# than a thread-local) because Typer's CliRunner runs synchronously in the
# test's main thread and the CLI itself is single-threaded.
_test_context: CliContext | None = None


def set_test_context(ctx: CliContext | None) -> None:
    """Install (or clear) a pre-built context for in-process CLI tests."""
    global _test_context
    _test_context = ctx


def default_actor() -> str:
    """Return the default actor string: ``cli:{local-username}``.

    Falls back to ``"cli:anonymous"`` when ``getpass.getuser`` fails (it can
    raise inside some sandboxed environments).
    """
    try:
        return f"cli:{getpass.getuser()}"
    except Exception:
        return "cli:anonymous"


def resolve_store_path(home: Path | None) -> Path:
    """Resolve the storage path from ``--home`` → ``VEMEM_HOME`` → ``~/.vemem``."""
    if home is not None:
        return home
    override = os.environ.get("VEMEM_HOME")
    if override:
        return Path(override)
    return Path.home() / ".vemem"


def build_cli_context(
    *,
    home: Path | None = None,
    actor: str | None = None,
) -> CliContext:
    """Construct a :class:`CliContext` for a CLI invocation.

    If a test context has been installed via :func:`set_test_context`, it is
    returned unchanged (with ``actor`` overridden when supplied). Otherwise
    this opens a LanceDB store at the resolved path and attempts to load
    InsightFace; encoder-load failures are captured in ``encoder_error`` so
    image-dependent commands can report them verbatim.
    """
    resolved_actor = actor or default_actor()

    if _test_context is not None:
        # Tests installed a pre-built context — honor actor overrides but
        # keep everything else intact.
        if actor is not None:
            from dataclasses import replace

            return replace(_test_context, actor=resolved_actor)
        return _test_context

    from vemem.storage.lancedb_store import LanceDBStore

    store = LanceDBStore(path=resolve_store_path(home))
    status = _load_encoder_status()
    return CliContext(
        store=store,
        clock=UTCClock(),
        encoder=status["encoder"],
        detector=status["detector"],
        encoder_error=status["error"],
        actor=resolved_actor,
    )


def _load_encoder_status() -> dict[str, Any]:
    """Best-effort load of InsightFace.

    Returns a dict with ``encoder``, ``detector``, and ``error`` keys so the
    CLI layer can inspect presence without importing the encoder modules at
    the callsite. Mirrors :class:`vemem.mcp_server.config.EncoderStatus` but
    avoids cross-package coupling — the MCP and CLI layers share pattern, not
    code, to keep dependency directions clean.
    """
    name = os.environ.get("VEMEM_ENCODER", "insightface").lower()

    if name != "insightface":
        return {
            "encoder": None,
            "detector": None,
            "error": f"unsupported VEMEM_ENCODER={name!r}; only 'insightface' is wired in v0",
        }

    try:
        # Lazy import so a fresh install without weights still works.
        from vemem.encoders.insightface_detector import InsightFaceDetector
        from vemem.encoders.insightface_encoder import InsightFaceEncoder

        return {
            "encoder": InsightFaceEncoder(),
            "detector": InsightFaceDetector(),
            "error": None,
        }
    except Exception as exc:
        return {
            "encoder": None,
            "detector": None,
            "error": (
                f"insightface encoder could not load: {exc}. "
                "On first use InsightFace downloads ~200MB of weights to "
                "~/.insightface/models/. Ensure network access or pre-populate "
                "that directory. Try: "
                "uv run python -c 'from insightface.app import FaceAnalysis; "
                'FaceAnalysis(name="buffalo_l").prepare(ctx_id=-1)\''
            ),
        }
