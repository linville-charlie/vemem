"""Server configuration and startup-time component loading.

Resolves the store path, encoder, and detector from environment variables,
with graceful fallbacks so the server starts even when heavy encoder weights
are unavailable. An MCP server that refuses to start on a bare install is a
worse user experience than one that starts and returns structured errors from
the image-dependent tools — the non-image tools (``label``, ``recall``,
``merge``, ``undo``, …) still work.

Environment variables:

- ``VEMEM_HOME`` — directory for the LanceDB dataset. Default: ``~/.vemem``.
- ``VEMEM_ENCODER`` — which encoder to load. Currently only ``insightface``
  (default) is wired; anything else produces a lazy-failure encoder.
- ``VEMEM_MCP_TEST_MODE`` — ``1`` swaps the real store + encoder for the
  in-memory FakeStore and a deterministic stub encoder/detector. Used by the
  integration test to roundtrip JSON-RPC messages without downloading weights.
  NOT intended for production use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class UTCClock:
    """Real-time ``Clock`` producing UTC-aware ``datetime`` values."""

    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass
class EncoderStatus:
    """Summary of whether the configured encoder/detector loaded cleanly.

    ``encoder`` and ``detector`` are ``None`` when weights are missing; the
    server still starts and image tools surface a clear error at call time.
    ``error`` carries the underlying failure so tools can echo it to the
    caller verbatim.
    """

    encoder: Any | None
    detector: Any | None
    error: str | None


def default_store_path() -> Path:
    """Return the configured store directory (``$VEMEM_HOME`` or ``~/.vemem``)."""
    override = os.environ.get("VEMEM_HOME")
    if override:
        return Path(override)
    return Path.home() / ".vemem"


def test_mode_enabled() -> bool:
    """Return True when ``VEMEM_MCP_TEST_MODE`` is set to a truthy value.

    The integration test uses this to swap real backends for in-memory fakes
    without needing network access or model weights.
    """
    return os.environ.get("VEMEM_MCP_TEST_MODE", "").lower() in {"1", "true", "yes"}


def load_encoder_and_detector() -> EncoderStatus:
    """Instantiate the configured encoder and detector, never raising.

    Returns an :class:`EncoderStatus` carrying either the loaded objects or a
    single human-readable error message if loading failed. The caller (the
    image-dependent MCP tools) inspects ``encoder`` / ``detector`` presence
    before running.
    """
    name = os.environ.get("VEMEM_ENCODER", "insightface").lower()

    if name == "insightface":
        try:
            # Imported lazily so a fresh install without weights still starts.
            from vemem.encoders.insightface_detector import InsightFaceDetector
            from vemem.encoders.insightface_encoder import InsightFaceEncoder

            encoder = InsightFaceEncoder()
            detector = InsightFaceDetector()
        except Exception as exc:
            return EncoderStatus(
                encoder=None,
                detector=None,
                error=(
                    f"insightface encoder could not load: {exc}. "
                    "On first use InsightFace downloads ~200MB of weights to "
                    "~/.insightface/models/. Ensure network access or "
                    "pre-populate that directory. Run: "
                    "uv run python -c 'from insightface.app import FaceAnalysis; "
                    'FaceAnalysis(name="buffalo_l").prepare(ctx_id=-1)\''
                ),
            )
        return EncoderStatus(encoder=encoder, detector=detector, error=None)

    return EncoderStatus(
        encoder=None,
        detector=None,
        error=f"unsupported VEMEM_ENCODER={name!r}; only 'insightface' is wired in v0",
    )
