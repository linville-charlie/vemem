"""Deterministic stub encoder + detector used by the MCP server test mode.

Lives outside ``conftest.py`` because ``src/vemem/mcp_server/server.py``
imports it directly in ``_build_test_context`` (so it runs in the subprocess
spawned by the integration test, which does NOT inherit pytest fixtures).

The stubs are hash-based: same image bytes → same bbox + same embedding. That
gives the integration test reproducible `identify_image` results without
needing any real vision model.
"""

from __future__ import annotations

import hashlib
import math


class StubEncoder:
    """4-d stub encoder — ``hash(image_bytes) → unit vector``.

    Useful properties for tests: same image → same vector; different images
    → near-orthogonal vectors (the 4-d projection of the SHA-256 has enough
    entropy to keep distinct images apart in cosine space).
    """

    id: str = "test-stub/encoder@0"
    dim: int = 4

    def embed(self, image_crop: bytes) -> tuple[float, ...]:
        digest = hashlib.sha256(image_crop).digest()
        # Turn the first 4 bytes into a float vector in [-1, 1].
        raw = [(b - 128) / 128.0 for b in digest[:4]]
        norm = math.sqrt(sum(x * x for x in raw))
        if norm == 0.0:
            return (1.0, 0.0, 0.0, 0.0)
        return tuple(x / norm for x in raw)


class StubDetector:
    """Single-face stub detector — one 100x100 bbox at (10, 10) per image."""

    id: str = "test-stub/detector@0"

    def detect(self, image_bytes: bytes) -> list[tuple[int, int, int, int]]:
        if not image_bytes:
            return []
        return [(10, 10, 100, 100)]
