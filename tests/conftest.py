"""Root conftest — shared fixtures and session-wide patches.

Most tests pass raw sentinel bytes like ``b"IMG_A"`` through the pipeline to
exercise ids / persistence / ops semantics without building valid images.
Two things in the production pipeline would choke on those bytes:

1. ``crop_image`` (PIL-backed) — monkeypatched to a passthrough here.
2. The :meth:`Encoder.embed_frame` dispatch — stub encoders in tests don't
   implement ``embed_frame``, so ``hasattr(encoder, "embed_frame")`` is False
   and the pipeline falls back to ``embed(crop)``, which is what we want. No
   patch needed.

The *real* crop and *real* embed_frame paths are covered by
``tests/encoders/test_crop.py`` and ``tests/encoders/test_insightface.py``
respectively.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _bypass_crop_in_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace crop_image in the pipeline module with a passthrough.

    Tests that need real cropping import ``crop_image`` directly from
    ``vemem.encoders.crop`` and are unaffected.
    """
    monkeypatch.setattr(
        "vemem.pipeline.crop_image",
        lambda image_bytes, bbox: image_bytes,
    )
