"""Root conftest — shared fixtures and session-wide patches.

The crop-leak fix (v0 review feedback item #2) made pipeline.observe_image
crop each bbox through PIL before encoding. This is correct for production
but breaks ~30 tests that pass raw byte strings like ``b"IMG_A"`` through the
pipeline. Rather than scatter valid-image generation into every test module, we
monkeypatch ``vemem.pipeline.crop_image`` to return the input unchanged for the
test session. The *real* crop path is covered by ``tests/encoders/test_crop.py``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _bypass_crop_in_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace crop_image in the pipeline module with a passthrough.

    Tests that need real cropping (``tests/encoders/test_crop.py``) import
    ``crop_image`` directly from ``vemem.encoders.crop`` and are unaffected.
    """
    monkeypatch.setattr(
        "vemem.pipeline.crop_image",
        lambda image_bytes, bbox: image_bytes,
    )
