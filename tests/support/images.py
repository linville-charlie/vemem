"""Tiny valid PNG generator for tests that go through pipeline.observe_image.

After the crop-leak fix, pipeline.observe_image crops each bbox via PIL before
passing to the encoder. Tests that used raw byte strings like b"IMG_A" now fail
because PIL can't decode them. This helper generates deterministic, decodable
100x100 PNGs keyed by a seed integer — different seeds produce different pixel
content, so encoders that hash the input produce different vectors.
"""

from __future__ import annotations

import io

from PIL import Image


def make_test_image(seed: int = 0, width: int = 100, height: int = 100) -> bytes:
    r = (seed * 37 + 50) % 256
    g = (seed * 53 + 100) % 256
    b = (seed * 71 + 150) % 256
    img = Image.new("RGB", (width, height), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
