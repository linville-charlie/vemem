"""Tests for the PIL-backed crop helper."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from vemem.encoders.crop import crop_image


def _make_png(width: int, height: int, color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_crop_returns_jpeg_decodable_by_pil() -> None:
    src = _make_png(100, 80)

    out = crop_image(src, (10, 10, 50, 40))

    decoded = Image.open(io.BytesIO(out))
    assert decoded.size == (50, 40)
    assert decoded.format == "JPEG"


def test_crop_preserves_colors_roughly() -> None:
    # a solid-color source should survive JPEG compression pretty much intact
    src = _make_png(64, 64, color=(12, 200, 70))

    out = crop_image(src, (0, 0, 32, 32))

    decoded = Image.open(io.BytesIO(out)).convert("RGB")
    pixel = decoded.getpixel((16, 16))
    assert isinstance(pixel, tuple) and len(pixel) == 3
    r, g, b = pixel
    # JPEG quantization wiggle — accept close-enough
    assert abs(r - 12) < 8
    assert abs(g - 200) < 8
    assert abs(b - 70) < 8


def test_crop_clamps_bbox_that_overflows_image() -> None:
    src = _make_png(50, 50)

    # bbox extends past right and bottom edges
    out = crop_image(src, (30, 30, 100, 100))

    decoded = Image.open(io.BytesIO(out))
    # clamped to (30,30) → (50,50) = 20x20
    assert decoded.size == (20, 20)


def test_crop_accepts_negative_origin() -> None:
    src = _make_png(50, 50)

    out = crop_image(src, (-10, -10, 30, 30))

    decoded = Image.open(io.BytesIO(out))
    # clamped to (0,0) → (20,20)
    assert decoded.size == (20, 20)


def test_crop_rejects_zero_size_bbox() -> None:
    src = _make_png(50, 50)

    with pytest.raises(ValueError):
        crop_image(src, (10, 10, 0, 10))

    with pytest.raises(ValueError):
        crop_image(src, (10, 10, 10, 0))


def test_crop_rejects_bbox_entirely_outside_image() -> None:
    src = _make_png(50, 50)

    with pytest.raises(ValueError):
        crop_image(src, (100, 100, 10, 10))
