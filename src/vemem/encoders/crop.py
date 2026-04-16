"""Minimal PIL-backed image cropping helper.

Encoders consume crop bytes, detectors return bboxes — this tiny helper bridges
the two without pulling PIL into the ``core`` module. Kept local to the
``encoders`` layer so callers can skip it and hand crop bytes in directly.
"""

from __future__ import annotations

import io

from PIL import Image


def crop_image(image_bytes: bytes, bbox: tuple[int, int, int, int]) -> bytes:
    """Crop ``image_bytes`` to the region ``(x, y, w, h)`` and return JPEG bytes.

    The bbox is clamped to the image's own bounds — a detector that returns a
    region slightly overflowing the frame (common on edge detections) still
    produces a valid crop rather than raising. A zero-area bbox after clamping
    raises ``ValueError``.

    Output format is JPEG (quality 95) in RGB mode — small, lossy-but-adequate
    for a recognition encoder, and matches what InsightFace expects if the
    caller hands it back downstream.
    """

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError(f"bbox has non-positive size: {bbox}")

    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")
        img_w, img_h = rgb.size

        left = max(0, x)
        top = max(0, y)
        right = min(img_w, x + w)
        bottom = min(img_h, y + h)

        if right <= left or bottom <= top:
            raise ValueError(f"bbox {bbox} is entirely outside image size {(img_w, img_h)}")

        cropped = rgb.crop((left, top, right, bottom))

        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
