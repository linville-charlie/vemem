"""Tests for InsightFaceEncoder and InsightFaceDetector.

Two tiers:

- **Unit tests** always run. They inspect the adapter classes structurally
  (class-level attributes, method signatures, Protocol satisfaction via
  constructed instances only when possible). They do NOT load model weights.
- **Integration tests** marked ``@pytest.mark.integration`` and skipped
  unless ``VEMEM_RUN_INTEGRATION=1`` is set. These load the ``buffalo_l``
  model pack (downloaded to ``~/.insightface`` on first use) and verify
  real-image similarity on fixture photos.
"""

from __future__ import annotations

import hashlib
import inspect
import io
from pathlib import Path
from typing import get_type_hints

import pytest
from PIL import Image

from vemem.core.protocols import Detector, Encoder
from vemem.encoders.insightface_detector import InsightFaceDetector
from vemem.encoders.insightface_encoder import InsightFaceEncoder

# ---------------------------------------------------------------------------
# Unit tests — no model weights required
# ---------------------------------------------------------------------------


def test_encoder_advertises_dim_512_at_class_level() -> None:
    assert InsightFaceEncoder.dim == 512


def test_encoder_has_embed_signature_bytes_to_tuple() -> None:
    sig = inspect.signature(InsightFaceEncoder.embed)
    params = list(sig.parameters.values())
    # self, image_crop
    assert len(params) == 2
    assert params[1].name == "image_crop"

    hints = get_type_hints(InsightFaceEncoder.embed)
    assert hints["image_crop"] is bytes
    # return is tuple[float, ...]
    assert str(hints["return"]).startswith("tuple[float")


def test_detector_has_detect_signature_bytes_to_list_of_tuples() -> None:
    sig = inspect.signature(InsightFaceDetector.detect)
    params = list(sig.parameters.values())
    assert len(params) == 2
    assert params[1].name == "image_bytes"

    hints = get_type_hints(InsightFaceDetector.detect)
    assert hints["image_bytes"] is bytes
    assert "tuple[int, int, int, int]" in str(hints["return"])


def test_encoder_surface_exposes_id_slot_via_instance_protocol() -> None:
    # The Encoder Protocol requires id/dim/embed. We can't construct the real
    # encoder without downloading ~200MB of weights, so we verify the class
    # sets ``id`` in __init__ (source inspection) and exposes ``dim`` at class
    # level. runtime_checkable only checks attribute names on instances.
    src = inspect.getsource(InsightFaceEncoder.__init__)
    assert "self.id =" in src
    assert "insightface/arcface@" in src


def test_detector_surface_exposes_id_via_init() -> None:
    src = inspect.getsource(InsightFaceDetector.__init__)
    assert "self.id =" in src
    assert "insightface/" in src


# ---------------------------------------------------------------------------
# Integration tests — gated behind VEMEM_RUN_INTEGRATION=1
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def encoder() -> InsightFaceEncoder:
    return InsightFaceEncoder()


@pytest.fixture(scope="module")
def detector() -> InsightFaceDetector:
    return InsightFaceDetector()


def _read_fixture(fixtures_dir: Path, name: str) -> bytes:
    path = fixtures_dir / name
    if not path.exists():
        pytest.skip(f"fixture {name} not present")
    return path.read_bytes()


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    # vectors are L2-normalized → cosine == dot product
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.mark.integration
def test_encoder_satisfies_encoder_protocol(encoder: InsightFaceEncoder) -> None:
    assert isinstance(encoder, Encoder)
    assert encoder.dim == 512
    assert encoder.id.startswith("insightface/arcface@")


@pytest.mark.integration
def test_detector_satisfies_detector_protocol(detector: InsightFaceDetector) -> None:
    assert isinstance(detector, Detector)
    assert detector.id.startswith("insightface/buffalo_l@")


@pytest.mark.integration
def test_encoder_output_is_512d_and_l2_normalized(
    encoder: InsightFaceEncoder, fixtures_dir: Path
) -> None:
    img = _read_fixture(fixtures_dir, "person_a_front.jpg")

    vec = encoder.embed(img)

    assert len(vec) == 512
    assert all(isinstance(x, float) for x in vec)
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4


@pytest.mark.integration
def test_same_person_different_angles_has_high_similarity(
    encoder: InsightFaceEncoder, fixtures_dir: Path
) -> None:
    a1 = encoder.embed(_read_fixture(fixtures_dir, "person_a_front.jpg"))
    a2 = encoder.embed(_read_fixture(fixtures_dir, "person_a_alt.jpg"))

    # Same person should be well above ArcFace's common threshold (~0.35).
    assert _cosine(a1, a2) > 0.35


@pytest.mark.integration
def test_different_people_have_low_similarity(
    encoder: InsightFaceEncoder, fixtures_dir: Path
) -> None:
    a = encoder.embed(_read_fixture(fixtures_dir, "person_a_front.jpg"))
    b = encoder.embed(_read_fixture(fixtures_dir, "person_b_front.jpg"))

    # Different people should be comfortably below the same threshold.
    assert _cosine(a, b) < 0.35


@pytest.mark.integration
def test_detector_returns_at_least_one_bbox_for_face_image(
    detector: InsightFaceDetector, fixtures_dir: Path
) -> None:
    bboxes = detector.detect(_read_fixture(fixtures_dir, "person_a_front.jpg"))

    assert len(bboxes) >= 1
    x, y, w, h = bboxes[0]
    assert w > 0 and h > 0
    assert x >= 0 and y >= 0


@pytest.mark.integration
def test_encoder_raises_when_no_face_detected(encoder: InsightFaceEncoder) -> None:
    # plain white 200x200 PNG — no face to detect
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), (255, 255, 255)).save(buf, format="PNG")

    with pytest.raises(RuntimeError, match="no face detected"):
        encoder.embed(buf.getvalue())


@pytest.mark.integration
def test_encoder_id_is_stable_within_process(encoder: InsightFaceEncoder) -> None:
    # Two instances should produce the same id (version-pinned).
    other = InsightFaceEncoder()
    assert encoder.id == other.id
    # Also sanity-check the id is hashable (used as a dict key in storage).
    hashlib.sha256(encoder.id.encode()).hexdigest()
