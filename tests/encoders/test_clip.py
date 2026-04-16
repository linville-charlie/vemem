"""Smoke tests for CLIPEncoder.

CLIP is experimental in v0 — these tests verify the adapter is structurally
correct (satisfies Encoder Protocol, produces the advertised dim,
L2-normalized) and nothing more. We do NOT tune similarity thresholds around
it.
"""

from __future__ import annotations

import inspect
import io
from pathlib import Path
from typing import get_type_hints

import pytest
from PIL import Image

from vemem.core.protocols import Encoder
from vemem.encoders.clip_encoder import CLIPEncoder

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_clip_has_embed_signature_bytes_to_tuple() -> None:
    sig = inspect.signature(CLIPEncoder.embed)
    params = list(sig.parameters.values())
    assert len(params) == 2
    assert params[1].name == "image_crop"

    hints = get_type_hints(CLIPEncoder.embed)
    assert hints["image_crop"] is bytes
    assert str(hints["return"]).startswith("tuple[float")


def test_clip_init_sets_id_with_version_in_string() -> None:
    src = inspect.getsource(CLIPEncoder.__init__)
    assert "self.id =" in src
    assert "open_clip/" in src


def test_clip_docstring_marks_experimental() -> None:
    assert "EXPERIMENTAL" in (CLIPEncoder.__doc__ or "")


# ---------------------------------------------------------------------------
# Integration tests — gated behind VEMEM_RUN_INTEGRATION=1
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clip_encoder() -> CLIPEncoder:
    return CLIPEncoder()


def _make_test_image_bytes(size: tuple[int, int] = (224, 224)) -> bytes:
    img = Image.new("RGB", size, (128, 64, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.integration
def test_clip_satisfies_encoder_protocol(clip_encoder: CLIPEncoder) -> None:
    assert isinstance(clip_encoder, Encoder)
    assert clip_encoder.id.startswith("open_clip/")
    assert clip_encoder.dim > 0


@pytest.mark.integration
def test_clip_embed_returns_tuple_of_advertised_dim(clip_encoder: CLIPEncoder) -> None:
    vec = clip_encoder.embed(_make_test_image_bytes())

    assert isinstance(vec, tuple)
    assert len(vec) == clip_encoder.dim
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.integration
def test_clip_output_is_l2_normalized(clip_encoder: CLIPEncoder) -> None:
    vec = clip_encoder.embed(_make_test_image_bytes())
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4


@pytest.mark.integration
def test_clip_is_deterministic_for_same_input(clip_encoder: CLIPEncoder) -> None:
    img = _make_test_image_bytes()
    v1 = clip_encoder.embed(img)
    v2 = clip_encoder.embed(img)
    # small floating-point drift on CPU is fine; assert near-identity
    diff = max(abs(a - b) for a, b in zip(v1, v2, strict=True))
    assert diff < 1e-5


@pytest.mark.integration
def test_clip_handles_real_fixture_if_present(
    clip_encoder: CLIPEncoder, fixtures_dir: Path
) -> None:
    path = fixtures_dir / "person_a_front.jpg"
    if not path.exists():
        pytest.skip("no fixture image")

    vec = clip_encoder.embed(path.read_bytes())
    assert len(vec) == clip_encoder.dim
