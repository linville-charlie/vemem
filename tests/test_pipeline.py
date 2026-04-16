"""Tests for ``vemem.pipeline.observe_image`` — the shared image-ingest recipe."""

from __future__ import annotations

from datetime import UTC, datetime

from tests.support.fake_store import FakeStore
from vemem.core.enums import Modality
from vemem.pipeline import observe_image


class _FixedClock:
    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at


class _StubDetector:
    id = "stub/detector@1"

    def __init__(self, bboxes: list[tuple[int, int, int, int]]) -> None:
        self._bboxes = bboxes

    def detect(self, image_bytes: bytes) -> list[tuple[int, int, int, int]]:
        return self._bboxes


class _StubEncoder:
    id = "stub/encoder@1"
    dim = 4

    def embed(self, image_bytes: bytes) -> tuple[float, ...]:
        # deterministic tiny vector derived from length
        n = len(image_bytes)
        return (float(n % 7), float((n >> 3) % 11), 1.0, 0.0)


T = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def test_observe_image_returns_one_observation_per_bbox() -> None:
    store = FakeStore()
    detector = _StubDetector([(0, 0, 50, 50), (60, 0, 50, 50)])
    encoder = _StubEncoder()

    result = observe_image(
        store,
        image_bytes=b"FAKE_JPEG_BYTES",
        detector=detector,
        encoder=encoder,
        clock=_FixedClock(T),
    )

    assert len(result) == 2
    for obs in result:
        assert store.get_observation(obs.id) is obs


def test_observe_image_is_idempotent_for_same_bytes_and_bbox() -> None:
    store = FakeStore()
    detector = _StubDetector([(0, 0, 50, 50)])
    encoder = _StubEncoder()
    clock = _FixedClock(T)

    first = observe_image(store, image_bytes=b"X", detector=detector, encoder=encoder, clock=clock)
    second = observe_image(store, image_bytes=b"X", detector=detector, encoder=encoder, clock=clock)

    assert first[0].id == second[0].id


def test_observe_image_defaults_source_uri_to_hash() -> None:
    store = FakeStore()
    result = observe_image(
        store,
        image_bytes=b"Y",
        detector=_StubDetector([(0, 0, 1, 1)]),
        encoder=_StubEncoder(),
        clock=_FixedClock(T),
    )
    assert result[0].source_uri.startswith("hash:")


def test_observe_image_uses_passed_source_uri() -> None:
    store = FakeStore()
    result = observe_image(
        store,
        image_bytes=b"Z",
        detector=_StubDetector([(0, 0, 1, 1)]),
        encoder=_StubEncoder(),
        clock=_FixedClock(T),
        source_uri="file:///tmp/a.jpg",
        modality=Modality.FACE,
    )
    assert result[0].source_uri == "file:///tmp/a.jpg"


def test_observe_image_empty_on_no_detections() -> None:
    store = FakeStore()
    result = observe_image(
        store,
        image_bytes=b"W",
        detector=_StubDetector([]),
        encoder=_StubEncoder(),
        clock=_FixedClock(T),
    )
    assert result == []
