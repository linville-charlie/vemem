"""Tests for :class:`vemem.Vemem` — the top-level convenience wrapper."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.support.fake_store import FakeStore
from vemem import Source, Vemem


class _FixedClock:
    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at


class _StubDetector:
    id = "stub/detector@1"

    def detect(self, image_bytes: bytes) -> list[tuple[int, int, int, int]]:
        return [(0, 0, 50, 50)]


class _StubEncoder:
    id = "stub/encoder@1"
    dim = 4

    def embed(self, image_bytes: bytes) -> tuple[float, ...]:
        # trivial deterministic embedding keyed by first byte
        b = image_bytes[0] if image_bytes else 0
        return (float(b), 1.0, 0.0, 0.0)


T = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def _build() -> Vemem:
    return Vemem(
        store=FakeStore(),
        encoder=_StubEncoder(),
        detector=_StubDetector(),
        clock=_FixedClock(T),
        actor="test:unit",
    )


# ---------- construction ----------


def test_vemem_accepts_injected_pieces() -> None:
    vem = _build()
    assert vem.actor == "test:unit"
    assert vem.store is not None


def test_vemem_is_a_context_manager() -> None:
    with _build() as vem:
        assert vem.actor == "test:unit"


# ---------- image path ----------


def test_vemem_observe_persists_observations() -> None:
    vem = _build()

    observations = vem.observe(b"IMG_A")

    assert len(observations) == 1
    assert vem.store.get_observation(observations[0].id) is not None


def test_vemem_identify_by_image_bytes_after_label() -> None:
    vem = _build()
    observations = vem.observe(b"IMG_A")
    entity = vem.label([o.id for o in observations], name="Charlie")

    candidates = vem.identify(b"IMG_A", k=5)

    assert len(candidates) == 1
    assert candidates[0].entity.id == entity.id


def test_vemem_identify_by_precomputed_vector_requires_encoder_id() -> None:
    vem = _build()
    with pytest.raises(ValueError, match="encoder_id is required"):
        vem.identify((1.0, 0.0, 0.0, 0.0), k=1)


def test_vemem_identify_by_precomputed_vector_routes_to_store() -> None:
    vem = _build()
    observations = vem.observe(b"IMG_A")
    entity = vem.label([o.id for o in observations], name="Charlie")

    # IMG_A produces embed() = (ord('I'), 1.0, 0.0, 0.0) — reuse that vector.
    vector = (float(ord("I")), 1.0, 0.0, 0.0)
    candidates = vem.identify(vector, encoder_id="stub/encoder@1", k=1)

    assert candidates[0].entity.id == entity.id


# ---------- ops passthroughs ----------


def test_vemem_full_roundtrip_label_remember_recall() -> None:
    vem = _build()
    observations = vem.observe(b"IMG_A")
    entity = vem.label([o.id for o in observations], name="Charlie")
    vem.remember(entity.id, "runs marathons", source=Source.USER)

    snapshot = vem.recall(entity.id)

    assert snapshot.entity.name == "Charlie"
    assert any(f.content == "runs marathons" for f in snapshot.facts)


def test_vemem_forget_then_export_tombstone() -> None:
    vem = _build()
    observations = vem.observe(b"IMG_A")
    entity = vem.label([o.id for o in observations], name="Temp")
    vem.remember(entity.id, "temporary fact")

    counts = vem.forget(entity.id)

    assert counts["bindings"] >= 1
    dump = vem.export(entity.id)
    assert dump["entity"]["status"] == "forgotten"
    assert dump["entity"]["name"] == ""


def test_vemem_undo_reverses_label() -> None:
    vem = _build()
    observations = vem.observe(b"IMG_A")
    vem.label([o.id for o in observations], name="Charlie")

    vem.undo()

    # After undo, no current positive binding on the observation
    current = vem.store.current_positive_bindings(observations[0].id)
    assert current == []


# ---------- graceful degradation ----------


def test_vemem_raises_clear_error_when_image_pipeline_unavailable() -> None:
    # Build without encoder/detector, and with a fake store so we don't trigger LanceDB
    vem = Vemem(
        store=FakeStore(),
        encoder=None,
        detector=None,
        clock=_FixedClock(T),
        actor="test:unit",
    )

    with pytest.raises(RuntimeError, match="image pipeline unavailable"):
        vem.observe(b"IMG_A")


# ---------- top-level re-exports ----------


def test_top_level_imports_match_public_surface() -> None:
    from vemem import (
        Binding,  # noqa: F401
        Candidate,  # noqa: F401
        Entity,  # noqa: F401
        EntityUnavailableError,  # noqa: F401
        Event,  # noqa: F401
        EventLog,  # noqa: F401
        Fact,  # noqa: F401
        Kind,  # noqa: F401
        ModalityMismatchError,  # noqa: F401
        Observation,  # noqa: F401
        Polarity,  # noqa: F401
        RecallSnapshot,  # noqa: F401
        Relationship,  # noqa: F401
        Source,  # noqa: F401
        Status,  # noqa: F401
        Vemem,  # noqa: F401
        VemError,  # noqa: F401
    )


def test_top_level_modality_enum_matches_core() -> None:
    from vemem import Modality as TopLevel
    from vemem.core.enums import Modality as Core

    assert TopLevel is Core
