"""End-to-end stack test: Vemem facade → real LanceDBStore → spec §4 scenarios.

Exercises the full production stack except the real encoders — stub
encoder/detector are injected so the test is offline and deterministic. The
point is to catch integration bugs (Protocol mismatches, serialization
roundtrips, ordering invariants across the full chain) without paying the
200MB model-download cost.

Tests the nine v0 acceptance scenarios from ``docs/plan/v0-implementation.md``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vemem import Source, Status, Vemem


class _FixedClock:
    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at


class _StubDetector:
    id = "stub/detector@1"

    def detect(self, image_bytes: bytes) -> list[tuple[int, int, int, int]]:
        # One bbox per first-byte "scene"; different bytes → different bboxes.
        return [(0, 0, 100, 100)]


class _StubEncoder:
    id = "stub/encoder@1"
    dim = 4

    def embed(self, image_bytes: bytes) -> tuple[float, ...]:
        # deterministic: keyed on first byte so IMG_A and IMG_B differ
        b = image_bytes[0] if image_bytes else 0
        return (float(b), 1.0, 0.0, 0.0)


T0 = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def vem(tmp_path: Path) -> Iterator[Vemem]:
    """Real LanceDBStore at a tempdir + stub encoder + stub detector."""
    instance = Vemem(
        home=tmp_path / "vemem",
        encoder=_StubEncoder(),
        detector=_StubDetector(),
        clock=_FixedClock(T0),
        actor="e2e:test",
    )
    try:
        yield instance
    finally:
        instance.close()


# ---- acceptance criteria 1: observe → label → identify roundtrip ----


def test_roundtrip_observe_label_identify(vem: Vemem) -> None:
    observations = vem.observe(b"IMG_CHARLIE")
    entity = vem.label([o.id for o in observations], name="Charlie")

    candidates = vem.identify(b"IMG_CHARLIE", k=5)

    assert len(candidates) == 1
    assert candidates[0].entity.id == entity.id
    assert candidates[0].confidence > 0.99


# ---- acceptance criteria 2: remember + recall populates facts ----


def test_remember_recall_populates_facts(vem: Vemem) -> None:
    observations = vem.observe(b"IMG_CHARLIE")
    entity = vem.label([o.id for o in observations], name="Charlie")

    vem.remember(entity.id, "runs marathons", source=Source.USER)
    vem.remember(entity.id, "works at Acme", source=Source.USER)

    snapshot = vem.recall(entity.id)
    contents = {f.content for f in snapshot.facts}
    assert contents == {"runs marathons", "works at Acme"}


# ---- acceptance criteria 3: correction test — merge + undo ----


def test_merge_then_undo_restores_entities(vem: Vemem) -> None:
    ob1 = vem.observe(b"IMG_A")
    ob2 = vem.observe(b"IMG_B")
    a = vem.label([o.id for o in ob1], name="unknown_7")
    b = vem.label([o.id for o in ob2], name="unknown_12")

    vem.remember(b.id, "spotted on 4/17")

    winner = vem.merge([a.id, b.id], keep="oldest")
    snapshot = vem.recall(winner.id)
    assert "spotted on 4/17" in {f.content for f in snapshot.facts}
    assert vem.store.get_entity(b.id).status is Status.MERGED_INTO  # type: ignore[union-attr]

    vem.undo()

    assert vem.store.get_entity(b.id).status is Status.ACTIVE  # type: ignore[union-attr]
    b_snapshot = vem.recall(b.id)
    assert "spotted on 4/17" in {f.content for f in b_snapshot.facts}


# ---- acceptance criteria 4: privacy test — forget + prune ----


def test_forget_removes_observation_rows_and_tombstones_entity(vem: Vemem) -> None:
    observations = vem.observe(b"IMG_CHARLIE")
    entity = vem.label([o.id for o in observations], name="Charlie")

    counts = vem.forget(entity.id)
    assert counts["observations"] >= 1
    assert counts["bindings"] >= 1

    ghost = vem.store.get_entity(entity.id)
    assert ghost is not None
    assert ghost.status is Status.FORGOTTEN
    assert ghost.name == ""

    # The original observation is gone; the bindings table no longer references it
    assert vem.store.get_observation(observations[0].id) is None

    # identify() with the same vector returns nothing — the entity is erased
    candidates = vem.identify(b"IMG_CHARLIE", k=5)
    assert candidates == []


# ---- acceptance criteria 5: export ----


def test_export_returns_structured_dump(vem: Vemem) -> None:
    observations = vem.observe(b"IMG_CHARLIE")
    entity = vem.label([o.id for o in observations], name="Charlie")
    vem.remember(entity.id, "runs marathons")

    dump = vem.export(entity.id)

    assert dump["entity"]["name"] == "Charlie"
    assert any(f["content"] == "runs marathons" for f in dump["facts"])
    assert len(dump["observations"]) == 1
    assert dump["embeddings"] == []  # excluded by default


def test_export_with_embeddings_includes_vectors(vem: Vemem) -> None:
    observations = vem.observe(b"IMG_CHARLIE")
    entity = vem.label([o.id for o in observations], name="Charlie")

    dump = vem.export(entity.id, include_embeddings=True)
    assert len(dump["embeddings"]) == 1
    assert dump["embeddings"][0]["encoder_id"] == "stub/encoder@1"


# ---- context-manager lifecycle ----


def test_vemem_context_manager_closes_store(tmp_path: Path) -> None:
    path = tmp_path / "vemem"
    with Vemem(
        home=path,
        encoder=_StubEncoder(),
        detector=_StubDetector(),
        clock=_FixedClock(T0),
        actor="e2e:test",
    ) as vem:
        observations = vem.observe(b"IMG_A")
        assert len(observations) == 1

    # After exit, reopening should still find the persisted observation
    with Vemem(
        home=path,
        encoder=_StubEncoder(),
        detector=_StubDetector(),
        clock=_FixedClock(T0),
        actor="e2e:test",
    ) as vem:
        assert vem.store.get_observation(observations[0].id) is not None


# ---- cross-encoder guard: different encoder id → empty results ----


def test_identify_refuses_cross_encoder_gallery(vem: Vemem) -> None:
    observations = vem.observe(b"IMG_A")
    vem.label([o.id for o in observations], name="Charlie")

    # Query with an encoder id that has no gallery
    results = vem.identify(
        (1.0, 0.0, 0.0, 0.0),
        encoder_id="different/encoder@99",
        k=1,
    )
    assert results == []
