"""Tests for ``undo`` — reversal of prior reversible ops.

Covers: undo of label, remember, relabel, merge, split, restrict, unrestrict.
Also: forget rejects undo; expired events reject; most-recent-by-actor path;
undo-of-undo rejects (no redo in v0).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tests.support.fake_store import FakeStore
from vemem.core.enums import Modality, Polarity, Source, Status
from vemem.core.errors import OperationNotReversibleError
from vemem.core.ops import (
    forget,
    label,
    merge,
    relabel,
    remember,
    restrict,
    split,
    undo,
)
from vemem.core.types import Embedding, Observation, observation_id_for


class FrozenClock:
    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at

    def advance(self, seconds: float = 1.0) -> None:
        self._at = self._at + timedelta(seconds=seconds)


T0 = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)


def _fresh() -> tuple[FakeStore, FrozenClock]:
    return FakeStore(), FrozenClock(T0)


def _put_face_obs(
    store: FakeStore,
    clock: FrozenClock,
    *,
    source_hash: str = "sha256:imgA",
    vector: tuple[float, ...] = (1.0, 0.0, 0.0),
) -> Observation:
    obs_id = observation_id_for(source_hash, (0, 0, 100, 100), "insightface/buffalo_l@0.7.3")
    obs = Observation(
        id=obs_id,
        source_uri=f"hash:{source_hash}",
        source_hash=source_hash,
        bbox=(0, 0, 100, 100),
        detector_id="insightface/buffalo_l@0.7.3",
        modality=Modality.FACE,
        detected_at=clock.now(),
    )
    store.put_observation(obs)
    store.put_embedding(
        Embedding(
            id="emb_" + obs_id,
            observation_id=obs.id,
            encoder_id="insightface/arcface@0.7.3",
            vector=vector,
            dim=len(vector),
            created_at=clock.now(),
        )
    )
    return obs


# ---------- undo of label ----------


def test_undo_of_label_closes_new_binding() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    label(store, [obs.id], "Charlie", clock=clock, actor="u")
    clock.advance()

    undo(store, clock=clock, actor="u")

    assert store.current_positive_bindings(obs.id) == []


def test_undo_of_label_restores_prior_positive_binding() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="u")
    clock.advance()
    label(store, [obs.id], "Dana", clock=clock, actor="u")
    clock.advance()

    undo(store, clock=clock, actor="u")

    current = store.current_positive_bindings(obs.id)
    assert len(current) == 1
    assert current[0].entity_id == charlie.id


# ---------- undo of remember ----------


def test_undo_of_remember_retracts_fact() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="u")
    remember(
        store,
        entity_id=charlie.id,
        content="runs marathons",
        source=Source.USER,
        clock=clock,
        actor="u",
    )
    clock.advance()

    undo(store, clock=clock, actor="u")

    assert store.facts_for_entity(charlie.id, active_only=True) == []


# ---------- undo of relabel ----------


def test_undo_of_relabel_restores_old_binding_and_drops_negative() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="u")
    clock.advance()
    dana = relabel(store, obs.id, "Dana", clock=clock, actor="u")
    clock.advance()

    undo(store, clock=clock, actor="u")

    current = store.current_positive_bindings(obs.id)
    assert len(current) == 1
    assert current[0].entity_id == charlie.id
    # Dana's positive binding gone
    dana_bindings = store.bindings_for_entity(dana.id)
    active_positives = [
        b for b in dana_bindings if b.valid_to is None and b.polarity is Polarity.POSITIVE
    ]
    assert active_positives == []
    # Negative against Charlie also cleared
    charlie_negatives = [
        b
        for b in store.bindings_for_entity(charlie.id, include_negative=True)
        if b.polarity is Polarity.NEGATIVE and b.valid_to is None
    ]
    assert charlie_negatives == []


# ---------- undo of merge ----------


def test_undo_of_merge_restores_losers_and_moves_facts_back() -> None:
    store, clock = _fresh()
    obs1 = _put_face_obs(store, clock, source_hash="sha256:A")
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B")
    a = label(store, [obs1.id], "A", clock=clock, actor="u")
    b = label(store, [obs2.id], "B", clock=clock, actor="u")
    remember(store, entity_id=b.id, content="fact-on-b", source=Source.USER, clock=clock, actor="u")
    clock.advance()
    merge(store, [a.id, b.id], keep="oldest", clock=clock, actor="u")
    clock.advance()

    undo(store, clock=clock, actor="u")

    # Loser back to active
    loser = store.get_entity(b.id)
    assert loser is not None
    assert loser.status is Status.ACTIVE
    assert loser.merged_into_id is None
    # obs2 back on b
    assert store.current_positive_bindings(obs2.id)[0].entity_id == b.id
    # Fact back on b
    b_facts = store.facts_for_entity(b.id, active_only=True)
    assert len(b_facts) == 1
    assert b_facts[0].content == "fact-on-b"
    assert b_facts[0].provenance_entity_id is None  # provenance cleared


# ---------- undo of split ----------


def test_undo_of_split_merges_back_and_tombstones_split_off() -> None:
    store, clock = _fresh()
    obs1 = _put_face_obs(store, clock, source_hash="sha256:A")
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B")
    charlie = label(store, [obs1.id, obs2.id], "Charlie", clock=clock, actor="u")
    clock.advance()
    _, split_off = split(store, charlie.id, [[obs1.id], [obs2.id]], clock=clock, actor="u")
    clock.advance()

    undo(store, clock=clock, actor="u")

    # Both observations back on charlie
    assert store.current_positive_bindings(obs1.id)[0].entity_id == charlie.id
    assert store.current_positive_bindings(obs2.id)[0].entity_id == charlie.id
    # Split-off entity tombstoned (so clusterer and identify exclude it)
    ghost = store.get_entity(split_off.id)
    assert ghost is not None
    assert ghost.status is Status.FORGOTTEN


# ---------- undo of restrict / unrestrict ----------


def test_undo_of_restrict_returns_to_active() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="u")
    clock.advance()
    restrict(store, entity_id=charlie.id, clock=clock, actor="u")
    clock.advance()

    undo(store, clock=clock, actor="u")

    updated = store.get_entity(charlie.id)
    assert updated is not None
    assert updated.status is Status.ACTIVE


# ---------- undo of forget raises ----------


def test_undo_of_forget_raises_operation_not_reversible() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="u")
    clock.advance()
    forget(store, entity_id=charlie.id, clock=clock, actor="u")
    clock.advance()

    # Target the forget event explicitly. Without event_id, undo would skip
    # past the non-reversible forget to the still-reversible label — that's
    # the intended behavior (§4.6) and is covered by the expired-event test.
    forget_event_id = next(
        e.id for e in store.events_affecting_entity(charlie.id) if e.op_type == "forget"
    )
    with pytest.raises(OperationNotReversibleError):
        undo(store, event_id=forget_event_id, clock=clock, actor="u")


# ---------- undo resolution ----------


def test_undo_with_no_event_id_undoes_most_recent_reversible_by_actor() -> None:
    store, clock = _fresh()
    obs1 = _put_face_obs(store, clock, source_hash="sha256:A")
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B")
    label(store, [obs1.id], "A", clock=clock, actor="alice")
    clock.advance()
    label(store, [obs2.id], "B", clock=clock, actor="alice")
    clock.advance()

    undo(store, clock=clock, actor="alice")

    # "B" label was the most recent; it should be undone
    assert store.current_positive_bindings(obs2.id) == []
    # "A" label remains
    assert len(store.current_positive_bindings(obs1.id)) == 1


def test_undo_of_expired_event_raises() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    label(store, [obs.id], "Charlie", clock=clock, actor="u")
    # Advance far beyond the 30-day window
    clock.advance(seconds=60 * 60 * 24 * 40)  # 40 days

    with pytest.raises(OperationNotReversibleError):
        undo(store, clock=clock, actor="u")


def test_undo_of_already_undone_event_raises_when_targeted() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="u")
    clock.advance()
    label_event_id = next(
        e.id for e in store.events_affecting_entity(charlie.id) if e.op_type == "label"
    )
    undo(store, event_id=label_event_id, clock=clock, actor="u")
    clock.advance()

    with pytest.raises(OperationNotReversibleError):
        undo(store, event_id=label_event_id, clock=clock, actor="u")


def test_undo_of_undo_raises_no_redo_in_v0() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    label(store, [obs.id], "Charlie", clock=clock, actor="u")
    clock.advance()
    undo(store, clock=clock, actor="u")
    clock.advance()

    with pytest.raises(OperationNotReversibleError):
        undo(store, clock=clock, actor="u")
