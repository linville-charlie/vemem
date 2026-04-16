"""Tests for the basic ops — label, identify, remember, recall.

Relabel / merge / split / forget / undo / restrict / export live in
``test_ops_corrections.py`` alongside the harder semantics.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.support.fake_store import FakeStore
from vemem.core.enums import Kind, Modality, Source, Status
from vemem.core.errors import EntityUnavailableError, ModalityMismatchError
from vemem.core.ops import identify, label, recall, remember
from vemem.core.types import Embedding, Observation, observation_id_for


class FrozenClock:
    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at

    def advance(self, seconds: float = 1.0) -> None:
        from datetime import timedelta

        self._at = self._at + timedelta(seconds=seconds)


T0 = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)


def _fresh() -> tuple[FakeStore, FrozenClock]:
    return FakeStore(), FrozenClock(T0)


def _put_face_obs(
    store: FakeStore,
    clock: FrozenClock,
    *,
    source_hash: str = "sha256:imgA",
    bbox: tuple[int, int, int, int] = (0, 0, 100, 100),
    encoder_id: str = "insightface/arcface@0.7.3",
    vector: tuple[float, ...] = (1.0, 0.0, 0.0),
    modality: Modality = Modality.FACE,
) -> Observation:
    obs_id = observation_id_for(source_hash, bbox, "insightface/buffalo_l@0.7.3")
    obs = Observation(
        id=obs_id,
        source_uri=f"hash:{source_hash}",
        source_hash=source_hash,
        bbox=bbox,
        detector_id="insightface/buffalo_l@0.7.3",
        modality=modality,
        detected_at=clock.now(),
    )
    store.put_observation(obs)
    store.put_embedding(
        Embedding(
            id="emb_" + obs_id,
            observation_id=obs.id,
            encoder_id=encoder_id,
            vector=vector,
            dim=len(vector),
            created_at=clock.now(),
        )
    )
    return obs


# ---------- label ----------


def test_label_creates_new_entity_on_first_use() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)

    entity = label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")

    assert entity.name == "Charlie"
    assert entity.kind is Kind.INSTANCE
    assert entity.modality is Modality.FACE
    assert entity.status is Status.ACTIVE


def test_label_reuses_existing_entity_by_name() -> None:
    store, clock = _fresh()
    obs1 = _put_face_obs(store, clock, source_hash="sha256:A")
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B")

    e1 = label(store, [obs1.id], "Charlie", clock=clock, actor="user:alice")
    e2 = label(store, [obs2.id], "Charlie", clock=clock, actor="user:alice")

    assert e1.id == e2.id


def test_label_opens_current_positive_binding() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)

    entity = label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")

    current = store.current_positive_bindings(obs.id)
    assert len(current) == 1
    assert current[0].entity_id == entity.id
    assert current[0].confidence == 1.0


def test_label_closes_prior_positive_binding_on_relabel_via_label() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    first = label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")
    clock.advance()
    second = label(store, [obs.id], "Dana", clock=clock, actor="user:alice")

    current = store.current_positive_bindings(obs.id)
    assert len(current) == 1
    assert current[0].entity_id == second.id
    assert second.id != first.id  # Dana is a new entity


def test_label_rejects_multi_modality_batch() -> None:
    store, clock = _fresh()
    face_obs = _put_face_obs(store, clock, source_hash="sha256:A")
    object_obs = _put_face_obs(store, clock, source_hash="sha256:B", modality=Modality.OBJECT)

    with pytest.raises(ModalityMismatchError):
        label(store, [face_obs.id, object_obs.id], "Mixed", clock=clock, actor="user:alice")


def test_label_rejects_forgotten_entity() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    entity = label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")
    # Manually mark as forgotten
    from dataclasses import replace

    store.put_entity(replace(entity, status=Status.FORGOTTEN, name=""))
    clock.advance()
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B")

    with pytest.raises(EntityUnavailableError):
        label(store, [obs2.id], entity.id, clock=clock, actor="user:alice")


def test_label_emits_event_log() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    entity = label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")

    events = store.events_affecting_entity(entity.id)
    assert len(events) == 1
    assert events[0].op_type == "label"
    assert events[0].actor == "user:alice"
    assert events[0].reversible_until is not None


# ---------- identify ----------


def test_identify_returns_empty_on_fresh_store() -> None:
    store, _ = _fresh()
    results = identify(store, encoder_id="insightface/arcface@0.7.3", vector=(1.0, 0.0, 0.0), k=5)
    assert results == []


def test_identify_returns_labeled_entity() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock, vector=(1.0, 0.0, 0.0))
    entity = label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")

    results = identify(
        store,
        encoder_id="insightface/arcface@0.7.3",
        vector=(1.0, 0.0, 0.0),
        k=5,
    )
    assert len(results) == 1
    assert results[0].entity.id == entity.id
    assert results[0].confidence > 0.99


def test_identify_respects_negative_bindings() -> None:
    """An observation with a negative binding must NOT surface its entity."""
    store, clock = _fresh()
    obs = _put_face_obs(store, clock, vector=(1.0, 0.0, 0.0))
    # Label and immediately record a negative binding against that entity
    entity = label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")
    clock.advance()
    from vemem.core.enums import Method, Polarity
    from vemem.core.ids import new_id
    from vemem.core.types import Binding

    store.append_binding(
        Binding(
            id="bnd_" + new_id(),
            observation_id=obs.id,
            entity_id=entity.id,
            polarity=Polarity.NEGATIVE,
            confidence=1.0,
            method=Method.USER_REJECT,
            valid_from=clock.now(),
            recorded_at=clock.now(),
            actor="user:alice",
        )
    )

    results = identify(
        store,
        encoder_id="insightface/arcface@0.7.3",
        vector=(1.0, 0.0, 0.0),
        k=5,
    )
    # The entity is negative-bound for this exact observation; should be excluded.
    assert all(c.entity.id != entity.id for c in results)


def test_identify_threshold_filters_low_confidence() -> None:
    store, clock = _fresh()
    # Put an embedding nearly orthogonal to the query
    _put_face_obs(store, clock, vector=(0.1, 0.995, 0.0))

    results = identify(
        store,
        encoder_id="insightface/arcface@0.7.3",
        vector=(1.0, 0.0, 0.0),
        k=5,
        min_confidence=0.5,
    )
    assert results == []


def test_identify_skips_unlabeled_observations() -> None:
    """Identify returns entity candidates, not raw observations.

    An observation with an embedding but no positive binding doesn't surface
    as a named candidate.
    """
    store, clock = _fresh()
    _put_face_obs(store, clock, vector=(1.0, 0.0, 0.0))  # unlabeled

    results = identify(
        store,
        encoder_id="insightface/arcface@0.7.3",
        vector=(1.0, 0.0, 0.0),
        k=5,
    )
    assert results == []


# ---------- remember / recall ----------


def test_remember_attaches_fact_to_entity() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    entity = label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")

    fact = remember(
        store,
        entity_id=entity.id,
        content="runs marathons",
        source=Source.USER,
        clock=clock,
        actor="user:alice",
    )

    facts = store.facts_for_entity(entity.id)
    assert len(facts) == 1
    assert facts[0].id == fact.id
    assert facts[0].content == "runs marathons"
    assert facts[0].valid_to is None


def test_recall_returns_active_facts_for_entity() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    entity = label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")
    remember(
        store,
        entity_id=entity.id,
        content="runs marathons",
        source=Source.USER,
        clock=clock,
        actor="user:alice",
    )
    remember(
        store,
        entity_id=entity.id,
        content="works at Acme",
        source=Source.USER,
        clock=clock,
        actor="user:alice",
    )

    snapshot = recall(store, entity_id=entity.id)
    assert snapshot.entity.id == entity.id
    assert {f.content for f in snapshot.facts} == {"runs marathons", "works at Acme"}


def test_identify_populates_candidate_facts() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock, vector=(1.0, 0.0, 0.0))
    entity = label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")
    remember(
        store,
        entity_id=entity.id,
        content="runs marathons",
        source=Source.USER,
        clock=clock,
        actor="user:alice",
    )

    results = identify(
        store,
        encoder_id="insightface/arcface@0.7.3",
        vector=(1.0, 0.0, 0.0),
        k=1,
    )
    assert len(results) == 1
    assert len(results[0].facts) == 1
    assert results[0].facts[0].content == "runs marathons"


def test_recall_includes_events_and_relationships() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")

    # Attach an event
    from vemem.core.ids import new_id
    from vemem.core.types import Event, Relationship

    store.put_event(
        Event(
            id="evt_" + new_id(),
            entity_id=charlie.id,
            content="met in kitchen",
            source=Source.USER,
            occurred_at=clock.now(),
            recorded_at=clock.now(),
        )
    )
    # And a relationship
    other = label(store, [], "Acme Inc", clock=clock, actor="user:alice")
    _ = other  # silence unused in this shape
    store.put_relationship(
        Relationship(
            id="rel_" + new_id(),
            from_entity_id=charlie.id,
            to_entity_id=other.id,
            relation_type="works_at",
            source=Source.USER,
            valid_from=clock.now(),
            recorded_at=clock.now(),
        )
    )

    snapshot = recall(store, entity_id=charlie.id)

    assert len(snapshot.events) == 1
    assert snapshot.events[0].content == "met in kitchen"
    assert len(snapshot.relationships) == 1
    assert snapshot.relationships[0].relation_type == "works_at"


def test_remember_rejects_forgotten_entity() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    entity = label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")
    from dataclasses import replace

    store.put_entity(replace(entity, status=Status.FORGOTTEN, name=""))

    with pytest.raises(EntityUnavailableError):
        remember(
            store,
            entity_id=entity.id,
            content="runs marathons",
            source=Source.USER,
            clock=clock,
            actor="user:alice",
        )
