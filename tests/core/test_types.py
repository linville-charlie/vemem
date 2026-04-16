"""Tests for the core domain types.

Covers: frozen/immutability, content-hash determinism for Observation,
default values, bi-temporal defaults on Binding.
"""

from datetime import UTC, datetime

import pytest

from vemem.core.enums import Kind, Method, Modality, Polarity, Source, Status
from vemem.core.types import (
    Binding,
    Candidate,
    Embedding,
    Entity,
    Event,
    EventLog,
    Fact,
    Observation,
    Relationship,
    observation_id_for,
)

T = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)


def _obs(modality: Modality = Modality.FACE) -> Observation:
    return Observation(
        id="obs_fixed",
        source_uri="file:///tmp/a.jpg",
        source_hash="sha256:abc",
        bbox=(0, 0, 100, 100),
        detector_id="insightface/buffalo_l@0.7.3",
        modality=modality,
        detected_at=T,
    )


def test_observation_is_frozen() -> None:
    obs = _obs()
    with pytest.raises((AttributeError, TypeError)):
        obs.source_uri = "file:///tmp/b.jpg"  # type: ignore[misc]


def test_observation_id_is_deterministic() -> None:
    a = observation_id_for(
        source_hash="sha256:abc",
        bbox=(0, 0, 100, 100),
        detector_id="insightface/buffalo_l@0.7.3",
    )
    b = observation_id_for(
        source_hash="sha256:abc",
        bbox=(0, 0, 100, 100),
        detector_id="insightface/buffalo_l@0.7.3",
    )
    assert a == b


def test_observation_id_differs_by_bbox() -> None:
    a = observation_id_for("sha256:abc", (0, 0, 100, 100), "d@1")
    b = observation_id_for("sha256:abc", (0, 0, 101, 100), "d@1")
    assert a != b


def test_observation_id_differs_by_detector() -> None:
    a = observation_id_for("sha256:abc", (0, 0, 100, 100), "d@1")
    b = observation_id_for("sha256:abc", (0, 0, 100, 100), "d@2")
    assert a != b


def test_entity_defaults_to_instance_kind() -> None:
    e = Entity(
        id="ent_1",
        kind=Kind.INSTANCE,
        name="Charlie",
        modality=Modality.FACE,
        status=Status.ACTIVE,
        created_at=T,
        last_seen=T,
    )
    assert e.kind is Kind.INSTANCE
    assert e.aliases == ()


def test_binding_default_polarity_positive_and_open() -> None:
    b = Binding(
        id="bnd_1",
        observation_id="obs_1",
        entity_id="ent_1",
        confidence=1.0,
        method=Method.USER_LABEL,
        valid_from=T,
        recorded_at=T,
        actor="user:alice",
    )
    assert b.polarity is Polarity.POSITIVE
    assert b.valid_to is None


def test_negative_binding_allowed() -> None:
    b = Binding(
        id="bnd_2",
        observation_id="obs_1",
        entity_id="ent_1",
        polarity=Polarity.NEGATIVE,
        confidence=1.0,
        method=Method.USER_REJECT,
        valid_from=T,
        recorded_at=T,
        actor="user:alice",
    )
    assert b.polarity is Polarity.NEGATIVE


def test_fact_bi_temporal_defaults() -> None:
    f = Fact(
        id="f_1",
        entity_id="ent_1",
        content="runs marathons",
        source=Source.USER,
        actor="user:alice",
        valid_from=T,
        recorded_at=T,
    )
    assert f.valid_to is None
    assert f.provenance_entity_id is None


def test_embedding_vector_is_tuple() -> None:
    emb = Embedding(
        id="emb_1",
        observation_id="obs_1",
        encoder_id="insightface/arcface@0.7.3",
        vector=(0.1, 0.2, 0.3),
        dim=3,
        created_at=T,
    )
    assert emb.vector == (0.1, 0.2, 0.3)
    assert emb.dim == 3


def test_eventlog_default_reversible_until_is_none_sentinel() -> None:
    """Concrete reversible_until is set by ops, not the dataclass default.

    The field is Optional[datetime] with default None — meaning the caller MUST
    decide explicitly. forget ops pass None; reversible ops pass at + 30 days.
    """
    ev = EventLog(
        id=1,
        op_type="label",
        payload={"entity_id": "ent_1"},
        actor="user:alice",
        affected_entity_ids=("ent_1",),
        at=T,
    )
    assert ev.reversible_until is None
    assert ev.reversed_by is None


def test_relationship_is_directed() -> None:
    r = Relationship(
        id="r_1",
        from_entity_id="ent_1",
        to_entity_id="ent_2",
        relation_type="cofounder_of",
        source=Source.USER,
        valid_from=T,
        recorded_at=T,
    )
    assert r.from_entity_id == "ent_1"
    assert r.to_entity_id == "ent_2"


def test_candidate_carries_confidence_and_matched_obs() -> None:
    entity = Entity(
        id="ent_1",
        kind=Kind.INSTANCE,
        name="Charlie",
        modality=Modality.FACE,
        status=Status.ACTIVE,
        created_at=T,
        last_seen=T,
    )
    c = Candidate(
        entity=entity,
        confidence=0.94,
        matched_observation_ids=("obs_1", "obs_2"),
        method=Method.USER_LABEL,
    )
    assert c.confidence == 0.94
    assert c.matched_observation_ids == ("obs_1", "obs_2")


def test_event_has_occurred_and_recorded_times() -> None:
    ev = Event(
        id="ev_1",
        entity_id="ent_1",
        content="met in kitchen",
        source=Source.USER,
        occurred_at=T,
        recorded_at=T,
    )
    assert ev.occurred_at == T
    assert ev.recorded_at == T
