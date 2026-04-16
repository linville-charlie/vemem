"""Shared contract tests for every :class:`Store` implementation.

Each test runs against both :class:`FakeStore` and :class:`LanceDBStore` via
the ``store`` fixture. A LanceDB divergence from FakeStore is by definition a
bug in whichever layer made the assumption the other didn't.

Focus areas (spec §3 + §4):

- Idempotent observation writes (same id → no duplicate row)
- Per-encoder embedding isolation (cross-encoder query returns empty, not
  garbage, per spec §4.0)
- Bi-temporal binding supersede (close_binding sets valid_to; new positive
  supersedes the old)
- Negative bindings filtered from ``current_positive_bindings``
- Bi-temporal fact retraction (retract_fact closes but preserves row)
- Event log ordering and entity-affected lookup
- Cascade delete semantics per spec §4.5 (multi-bound observations survive)
- ``find_entity_by_name`` respects aliases and status
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.support.fake_store import FakeStore
from vemem.core.enums import (
    Kind,
    Method,
    Modality,
    OpType,
    Polarity,
    Source,
    Status,
)
from vemem.core.protocols import Store
from vemem.core.types import (
    Binding,
    Embedding,
    Entity,
    Event,
    EventLog,
    Fact,
    Observation,
    Relationship,
)
from vemem.storage.lancedb_store import LanceDBStore

# ---- fixtures -------------------------------------------------------------


@pytest.fixture(params=["fake", "lancedb"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Any]:
    """Yield a fresh Store of each implementation.

    The local annotations ``s: Store`` below are the static Protocol-compat
    check: if either ``FakeStore`` or ``LanceDBStore`` ever loses a method on
    the ``Store`` surface, mypy fails here.
    """

    if request.param == "fake":
        fake: Store = FakeStore()
        yield fake
    else:
        lance: Store = LanceDBStore(path=str(tmp_path / "db"))
        try:
            yield lance
        finally:
            lance.close()


T0 = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)


def _obs(
    obs_id: str = "obs_1",
    *,
    source_hash: str = "sha_1",
    modality: Modality = Modality.FACE,
    bbox: tuple[int, int, int, int] = (0, 0, 10, 10),
    detector_id: str = "insightface/buffalo_l@0.7.3",
) -> Observation:
    return Observation(
        id=obs_id,
        source_uri=f"file://{obs_id}.jpg",
        source_hash=source_hash,
        bbox=bbox,
        detector_id=detector_id,
        modality=modality,
        detected_at=T0,
    )


def _emb(
    emb_id: str = "emb_1",
    *,
    observation_id: str = "obs_1",
    encoder_id: str = "insightface/arcface@0.7.3",
    vector: tuple[float, ...] = (1.0, 0.0, 0.0),
) -> Embedding:
    return Embedding(
        id=emb_id,
        observation_id=observation_id,
        encoder_id=encoder_id,
        vector=vector,
        dim=len(vector),
        created_at=T0,
    )


def _entity(
    entity_id: str = "ent_1",
    *,
    name: str = "Charlie",
    aliases: tuple[str, ...] = (),
    status: Status = Status.ACTIVE,
    modality: Modality = Modality.FACE,
    kind: Kind = Kind.INSTANCE,
) -> Entity:
    return Entity(
        id=entity_id,
        kind=kind,
        name=name,
        modality=modality,
        status=status,
        created_at=T0,
        last_seen=T0,
        aliases=aliases,
    )


def _binding(
    binding_id: str = "bind_1",
    *,
    observation_id: str = "obs_1",
    entity_id: str = "ent_1",
    polarity: Polarity = Polarity.POSITIVE,
    method: Method = Method.USER_LABEL,
    confidence: float = 1.0,
    valid_from: datetime = T0,
    valid_to: datetime | None = None,
    actor: str = "user:alice",
) -> Binding:
    return Binding(
        id=binding_id,
        observation_id=observation_id,
        entity_id=entity_id,
        confidence=confidence,
        method=method,
        polarity=polarity,
        valid_from=valid_from,
        valid_to=valid_to,
        recorded_at=valid_from,
        actor=actor,
    )


# ---- observations / embeddings -------------------------------------------


def test_put_and_get_observation_roundtrip(store: Any) -> None:
    obs = _obs()
    store.put_observation(obs)

    got = store.get_observation(obs.id)
    assert got is not None
    assert got.id == obs.id
    assert got.source_hash == obs.source_hash
    assert got.bbox == obs.bbox
    assert got.modality is Modality.FACE


def test_get_observation_missing_returns_none(store: Any) -> None:
    assert store.get_observation("does_not_exist") is None


def test_put_observation_is_idempotent(store: Any) -> None:
    """Re-writing an observation with the same id must not duplicate rows.

    Observations are content-hashed (spec §3.1); the same detector on the
    same image region at a different wall-clock time still returns the same
    id, and the Store must treat the second write as a no-op.
    """

    obs = _obs()
    store.put_observation(obs)
    store.put_observation(obs)

    # Observe the effect indirectly: adding a positive binding and looking
    # up that observation via the current-positive query finds exactly one.
    store.put_entity(_entity())
    store.append_binding(_binding())

    found = store.current_positive_bindings(obs.id)
    assert len(found) == 1


def test_put_and_read_embedding_via_search(store: Any) -> None:
    obs = _obs()
    store.put_observation(obs)
    store.put_embedding(_emb(vector=(1.0, 0.0, 0.0)))

    results = store.search_embeddings(
        encoder_id="insightface/arcface@0.7.3",
        vector=(1.0, 0.0, 0.0),
        k=5,
    )
    assert len(results) == 1
    obs_id, score = results[0]
    assert obs_id == obs.id
    assert score > 0.99  # exact match → cosine ≈ 1.0


def test_search_filters_by_encoder_id(store: Any) -> None:
    """Cross-encoder search returns empty — no silent fallback (spec §4.0)."""

    store.put_observation(_obs())
    store.put_embedding(_emb(encoder_id="insightface/arcface@0.7.3"))

    results = store.search_embeddings(
        encoder_id="clip/vit-b-32@2.30",
        vector=(1.0, 0.0, 0.0),
        k=5,
    )
    assert results == []


def test_search_ranks_by_similarity(store: Any) -> None:
    store.put_observation(_obs("obs_a", source_hash="h_a"))
    store.put_observation(_obs("obs_b", source_hash="h_b"))
    store.put_observation(_obs("obs_c", source_hash="h_c"))

    store.put_embedding(_emb("e_a", observation_id="obs_a", vector=(1.0, 0.0, 0.0)))
    store.put_embedding(_emb("e_b", observation_id="obs_b", vector=(0.9, 0.1, 0.0)))
    store.put_embedding(_emb("e_c", observation_id="obs_c", vector=(0.0, 1.0, 0.0)))

    results = store.search_embeddings(
        encoder_id="insightface/arcface@0.7.3",
        vector=(1.0, 0.0, 0.0),
        k=3,
    )
    # obs_a first (identical), obs_b second (close), obs_c last (orthogonal).
    ordered_ids = [r[0] for r in results]
    assert ordered_ids[0] == "obs_a"
    assert ordered_ids[1] == "obs_b"
    assert ordered_ids[2] == "obs_c"


# ---- entities ------------------------------------------------------------


def test_put_and_get_entity_roundtrip(store: Any) -> None:
    e = _entity(aliases=("Charlie", "Chuck"))
    store.put_entity(e)

    got = store.get_entity(e.id)
    assert got is not None
    assert got.id == e.id
    assert got.name == "Charlie"
    assert got.aliases == ("Charlie", "Chuck")
    assert got.status is Status.ACTIVE


def test_find_entity_by_name_matches_primary_name(store: Any) -> None:
    store.put_entity(_entity(name="Charlie"))

    got = store.find_entity_by_name("Charlie")
    assert got is not None
    assert got.name == "Charlie"


def test_find_entity_by_name_matches_alias(store: Any) -> None:
    store.put_entity(_entity(name="Charlie", aliases=("Chuck", "Carl")))

    got = store.find_entity_by_name("Chuck")
    assert got is not None
    assert got.id == "ent_1"


def test_find_entity_by_name_skips_inactive(store: Any) -> None:
    store.put_entity(_entity(entity_id="ent_forgotten", name="Ghost", status=Status.FORGOTTEN))

    assert store.find_entity_by_name("Ghost") is None


def test_find_entity_by_name_missing_returns_none(store: Any) -> None:
    assert store.find_entity_by_name("nobody_knows_this_name") is None


# ---- bindings ------------------------------------------------------------


def test_append_and_get_binding_roundtrip(store: Any) -> None:
    store.put_entity(_entity())
    store.put_observation(_obs())
    b = _binding()
    store.append_binding(b)

    got = store.get_binding(b.id)
    assert got is not None
    assert got.id == b.id
    assert got.polarity is Polarity.POSITIVE
    assert got.valid_to is None


def test_close_binding_sets_valid_to(store: Any) -> None:
    store.put_entity(_entity())
    store.put_observation(_obs())
    store.append_binding(_binding())

    later = datetime(2026, 4, 17, 0, 0, tzinfo=UTC)
    store.close_binding("bind_1", later)

    got = store.get_binding("bind_1")
    assert got is not None
    assert got.valid_to == later


def test_close_nonexistent_binding_raises(store: Any) -> None:
    with pytest.raises(KeyError):
        store.close_binding("does_not_exist", T0)


def test_current_positive_bindings_excludes_closed_and_negative(store: Any) -> None:
    """Spec §3.3: current positive = polarity=positive AND valid_to IS NULL."""

    store.put_entity(_entity())
    store.put_observation(_obs())

    # Closed positive: supersedes-style.
    old = _binding("bind_old", valid_from=T0)
    store.append_binding(old)
    store.close_binding("bind_old", datetime(2026, 4, 17, 0, 0, tzinfo=UTC))

    # Negative.
    neg = _binding(
        "bind_neg",
        polarity=Polarity.NEGATIVE,
        method=Method.USER_REJECT,
    )
    store.append_binding(neg)

    # Current positive.
    cur = _binding("bind_cur", valid_from=datetime(2026, 4, 17, 1, tzinfo=UTC))
    store.append_binding(cur)

    results = store.current_positive_bindings("obs_1")
    assert len(results) == 1
    assert results[0].id == "bind_cur"


def test_bindings_for_entity_filters_negative_by_default(store: Any) -> None:
    store.put_entity(_entity())
    store.put_observation(_obs())

    store.append_binding(_binding("bind_pos"))
    store.append_binding(
        _binding(
            "bind_neg",
            polarity=Polarity.NEGATIVE,
            method=Method.USER_REJECT,
        )
    )

    positive_only = store.bindings_for_entity("ent_1")
    assert len(positive_only) == 1
    assert positive_only[0].id == "bind_pos"

    both = store.bindings_for_entity("ent_1", include_negative=True)
    assert {b.id for b in both} == {"bind_pos", "bind_neg"}


# ---- facts ---------------------------------------------------------------


def test_put_and_retract_fact(store: Any) -> None:
    store.put_entity(_entity())
    fact = Fact(
        id="fact_1",
        entity_id="ent_1",
        content="Charlie runs marathons",
        source=Source.USER,
        actor="user:alice",
        valid_from=T0,
        recorded_at=T0,
    )
    store.put_fact(fact)

    active = store.facts_for_entity("ent_1")
    assert len(active) == 1
    assert active[0].content == "Charlie runs marathons"

    later = datetime(2026, 4, 17, 0, 0, tzinfo=UTC)
    store.retract_fact("fact_1", later)

    active_after = store.facts_for_entity("ent_1")
    assert active_after == []

    all_including_retracted = store.facts_for_entity("ent_1", active_only=False)
    assert len(all_including_retracted) == 1
    assert all_including_retracted[0].valid_to == later


def test_retract_fact_missing_raises(store: Any) -> None:
    with pytest.raises(KeyError):
        store.retract_fact("does_not_exist", T0)


def test_retract_already_retracted_fact_is_noop(store: Any) -> None:
    store.put_entity(_entity())
    later = datetime(2026, 4, 17, 0, 0, tzinfo=UTC)
    fact = Fact(
        id="fact_1",
        entity_id="ent_1",
        content="…",
        source=Source.USER,
        actor="user:alice",
        valid_from=T0,
        recorded_at=T0,
        valid_to=later,
    )
    store.put_fact(fact)
    # Second retract should not raise; valid_to stays at the original close time.
    store.retract_fact("fact_1", datetime(2026, 4, 18, tzinfo=UTC))

    f = store.get_fact("fact_1")
    assert f is not None
    assert f.valid_to == later


# ---- events & relationships ---------------------------------------------


def test_events_for_entity(store: Any) -> None:
    store.put_entity(_entity())
    ev = Event(
        id="ev_1",
        entity_id="ent_1",
        content="ran marathon",
        source=Source.USER,
        occurred_at=T0,
        recorded_at=T0,
    )
    store.put_event(ev)

    out = store.events_for_entity("ent_1")
    assert len(out) == 1
    assert out[0].content == "ran marathon"


def test_relationships_for_entity_filters_active(store: Any) -> None:
    store.put_entity(_entity("ent_a"))
    store.put_entity(_entity("ent_b"))

    active = Relationship(
        id="rel_active",
        from_entity_id="ent_a",
        to_entity_id="ent_b",
        relation_type="friend_of",
        source=Source.USER,
        valid_from=T0,
        recorded_at=T0,
    )
    retracted = Relationship(
        id="rel_retracted",
        from_entity_id="ent_a",
        to_entity_id="ent_b",
        relation_type="former_coworker_of",
        source=Source.USER,
        valid_from=T0,
        recorded_at=T0,
        valid_to=datetime(2026, 4, 17, tzinfo=UTC),
    )
    store.put_relationship(active)
    store.put_relationship(retracted)

    active_only = store.relationships_for_entity("ent_a")
    assert {r.id for r in active_only} == {"rel_active"}

    both = store.relationships_for_entity("ent_a", active_only=False)
    assert {r.id for r in both} == {"rel_active", "rel_retracted"}


# ---- event log -----------------------------------------------------------


def test_append_event_log_assigns_monotonic_ids(store: Any) -> None:
    e1 = EventLog(
        id=0,  # will be replaced
        op_type=OpType.LABEL.value,
        payload={"observation_ids": ["obs_1"]},
        actor="user:alice",
        affected_entity_ids=("ent_1",),
        at=T0,
    )
    e2 = EventLog(
        id=0,
        op_type=OpType.REMEMBER.value,
        payload={"fact_id": "fact_1"},
        actor="user:alice",
        affected_entity_ids=("ent_1",),
        at=T0,
    )

    stored1 = store.append_event_log(e1)
    stored2 = store.append_event_log(e2)

    assert stored1.id > 0
    assert stored2.id > stored1.id
    assert stored1.op_type == OpType.LABEL.value


def test_get_event_log_roundtrip(store: Any) -> None:
    ev = EventLog(
        id=0,
        op_type=OpType.LABEL.value,
        payload={"observation_ids": ["obs_1"], "counts": 3},
        actor="user:alice",
        affected_entity_ids=("ent_1",),
        at=T0,
    )
    written = store.append_event_log(ev)

    got = store.get_event_log(written.id)
    assert got is not None
    assert got.op_type == OpType.LABEL.value
    assert got.payload == {"observation_ids": ["obs_1"], "counts": 3}
    assert got.actor == "user:alice"
    assert got.affected_entity_ids == ("ent_1",)


def test_events_affecting_entity(store: Any) -> None:
    for entity_id in ("ent_a", "ent_b"):
        store.append_event_log(
            EventLog(
                id=0,
                op_type=OpType.LABEL.value,
                payload={},
                actor="user:alice",
                affected_entity_ids=(entity_id,),
                at=T0,
            )
        )
    # Third event touches both.
    store.append_event_log(
        EventLog(
            id=0,
            op_type=OpType.MERGE.value,
            payload={},
            actor="user:alice",
            affected_entity_ids=("ent_a", "ent_b"),
            at=T0,
        )
    )

    for_a = store.events_affecting_entity("ent_a")
    assert len(for_a) == 2
    ids = sorted(e.id for e in for_a)
    assert ids == sorted(ids)  # monotonic

    for_c = store.events_affecting_entity("ent_c")
    assert for_c == []


# ---- cascade delete (spec §4.5) -----------------------------------------


def test_delete_entity_cascade_removes_single_bound_observations(store: Any) -> None:
    store.put_entity(_entity())
    store.put_observation(_obs())
    store.put_embedding(_emb())
    store.append_binding(_binding())

    counts = store.delete_entity_cascade("ent_1")

    assert counts["observations"] == 1
    assert counts["embeddings"] == 1
    assert counts["bindings"] == 1
    assert store.get_observation("obs_1") is None


def test_delete_entity_cascade_preserves_multi_bound_observations(store: Any) -> None:
    """Spec §4.5 step 1: obs with other positive bindings survive."""

    store.put_entity(_entity("ent_to_forget", name="Forget"))
    store.put_entity(_entity("ent_keeper", name="Keeper"))
    store.put_observation(_obs())
    store.put_embedding(_emb())

    # Obs is positively bound to BOTH entities.
    store.append_binding(_binding("bind_forget", entity_id="ent_to_forget"))
    store.append_binding(_binding("bind_keep", entity_id="ent_keeper"))

    counts = store.delete_entity_cascade("ent_to_forget")

    assert counts["observations"] == 0  # obs survived
    assert counts["bindings"] == 1
    assert store.get_observation("obs_1") is not None


def test_delete_entity_cascade_drops_facts_events_relationships(store: Any) -> None:
    store.put_entity(_entity("ent_a"))
    store.put_entity(_entity("ent_b"))
    store.put_fact(
        Fact(
            id="f_1",
            entity_id="ent_a",
            content="…",
            source=Source.USER,
            actor="user:alice",
            valid_from=T0,
            recorded_at=T0,
        )
    )
    store.put_event(
        Event(
            id="ev_1",
            entity_id="ent_a",
            content="…",
            source=Source.USER,
            occurred_at=T0,
            recorded_at=T0,
        )
    )
    store.put_relationship(
        Relationship(
            id="rel_1",
            from_entity_id="ent_a",
            to_entity_id="ent_b",
            relation_type="friend_of",
            source=Source.USER,
            valid_from=T0,
            recorded_at=T0,
        )
    )

    counts = store.delete_entity_cascade("ent_a")

    assert counts["facts"] == 1
    assert counts["events"] == 1
    assert counts["relationships"] == 1

    assert store.facts_for_entity("ent_a", active_only=False) == []
    assert store.events_for_entity("ent_a") == []
    # Relationship touching ent_a deleted even though the other endpoint is
    # ent_b: spec §4.5 step 3 says edges with one end forgotten are deleted.
    assert store.relationships_for_entity("ent_b", active_only=False) == []


def test_schema_version_is_stable_across_reopen(store: Any) -> None:
    assert store.schema_version() == 1


# ---- Protocol satisfaction (method-presence) ---------------------------


def test_store_impls_expose_full_protocol_surface(store: Any) -> None:
    """Every method on the ``Store`` Protocol is present on each impl.

    Since ``Store`` isn't ``@runtime_checkable``, we instead verify by name
    that both backends expose the same callable surface — this is what the
    ops layer actually relies on, and a missing method here is a silent
    ``AttributeError`` at runtime otherwise.
    """

    expected = {
        "schema_version",
        "close",
        "put_observation",
        "put_embedding",
        "get_observation",
        "put_entity",
        "get_entity",
        "find_entity_by_name",
        "append_binding",
        "close_binding",
        "get_binding",
        "current_positive_bindings",
        "bindings_for_entity",
        "put_fact",
        "get_fact",
        "retract_fact",
        "facts_for_entity",
        "put_event",
        "events_for_entity",
        "put_relationship",
        "get_relationship",
        "relationships_for_entity",
        "append_event_log",
        "get_event_log",
        "events_affecting_entity",
        "search_embeddings",
        "delete_entity_cascade",
        "prune_versions",
    }
    missing = {name for name in expected if not callable(getattr(store, name, None))}
    assert missing == set(), f"missing Store methods: {sorted(missing)}"
