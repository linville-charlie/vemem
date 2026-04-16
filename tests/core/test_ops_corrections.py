"""Tests for correction / lifecycle ops.

Covers: relabel, merge, split, forget, restrict, unrestrict, export.
Undo lives in ``test_ops_undo.py`` because it depends on these ops existing
first — undo must be able to reverse any op in this file.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from tests.support.fake_store import FakeStore
from vemem.core.enums import Kind, Modality, Polarity, Source, Status
from vemem.core.errors import (
    KindMismatchError,
    ModalityMismatchError,
)
from vemem.core.ids import new_id
from vemem.core.ops import (
    export,
    forget,
    identify,
    label,
    merge,
    recall,
    relabel,
    remember,
    restrict,
    split,
    unrestrict,
)
from vemem.core.types import (
    Embedding,
    Observation,
    Relationship,
    observation_id_for,
)


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


# ---------- relabel ----------


def test_relabel_moves_binding_to_new_entity() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")
    clock.advance()

    dana = relabel(store, obs.id, "Dana", clock=clock, actor="user:alice")

    current = store.current_positive_bindings(obs.id)
    assert len(current) == 1
    assert current[0].entity_id == dana.id
    assert dana.id != charlie.id


def test_relabel_emits_negative_binding_against_old_entity() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")
    clock.advance()

    relabel(store, obs.id, "Dana", clock=clock, actor="user:alice")

    charlie_bindings = store.bindings_for_entity(charlie.id, include_negative=True)
    negatives = [b for b in charlie_bindings if b.polarity is Polarity.NEGATIVE]
    assert any(b.observation_id == obs.id and b.valid_to is None for b in negatives)


def test_relabel_does_not_merge_by_name_reuse() -> None:
    """Apple Photos rule — a new name that matches an existing entity-name
    still creates a new entity. Names are display only; identity is the id.
    """
    store, clock = _fresh()
    obs_a = _put_face_obs(store, clock, source_hash="sha256:A")
    obs_b = _put_face_obs(store, clock, source_hash="sha256:B")
    charlie = label(store, [obs_a.id], "Charlie", clock=clock, actor="user:alice")
    clock.advance()

    # Another observation labeled "Charlie" re-uses the existing entity by name.
    charlie_again = label(store, [obs_b.id], "Charlie", clock=clock, actor="user:alice")
    assert charlie.id == charlie_again.id

    # But a relabel onto a NEW name gets a new entity — it does not merge with
    # an older deleted-by-forget or merged "Charlie".
    clock.advance()
    store.put_entity(replace(charlie, status=Status.FORGOTTEN, name=""))
    obs_c = _put_face_obs(store, clock, source_hash="sha256:C")
    # Relabel of a fresh obs onto "Charlie" — finds no active entity, creates new.
    new_charlie = relabel(store, obs_c.id, "Charlie", clock=clock, actor="user:alice")
    assert new_charlie.id != charlie.id


# ---------- merge ----------


def test_merge_combines_positive_bindings_on_winner() -> None:
    store, clock = _fresh()
    obs1 = _put_face_obs(store, clock, source_hash="sha256:A")
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B")
    unknown_7 = label(store, [obs1.id], "unknown_7", clock=clock, actor="user:alice")
    unknown_12 = label(store, [obs2.id], "unknown_12", clock=clock, actor="user:alice")
    clock.advance()

    winner = merge(
        store, [unknown_7.id, unknown_12.id], keep="oldest", clock=clock, actor="user:alice"
    )

    assert winner.id == unknown_7.id
    # obs2 now bound to winner, not loser
    current = store.current_positive_bindings(obs2.id)
    assert len(current) == 1
    assert current[0].entity_id == winner.id
    # loser status
    loser = store.get_entity(unknown_12.id)
    assert loser is not None
    assert loser.status is Status.MERGED_INTO
    assert loser.merged_into_id == winner.id


def test_merge_migrates_facts_with_provenance() -> None:
    store, clock = _fresh()
    obs1 = _put_face_obs(store, clock, source_hash="sha256:A")
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B")
    a = label(store, [obs1.id], "A", clock=clock, actor="user:alice")
    b = label(store, [obs2.id], "B", clock=clock, actor="user:alice")
    remember(
        store, entity_id=b.id, content="lives in Austin", source=Source.USER, clock=clock, actor="u"
    )
    clock.advance()

    winner = merge(store, [a.id, b.id], keep="oldest", clock=clock, actor="user:alice")

    facts = store.facts_for_entity(winner.id)
    assert len(facts) == 1
    assert facts[0].content == "lives in Austin"
    assert facts[0].provenance_entity_id == b.id


def test_merge_rejects_modality_mismatch() -> None:
    store, clock = _fresh()
    obs1 = _put_face_obs(store, clock, source_hash="sha256:A")
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B", modality=Modality.OBJECT)
    face_ent = label(store, [obs1.id], "F", clock=clock, actor="u")
    obj_ent = label(store, [obs2.id], "O", clock=clock, actor="u")

    with pytest.raises(ModalityMismatchError):
        merge(store, [face_ent.id, obj_ent.id], clock=clock, actor="u")


def test_merge_rejects_kind_mismatch() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    instance = label(store, [obs.id], "my mug", clock=clock, actor="u")
    # create a type entity manually
    type_entity = replace(instance, id="ent_type_mug", kind=Kind.TYPE, name="mugs")
    store.put_entity(type_entity)

    with pytest.raises(KindMismatchError):
        merge(store, [instance.id, type_entity.id], clock=clock, actor="u")


def test_merge_drops_negative_bindings_on_loser() -> None:
    """Negative bindings on losers are not migrated — they would make false
    claims about the winner's other observations (§4.3).
    """
    store, clock = _fresh()
    obs1 = _put_face_obs(store, clock, source_hash="sha256:A")
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B")
    a = label(store, [obs1.id], "A", clock=clock, actor="u")
    b = label(store, [obs2.id], "B", clock=clock, actor="u")
    clock.advance()
    relabel(store, obs2.id, "B2", clock=clock, actor="u")  # leaves negative against B
    clock.advance()

    # Merge A and B
    merge(store, [a.id, b.id], keep="oldest", clock=clock, actor="u")

    # The negative binding (obs2 ↛ B) should be closed, not migrated to winner
    loser_negatives = [
        bnd
        for bnd in store.bindings_for_entity(b.id, include_negative=True)
        if bnd.polarity is Polarity.NEGATIVE and bnd.valid_to is None
    ]
    assert loser_negatives == []


def test_merge_collapses_self_loop_relationship() -> None:
    store, clock = _fresh()
    obs1 = _put_face_obs(store, clock, source_hash="sha256:A")
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B")
    a = label(store, [obs1.id], "A", clock=clock, actor="u")
    b = label(store, [obs2.id], "B", clock=clock, actor="u")
    rel = Relationship(
        id="r_" + new_id(),
        from_entity_id=a.id,
        to_entity_id=b.id,
        relation_type="cofounder_of",
        source=Source.USER,
        valid_from=clock.now(),
        recorded_at=clock.now(),
    )
    store.put_relationship(rel)
    clock.advance()

    winner = merge(store, [a.id, b.id], keep="oldest", clock=clock, actor="u")

    rels = store.relationships_for_entity(winner.id)
    # The relationship became self-loop after merge; active relationships should be empty.
    assert rels == []


# ---------- split ----------


def test_split_creates_new_entities_for_each_group_after_first() -> None:
    store, clock = _fresh()
    obs1 = _put_face_obs(store, clock, source_hash="sha256:A")
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B")
    obs3 = _put_face_obs(store, clock, source_hash="sha256:C")
    charlie = label(store, [obs1.id, obs2.id, obs3.id], "Charlie", clock=clock, actor="u")
    clock.advance()

    results = split(store, charlie.id, [[obs1.id, obs2.id], [obs3.id]], clock=clock, actor="u")

    assert len(results) == 2
    assert results[0].id == charlie.id  # groups[0] stays
    assert results[1].id != charlie.id
    assert results[1].modality is Modality.FACE
    assert results[1].kind is Kind.INSTANCE


def test_split_emits_cross_wise_negative_bindings() -> None:
    store, clock = _fresh()
    obs1 = _put_face_obs(store, clock, source_hash="sha256:A")
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B")
    charlie = label(store, [obs1.id, obs2.id], "Charlie", clock=clock, actor="u")
    clock.advance()

    _, split_off = split(store, charlie.id, [[obs1.id], [obs2.id]], clock=clock, actor="u")

    # obs1 should have a negative binding against split_off entity
    rejects = [
        bnd
        for bnd in store.bindings_for_entity(split_off.id, include_negative=True)
        if bnd.polarity is Polarity.NEGATIVE
        and bnd.observation_id == obs1.id
        and bnd.valid_to is None
    ]
    assert len(rejects) == 1
    # obs2 should have a negative binding against charlie (the original)
    rejects2 = [
        bnd
        for bnd in store.bindings_for_entity(charlie.id, include_negative=True)
        if bnd.polarity is Polarity.NEGATIVE
        and bnd.observation_id == obs2.id
        and bnd.valid_to is None
    ]
    assert len(rejects2) == 1


def test_split_facts_stay_on_original_by_default() -> None:
    store, clock = _fresh()
    obs1 = _put_face_obs(store, clock, source_hash="sha256:A")
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B")
    charlie = label(store, [obs1.id, obs2.id], "Charlie", clock=clock, actor="u")
    remember(
        store,
        entity_id=charlie.id,
        content="runs marathons",
        source=Source.USER,
        clock=clock,
        actor="u",
    )
    clock.advance()

    _, split_off = split(store, charlie.id, [[obs1.id], [obs2.id]], clock=clock, actor="u")

    assert len(store.facts_for_entity(charlie.id)) == 1
    assert len(store.facts_for_entity(split_off.id)) == 0


def test_split_ungrouped_observations_stay_on_original() -> None:
    """Spec §4.4 — observations not listed in any group stay on the original.

    Here obs3 is ungrouped: it stays bound to Charlie while obs1 (groups[0])
    also stays on Charlie and obs2 (groups[1]) moves to the split-off entity.
    """
    store, clock = _fresh()
    obs1 = _put_face_obs(store, clock, source_hash="sha256:A")
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B")
    obs3 = _put_face_obs(store, clock, source_hash="sha256:C")
    charlie = label(store, [obs1.id, obs2.id, obs3.id], "Charlie", clock=clock, actor="u")
    clock.advance()

    charlie_after, split_off = split(
        store, charlie.id, [[obs1.id], [obs2.id]], clock=clock, actor="u"
    )

    assert charlie_after.id == charlie.id
    assert store.current_positive_bindings(obs1.id)[0].entity_id == charlie.id
    assert store.current_positive_bindings(obs2.id)[0].entity_id == split_off.id
    # obs3 was ungrouped — it stays on Charlie untouched
    assert store.current_positive_bindings(obs3.id)[0].entity_id == charlie.id


# ---------- forget ----------


def test_forget_deletes_single_bound_observations() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="u")
    clock.advance()

    counts = forget(store, entity_id=charlie.id, clock=clock, actor="u")

    assert counts["observations"] >= 1
    assert store.get_observation(obs.id) is None
    # Entity tombstone present but unusable
    ghost = store.get_entity(charlie.id)
    assert ghost is not None
    assert ghost.status is Status.FORGOTTEN
    assert ghost.name == ""
    assert ghost.aliases == ()


def test_forget_preserves_multi_bound_observations() -> None:
    """An observation bound to BOTH an instance and a type survives when one
    of them is forgotten.
    """
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    instance = label(store, [obs.id], "my red mug", clock=clock, actor="u")
    # Manually create a type entity and bind the same observation to it too.
    from vemem.core.enums import Method
    from vemem.core.ids import new_id as _nid
    from vemem.core.types import Binding

    type_entity = replace(instance, id="ent_type", kind=Kind.TYPE, name="mugs")
    store.put_entity(type_entity)
    store.append_binding(
        Binding(
            id="bnd_" + _nid(),
            observation_id=obs.id,
            entity_id=type_entity.id,
            polarity=Polarity.POSITIVE,
            confidence=0.7,
            method=Method.AUTO_SUGGEST,
            valid_from=clock.now(),
            recorded_at=clock.now(),
            actor="sys",
        )
    )
    clock.advance()

    forget(store, entity_id=instance.id, clock=clock, actor="u")

    # observation still alive because type entity still references it
    assert store.get_observation(obs.id) is not None


def test_forget_payload_is_counts_only_no_content() -> None:
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

    forget(store, entity_id=charlie.id, clock=clock, actor="u")

    events = store.events_affecting_entity(charlie.id)
    forget_event = next(e for e in events if e.op_type == "forget")
    # Payload may contain counts but must not contain any content strings
    payload_text = str(forget_event.payload)
    assert "marathons" not in payload_text
    assert forget_event.reversible_until is None


# ---------- restrict / unrestrict ----------


def test_restrict_changes_status() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="u")
    clock.advance()

    restrict(store, entity_id=charlie.id, clock=clock, actor="u")

    updated = store.get_entity(charlie.id)
    assert updated is not None
    assert updated.status is Status.RESTRICTED


def test_unrestrict_returns_to_active() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock)
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="u")
    clock.advance()
    restrict(store, entity_id=charlie.id, clock=clock, actor="u")
    clock.advance()

    unrestrict(store, entity_id=charlie.id, clock=clock, actor="u")

    updated = store.get_entity(charlie.id)
    assert updated is not None
    assert updated.status is Status.ACTIVE


def test_restricted_entity_excluded_from_identify() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock, vector=(1.0, 0.0, 0.0))
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="u")
    clock.advance()
    restrict(store, entity_id=charlie.id, clock=clock, actor="u")

    results = identify(store, encoder_id="insightface/arcface@0.7.3", vector=(1.0, 0.0, 0.0), k=5)
    assert all(c.entity.id != charlie.id for c in results)


# ---------- export ----------


def test_export_returns_entity_and_facts() -> None:
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

    dump = export(store, entity_id=charlie.id)

    assert dump["entity"]["id"] == charlie.id
    assert dump["entity"]["name"] == "Charlie"
    assert any(f["content"] == "runs marathons" for f in dump["facts"])


def test_export_excludes_embeddings_by_default() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock, vector=(0.1, 0.2, 0.3))
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="u")

    dump = export(store, entity_id=charlie.id)

    # Observations present but embeddings list empty unless flag is set
    assert len(dump["observations"]) == 1
    assert dump["embeddings"] == []


def test_export_includes_embeddings_when_flagged() -> None:
    store, clock = _fresh()
    obs = _put_face_obs(store, clock, vector=(0.1, 0.2, 0.3))
    charlie = label(store, [obs.id], "Charlie", clock=clock, actor="u")

    dump = export(store, entity_id=charlie.id, include_embeddings=True)

    assert len(dump["embeddings"]) == 1
    assert dump["embeddings"][0]["vector"] == [0.1, 0.2, 0.3]


# ---------- full-stack recall works after corrections ----------


def test_recall_after_merge_returns_combined_facts() -> None:
    store, clock = _fresh()
    obs1 = _put_face_obs(store, clock, source_hash="sha256:A")
    obs2 = _put_face_obs(store, clock, source_hash="sha256:B")
    a = label(store, [obs1.id], "A", clock=clock, actor="u")
    b = label(store, [obs2.id], "B", clock=clock, actor="u")
    remember(store, entity_id=a.id, content="first", source=Source.USER, clock=clock, actor="u")
    remember(store, entity_id=b.id, content="second", source=Source.USER, clock=clock, actor="u")
    clock.advance()

    winner = merge(store, [a.id, b.id], keep="oldest", clock=clock, actor="u")
    snapshot = recall(store, entity_id=winner.id)

    assert {f.content for f in snapshot.facts} == {"first", "second"}
