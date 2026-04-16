"""Tests that exercise LanceDB-specific behavior the Protocol itself can't pin.

These are the behaviors FakeStore can't mirror — version history, physical
disk layout, cross-encoder table routing — but that the spec depends on.

The most important test here is
``test_forget_physically_removes_vectors_from_version_history``: it proves
that ``delete_entity_cascade + prune_versions`` leaves no recoverable vector
anywhere in the on-disk version history. Without that guarantee, the
library's GDPR Art. 17 claim in the README is false.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vemem.core.enums import Kind, Method, Modality, Polarity, Status
from vemem.core.errors import SchemaVersionError
from vemem.core.types import Binding, Embedding, Entity, Observation
from vemem.storage.lancedb_store import LanceDBStore
from vemem.storage.migrations import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    read_schema_version,
    write_schema_version,
)
from vemem.storage.schemas import (
    EVENT_LOG_TABLE,
    OBSERVATIONS_TABLE,
    sanitize_encoder_id,
)

T0 = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[LanceDBStore]:
    s = LanceDBStore(path=str(tmp_path / "db"))
    try:
        yield s
    finally:
        s.close()


# ---- schema version -------------------------------------------------------


def test_fresh_store_writes_current_schema_version(tmp_path: Path) -> None:
    LanceDBStore(path=str(tmp_path / "db"))

    import lancedb

    db = lancedb.connect(str(tmp_path / "db"))
    assert read_schema_version(db) == CURRENT_SCHEMA_VERSION


def test_store_rejects_newer_on_disk_schema(tmp_path: Path) -> None:
    """Opening a store whose on-disk version is newer than the code refuses."""

    # Bootstrap the store once so the meta table exists.
    LanceDBStore(path=str(tmp_path / "db"))
    import lancedb

    db = lancedb.connect(str(tmp_path / "db"))
    write_schema_version(db, CURRENT_SCHEMA_VERSION + 1)

    with pytest.raises(SchemaVersionError):
        LanceDBStore(path=str(tmp_path / "db"))


def test_schema_version_key_survives_reopen(tmp_path: Path) -> None:
    LanceDBStore(path=str(tmp_path / "db")).close()
    LanceDBStore(path=str(tmp_path / "db")).close()

    import lancedb

    db = lancedb.connect(str(tmp_path / "db"))
    assert read_schema_version(db) == CURRENT_SCHEMA_VERSION
    # Defensive: the key name matches the migration module's constant.
    assert SCHEMA_VERSION_KEY == "schema_version"


# ---- encoder-per-table layout --------------------------------------------


def test_embeddings_split_by_encoder_into_separate_tables(
    store: LanceDBStore,
) -> None:
    """Different encoder_ids land in different LanceDB tables (see DONE.md)."""

    obs = Observation(
        id="obs_1",
        source_uri="f",
        source_hash="h",
        bbox=(0, 0, 1, 1),
        detector_id="det",
        modality=Modality.FACE,
        detected_at=T0,
    )
    store.put_observation(obs)

    arcface = Embedding(
        id="e_a",
        observation_id="obs_1",
        encoder_id="insightface/arcface@0.7.3",
        vector=(1.0, 0.0, 0.0),
        dim=3,
        created_at=T0,
    )
    clip = Embedding(
        id="e_c",
        observation_id="obs_1",
        encoder_id="clip/vit-b-32@2.30",
        vector=(0.5, 0.5, 0.5, 0.5),  # different dim!
        dim=4,
        created_at=T0,
    )
    store.put_embedding(arcface)
    store.put_embedding(clip)

    arcface_table = sanitize_encoder_id("insightface/arcface@0.7.3")
    clip_table = sanitize_encoder_id("clip/vit-b-32@2.30")

    assert store._table(arcface_table).count_rows() == 1
    assert store._table(clip_table).count_rows() == 1
    assert arcface_table != clip_table


def test_encoder_registry_roundtrips_across_reopen(tmp_path: Path) -> None:
    first = LanceDBStore(path=str(tmp_path / "db"))
    first.put_observation(
        Observation(
            id="obs_1",
            source_uri="f",
            source_hash="h",
            bbox=(0, 0, 1, 1),
            detector_id="det",
            modality=Modality.FACE,
            detected_at=T0,
        )
    )
    first.put_embedding(
        Embedding(
            id="e_1",
            observation_id="obs_1",
            encoder_id="insightface/arcface@0.7.3",
            vector=(1.0, 0.0, 0.0),
            dim=3,
            created_at=T0,
        )
    )
    first.close()

    second = LanceDBStore(path=str(tmp_path / "db"))
    try:
        results = second.search_embeddings(
            encoder_id="insightface/arcface@0.7.3",
            vector=(1.0, 0.0, 0.0),
            k=1,
        )
        assert len(results) == 1
    finally:
        second.close()


# ---- version history + prune-on-forget (spec §4.5) -----------------------


def test_forget_physically_removes_vectors_from_version_history(
    store: LanceDBStore,
) -> None:
    """Load-bearing GDPR test.

    After ``delete_entity_cascade`` followed by ``prune_versions``, LanceDB
    must not allow ``checkout(pre_forget_version)`` to recover the deleted
    embedding vectors. This is the spec §4.5 step 6 invariant.

    Test strategy:
      1. Create an entity with a positive binding to an observation whose
         embedding holds a recognizable vector.
      2. Record LanceDB's per-table versions (observations + the per-encoder
         embeddings table).
      3. Call ``delete_entity_cascade`` then ``prune_versions(now)``.
      4. Attempt ``table.checkout(pre_forget_version)`` — must raise
         ``ValueError`` because the old version files are physically gone.
      5. Assert the embeddings table no longer contains the observation's
         vector at the current version.
    """

    obs = Observation(
        id="obs_to_forget",
        source_uri="file:///tmp/charlie.jpg",
        source_hash="h1",
        bbox=(0, 0, 10, 10),
        detector_id="insightface/buffalo_l@0.7.3",
        modality=Modality.FACE,
        detected_at=T0,
    )
    emb = Embedding(
        id="emb_to_forget",
        observation_id="obs_to_forget",
        encoder_id="insightface/arcface@0.7.3",
        vector=(0.1, 0.2, 0.3, 0.4),
        dim=4,
        created_at=T0,
    )
    entity = Entity(
        id="ent_charlie",
        kind=entity_default_kind(),
        name="Charlie",
        modality=Modality.FACE,
        status=Status.ACTIVE,
        created_at=T0,
        last_seen=T0,
    )
    binding = Binding(
        id="bind_1",
        observation_id="obs_to_forget",
        entity_id="ent_charlie",
        confidence=1.0,
        method=Method.USER_LABEL,
        polarity=Polarity.POSITIVE,
        valid_from=T0,
        recorded_at=T0,
        actor="user:alice",
    )
    store.put_observation(obs)
    store.put_embedding(emb)
    store.put_entity(entity)
    store.append_binding(binding)

    emb_table_name = sanitize_encoder_id("insightface/arcface@0.7.3")
    emb_table = store._table(emb_table_name)
    obs_table = store._table(OBSERVATIONS_TABLE)

    pre_forget_emb_version = emb_table.version
    pre_forget_obs_version = obs_table.version
    # Sanity: the embedding is there pre-forget.
    assert emb_table.count_rows() == 1
    pre_arrow = emb_table.to_arrow().to_pylist()
    assert pre_arrow[0]["observation_id"] == "obs_to_forget"

    # Perform forget.
    counts = store.delete_entity_cascade("ent_charlie")
    assert counts["observations"] == 1
    assert counts["embeddings"] == 1
    assert counts["bindings"] == 1

    # Prune everything older than right now (spec §4.5 step 6).
    store.prune_versions(datetime.now(UTC))

    # Now try to checkout the pre-forget version. LanceDB must refuse —
    # the underlying version files have been physically deleted.
    with pytest.raises(ValueError) as err_emb:
        emb_table.checkout(pre_forget_emb_version)
    assert "cleaned up" in str(err_emb.value) or "no longer exists" in str(err_emb.value)

    with pytest.raises(ValueError) as err_obs:
        obs_table.checkout(pre_forget_obs_version)
    assert "cleaned up" in str(err_obs.value) or "no longer exists" in str(err_obs.value)

    # And the current state has zero rows for this observation.
    emb_table_fresh = store._table(emb_table_name)
    assert emb_table_fresh.count_rows() == 0

    obs_table_fresh = store._table(OBSERVATIONS_TABLE)
    post = obs_table_fresh.search().where("id = 'obs_to_forget'").limit(1).to_arrow()
    assert post.num_rows == 0


def test_prune_versions_older_than_future_is_safe(store: LanceDBStore) -> None:
    """``prune_versions`` with a timestamp in the future is clamped to now."""

    obs = Observation(
        id="obs_1",
        source_uri="f",
        source_hash="h",
        bbox=(0, 0, 1, 1),
        detector_id="det",
        modality=Modality.FACE,
        detected_at=T0,
    )
    store.put_observation(obs)

    # Future time → interpret as "prune everything". Should not raise.
    future = datetime.now(UTC) + timedelta(days=365)
    store.prune_versions(future)


# ---- event log integer ordering + reopen --------------------------------


def test_event_log_id_sequence_persists_across_reopen(tmp_path: Path) -> None:
    """After reopen, the next id continues from the max on disk."""

    from vemem.core.enums import OpType
    from vemem.core.types import EventLog

    first = LanceDBStore(path=str(tmp_path / "db"))
    e1 = first.append_event_log(
        EventLog(
            id=0,
            op_type=OpType.LABEL.value,
            payload={},
            actor="user:alice",
            affected_entity_ids=("ent_1",),
            at=T0,
        )
    )
    e2 = first.append_event_log(
        EventLog(
            id=0,
            op_type=OpType.LABEL.value,
            payload={},
            actor="user:alice",
            affected_entity_ids=("ent_1",),
            at=T0,
        )
    )
    assert e2.id == e1.id + 1
    first.close()

    second = LanceDBStore(path=str(tmp_path / "db"))
    try:
        e3 = second.append_event_log(
            EventLog(
                id=0,
                op_type=OpType.LABEL.value,
                payload={},
                actor="user:alice",
                affected_entity_ids=("ent_1",),
                at=T0,
            )
        )
        assert e3.id == e2.id + 1
    finally:
        second.close()


# ---- transactional ordering (spec §6) -----------------------------------


def test_cascade_delete_ordering_matches_spec(store: LanceDBStore) -> None:
    """Spec §6: cross-table ops execute bindings → facts → rel → entity.

    This test doesn't inspect ordering directly (LanceDB's per-table atomicity
    isn't observable from outside) but it verifies the end state is coherent:
    no dangling binding referring to a deleted observation, no orphan
    embedding pointing at a deleted observation.
    """

    from vemem.core.enums import Source
    from vemem.core.types import Fact

    store.put_entity(
        Entity(
            id="ent_1",
            kind=entity_default_kind(),
            name="Charlie",
            modality=Modality.FACE,
            status=Status.ACTIVE,
            created_at=T0,
            last_seen=T0,
        )
    )
    store.put_observation(
        Observation(
            id="obs_1",
            source_uri="f",
            source_hash="h",
            bbox=(0, 0, 1, 1),
            detector_id="det",
            modality=Modality.FACE,
            detected_at=T0,
        )
    )
    store.put_embedding(
        Embedding(
            id="emb_1",
            observation_id="obs_1",
            encoder_id="enc_a",
            vector=(1.0, 0.0),
            dim=2,
            created_at=T0,
        )
    )
    store.append_binding(
        Binding(
            id="bind_1",
            observation_id="obs_1",
            entity_id="ent_1",
            confidence=1.0,
            method=Method.USER_LABEL,
            polarity=Polarity.POSITIVE,
            valid_from=T0,
            recorded_at=T0,
            actor="user:alice",
        )
    )
    store.put_fact(
        Fact(
            id="fact_1",
            entity_id="ent_1",
            content="runs",
            source=Source.USER,
            actor="user:alice",
            valid_from=T0,
            recorded_at=T0,
        )
    )

    counts = store.delete_entity_cascade("ent_1")

    assert counts["bindings"] == 1
    assert counts["facts"] == 1
    assert counts["observations"] == 1
    assert counts["embeddings"] == 1

    assert store.get_observation("obs_1") is None
    assert store.bindings_for_entity("ent_1") == []
    assert store.facts_for_entity("ent_1", active_only=False) == []


# ---- small helpers ------------------------------------------------------


def entity_default_kind() -> Kind:
    """Small helper to keep fixture entities tidy across the GDPR tests."""

    return Kind.INSTANCE


# ---- event log retrieval after reopen ----------------------------------


def test_event_log_payload_roundtrips_across_reopen(tmp_path: Path) -> None:
    """JSON-serialized payloads must survive a store close/reopen."""

    from vemem.core.enums import OpType
    from vemem.core.types import EventLog

    first = LanceDBStore(path=str(tmp_path / "db"))
    ev = first.append_event_log(
        EventLog(
            id=0,
            op_type=OpType.FORGET.value,
            payload={
                "observations": 3,
                "embeddings": 3,
                "bindings": 5,
                "notes": "cleanup complete",
            },
            actor="user:alice",
            affected_entity_ids=("ent_1",),
            at=T0,
        )
    )
    first.close()

    second = LanceDBStore(path=str(tmp_path / "db"))
    try:
        got = second.get_event_log(ev.id)
        assert got is not None
        assert got.payload == {
            "observations": 3,
            "embeddings": 3,
            "bindings": 5,
            "notes": "cleanup complete",
        }
        # Table still holds exactly one row for this id.
        t = second._table(EVENT_LOG_TABLE)
        assert t.count_rows() == 1
    finally:
        second.close()
