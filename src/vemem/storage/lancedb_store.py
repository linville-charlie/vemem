"""LanceDB-backed :class:`vemem.core.protocols.Store` implementation.

Design choices (see ``DONE.md`` for full rationale):

- **One embeddings table per encoder_id.** Different encoders have different
  dimensions, and LanceDB's ``FixedSizeList`` column width is per-table. Split
  by encoder we get a tight vector column, simple ANN search, and clean
  per-encoder forget+prune. Price: creating a second encoder's table is a
  runtime event we handle on first write.
- **Payload-as-JSON in EventLog.** The ``payload`` on :class:`EventLog` is an
  open ``dict[str, Any]``; LanceDB's pyarrow schemas want a fixed shape.
  Serializing to JSON text keeps the schema simple and avoids churn as ops
  add new payload keys.
- **No index-on-write.** v0 uses brute-force ANN (LanceDB does a full scan when
  no index exists). For personal-scale galleries (<10k vectors) this is fast
  enough and dodges the "index is stale for 30s after write" gotcha.
- **All timestamps are UTC-aware.** Core types pass tz-aware ``datetime``;
  pyarrow rejects naive↔aware mixes, so the schema is ``timestamp[us, tz=UTC]``
  everywhere and the store preserves tz on readback.

This module deliberately does NOT depend on anything under :mod:`vemem.core`
except types, enums, errors, and the ``Store`` Protocol. Core ops talk to it
through that Protocol alone.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import lancedb
import pyarrow as pa

from vemem.core.enums import (
    Kind,
    Method,
    Modality,
    Polarity,
    Source,
    Status,
)
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
from vemem.storage.migrations import (
    CURRENT_SCHEMA_VERSION,
    check_schema_compat,
    read_schema_version,
    write_schema_version,
)
from vemem.storage.schemas import (
    ALL_FIXED_TABLES,
    BINDINGS_TABLE,
    ENCODERS_TABLE,
    ENTITIES_TABLE,
    EVENT_LOG_TABLE,
    EVENTS_TABLE,
    FACTS_TABLE,
    OBSERVATIONS_TABLE,
    RELATIONSHIPS_TABLE,
    TS,
    embeddings_schema,
    sanitize_encoder_id,
)


def _default_path() -> Path:
    """Default storage directory, honoring ``VEMEM_HOME``.

    Falls back to ``~/.vemem`` to match the project-wide convention. We do NOT
    create the directory here — :class:`LanceDBStore` does it lazily on open.
    """

    override = os.environ.get("VEMEM_HOME")
    if override:
        return Path(override)
    return Path.home() / ".vemem"


def _to_utc(ts: datetime | None) -> datetime | None:
    """Coerce a datetime to UTC-aware form.

    PyArrow's ``timestamp[us, tz=UTC]`` column refuses naive datetimes. Core
    types are documented as UTC-aware; we still defend with a best-effort
    conversion so test code writing naive ``datetime.now()`` doesn't panic.
    """

    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


class LanceDBStore:
    """Concrete :class:`~vemem.core.protocols.Store` backed by LanceDB.

    Parameters
    ----------
    path:
        Directory for the LanceDB dataset. Defaults to ``$VEMEM_HOME`` or
        ``~/.vemem``. Created on first use.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._path = Path(path) if path is not None else _default_path()
        self._path.mkdir(parents=True, exist_ok=True)

        self._db = lancedb.connect(str(self._path))
        self._init_fixed_tables()
        self._ensure_schema_version()
        # encoder_id → table name cache; populated from the ``encoders`` table
        # on open, extended on first write to a new encoder.
        self._encoder_tables: dict[str, str] = self._load_encoder_tables()
        self._next_event_id: int = self._compute_next_event_id()
        self._closed: bool = False

    # ---- init helpers ------------------------------------------------------

    def _init_fixed_tables(self) -> None:
        existing = set(self._db.table_names())
        for name, schema in ALL_FIXED_TABLES:
            if name not in existing:
                self._db.create_table(name, schema=schema, mode="create")

    def _ensure_schema_version(self) -> None:
        on_disk = read_schema_version(self._db)
        check_schema_compat(on_disk)
        if on_disk is None:
            write_schema_version(self._db, CURRENT_SCHEMA_VERSION)

    def _load_encoder_tables(self) -> dict[str, str]:
        table = self._db.open_table(ENCODERS_TABLE)
        arr = table.to_arrow()
        result: dict[str, str] = {}
        for row in arr.to_pylist():
            result[row["encoder_id"]] = row["table_name"]
        return result

    def _compute_next_event_id(self) -> int:
        table = self._db.open_table(EVENT_LOG_TABLE)
        arr = table.to_arrow()
        if arr.num_rows == 0:
            return 1
        max_id = int(pa.compute.max(arr.column("id")).as_py())
        return max_id + 1

    # ---- schema / lifecycle -----------------------------------------------

    def schema_version(self) -> int:
        return CURRENT_SCHEMA_VERSION

    def close(self) -> None:
        self._closed = True

    @property
    def path(self) -> Path:
        return self._path

    # ---- observations / embeddings ----------------------------------------

    def put_observation(self, obs: Observation) -> None:
        table = self._db.open_table(OBSERVATIONS_TABLE)
        # Idempotent: if the id already exists, treat as no-op (observations
        # are content-addressed and immutable per spec §3.1).
        existing = table.search().where(f"id = '{obs.id}'").limit(1).to_arrow()
        if existing.num_rows > 0:
            return
        table.add(
            [
                {
                    "id": obs.id,
                    "source_uri": obs.source_uri,
                    "source_hash": obs.source_hash,
                    "bbox": list(obs.bbox),
                    "detector_id": obs.detector_id,
                    "modality": obs.modality.value,
                    "detected_at": _to_utc(obs.detected_at),
                    "source_ts": _to_utc(obs.source_ts),
                    "source_frame": obs.source_frame,
                }
            ]
        )

    def get_observation(self, observation_id: str) -> Observation | None:
        table = self._db.open_table(OBSERVATIONS_TABLE)
        arr = table.search().where(f"id = '{observation_id}'").limit(1).to_arrow()
        if arr.num_rows == 0:
            return None
        row = arr.to_pylist()[0]
        return _row_to_observation(row)

    def put_embedding(self, emb: Embedding) -> None:
        table_name = self._ensure_encoder_table(emb.encoder_id, emb.dim)
        table = self._db.open_table(table_name)
        existing = table.search().where(f"id = '{emb.id}'").limit(1).to_arrow()
        if existing.num_rows > 0:
            return
        table.add(
            [
                {
                    "id": emb.id,
                    "observation_id": emb.observation_id,
                    "encoder_id": emb.encoder_id,
                    "vector": list(emb.vector),
                    "dim": emb.dim,
                    "created_at": _to_utc(emb.created_at),
                    "key_id": emb.key_id,
                }
            ]
        )

    def _ensure_encoder_table(self, encoder_id: str, dim: int) -> str:
        """Return the table name for ``encoder_id``, creating it if needed."""

        if encoder_id in self._encoder_tables:
            return self._encoder_tables[encoder_id]
        table_name = sanitize_encoder_id(encoder_id)
        if table_name not in self._db.table_names():
            self._db.create_table(table_name, schema=embeddings_schema(dim), mode="create")
        reg = self._db.open_table(ENCODERS_TABLE)
        reg.add(
            [
                {
                    "encoder_id": encoder_id,
                    "table_name": table_name,
                    "dim": dim,
                    "created_at": datetime.now(UTC),
                }
            ]
        )
        self._encoder_tables[encoder_id] = table_name
        return table_name

    # ---- entities ----------------------------------------------------------

    def put_entity(self, entity: Entity) -> None:
        table = self._db.open_table(ENTITIES_TABLE)
        # upsert: delete by id then add. LanceDB also has ``merge_insert``
        # but delete+add is simpler and avoids its quirks around mixed-type
        # nullable columns.
        table.delete(f"id = '{entity.id}'")
        table.add([_entity_to_row(entity)])

    def get_entity(self, entity_id: str) -> Entity | None:
        table = self._db.open_table(ENTITIES_TABLE)
        arr = table.search().where(f"id = '{entity_id}'").limit(1).to_arrow()
        if arr.num_rows == 0:
            return None
        return _row_to_entity(arr.to_pylist()[0])

    def find_entity_by_name(self, name: str) -> Entity | None:
        table = self._db.open_table(ENTITIES_TABLE)
        arr = table.search().where(f"status = '{Status.ACTIVE.value}'").to_arrow()
        for row in arr.to_pylist():
            if row["name"] == name or name in (row["aliases"] or ()):
                return _row_to_entity(row)
        return None

    # ---- bindings ----------------------------------------------------------

    def append_binding(self, binding: Binding) -> None:
        table = self._db.open_table(BINDINGS_TABLE)
        table.add([_binding_to_row(binding)])

    def close_binding(self, binding_id: str, at: datetime) -> None:
        table = self._db.open_table(BINDINGS_TABLE)
        arr = table.search().where(f"id = '{binding_id}'").limit(1).to_arrow()
        if arr.num_rows == 0:
            raise KeyError(f"binding {binding_id} not found")
        row = arr.to_pylist()[0]
        if row["valid_to"] is not None:
            raise KeyError(f"binding {binding_id} already closed")
        # Delete + re-add with valid_to set. Append-only audit is preserved
        # at the ops layer via the EventLog; the bindings table stores the
        # current effective state of each row.
        table.delete(f"id = '{binding_id}'")
        row["valid_to"] = _to_utc(at)
        row["valid_from"] = _to_utc(row["valid_from"])
        row["recorded_at"] = _to_utc(row["recorded_at"])
        table.add([row])

    def get_binding(self, binding_id: str) -> Binding | None:
        table = self._db.open_table(BINDINGS_TABLE)
        arr = table.search().where(f"id = '{binding_id}'").limit(1).to_arrow()
        if arr.num_rows == 0:
            return None
        return _row_to_binding(arr.to_pylist()[0])

    def current_positive_bindings(self, observation_id: str) -> list[Binding]:
        table = self._db.open_table(BINDINGS_TABLE)
        arr = (
            table.search()
            .where(
                f"observation_id = '{observation_id}' "
                f"AND polarity = '{Polarity.POSITIVE.value}' "
                "AND valid_to IS NULL"
            )
            .to_arrow()
        )
        return [_row_to_binding(r) for r in arr.to_pylist()]

    def bindings_for_entity(
        self, entity_id: str, *, include_negative: bool = False
    ) -> list[Binding]:
        table = self._db.open_table(BINDINGS_TABLE)
        clause = f"entity_id = '{entity_id}'"
        if not include_negative:
            clause += f" AND polarity = '{Polarity.POSITIVE.value}'"
        arr = table.search().where(clause).to_arrow()
        return [_row_to_binding(r) for r in arr.to_pylist()]

    # ---- facts / events / relationships -----------------------------------

    def put_fact(self, fact: Fact) -> None:
        table = self._db.open_table(FACTS_TABLE)
        table.delete(f"id = '{fact.id}'")
        table.add([_fact_to_row(fact)])

    def get_fact(self, fact_id: str) -> Fact | None:
        table = self._db.open_table(FACTS_TABLE)
        arr = table.search().where(f"id = '{fact_id}'").limit(1).to_arrow()
        if arr.num_rows == 0:
            return None
        return _row_to_fact(arr.to_pylist()[0])

    def retract_fact(self, fact_id: str, at: datetime) -> None:
        table = self._db.open_table(FACTS_TABLE)
        arr = table.search().where(f"id = '{fact_id}'").limit(1).to_arrow()
        if arr.num_rows == 0:
            raise KeyError(f"fact {fact_id} not found")
        row = arr.to_pylist()[0]
        if row["valid_to"] is not None:
            return
        table.delete(f"id = '{fact_id}'")
        row["valid_to"] = _to_utc(at)
        row["valid_from"] = _to_utc(row["valid_from"])
        row["recorded_at"] = _to_utc(row["recorded_at"])
        table.add([row])

    def facts_for_entity(self, entity_id: str, *, active_only: bool = True) -> list[Fact]:
        table = self._db.open_table(FACTS_TABLE)
        clause = f"entity_id = '{entity_id}'"
        if active_only:
            clause += " AND valid_to IS NULL"
        arr = table.search().where(clause).to_arrow()
        return [_row_to_fact(r) for r in arr.to_pylist()]

    def put_event(self, event: Event) -> None:
        table = self._db.open_table(EVENTS_TABLE)
        table.delete(f"id = '{event.id}'")
        table.add([_event_to_row(event)])

    def events_for_entity(self, entity_id: str) -> list[Event]:
        table = self._db.open_table(EVENTS_TABLE)
        arr = table.search().where(f"entity_id = '{entity_id}'").to_arrow()
        return [_row_to_event(r) for r in arr.to_pylist()]

    def put_relationship(self, rel: Relationship) -> None:
        table = self._db.open_table(RELATIONSHIPS_TABLE)
        table.delete(f"id = '{rel.id}'")
        table.add([_relationship_to_row(rel)])

    def get_relationship(self, relationship_id: str) -> Relationship | None:
        table = self._db.open_table(RELATIONSHIPS_TABLE)
        arr = table.search().where(f"id = '{relationship_id}'").limit(1).to_arrow()
        if arr.num_rows == 0:
            return None
        return _row_to_relationship(arr.to_pylist()[0])

    def relationships_for_entity(
        self, entity_id: str, *, active_only: bool = True
    ) -> list[Relationship]:
        table = self._db.open_table(RELATIONSHIPS_TABLE)
        clause = f"(from_entity_id = '{entity_id}' OR to_entity_id = '{entity_id}')"
        if active_only:
            clause += " AND valid_to IS NULL"
        arr = table.search().where(clause).to_arrow()
        return [_row_to_relationship(r) for r in arr.to_pylist()]

    # ---- event log ---------------------------------------------------------

    def append_event_log(self, event: EventLog) -> EventLog:
        assigned_id = self._next_event_id
        self._next_event_id += 1
        table = self._db.open_table(EVENT_LOG_TABLE)
        table.add(
            [
                {
                    "id": assigned_id,
                    "op_type": event.op_type,
                    "payload_json": json.dumps(event.payload, default=str),
                    "actor": event.actor,
                    "affected_entity_ids": list(event.affected_entity_ids),
                    "at": _to_utc(event.at),
                    "reversible_until": _to_utc(event.reversible_until),
                    "reversed_by": event.reversed_by,
                }
            ]
        )
        from dataclasses import replace

        return replace(event, id=assigned_id)

    def get_event_log(self, event_id: int) -> EventLog | None:
        table = self._db.open_table(EVENT_LOG_TABLE)
        arr = table.search().where(f"id = {event_id}").limit(1).to_arrow()
        if arr.num_rows == 0:
            return None
        return _row_to_event_log(arr.to_pylist()[0])

    def events_affecting_entity(self, entity_id: str) -> list[EventLog]:
        table = self._db.open_table(EVENT_LOG_TABLE)
        arr = table.to_arrow()
        result: list[EventLog] = []
        for row in arr.to_pylist():
            if entity_id in (row["affected_entity_ids"] or ()):
                result.append(_row_to_event_log(row))
        result.sort(key=lambda e: e.id)
        return result

    def list_events(
        self, *, actor: str | None = None, since: datetime | None = None
    ) -> list[EventLog]:
        table = self._db.open_table(EVENT_LOG_TABLE)
        arr = table.to_arrow()
        result: list[EventLog] = []
        for row in arr.to_pylist():
            event = _row_to_event_log(row)
            if actor is not None and event.actor != actor:
                continue
            if since is not None and event.at < since:
                continue
            result.append(event)
        result.sort(key=lambda e: e.id)
        return result

    # ---- vector search -----------------------------------------------------

    def search_embeddings(
        self, *, encoder_id: str, vector: tuple[float, ...], k: int
    ) -> list[tuple[str, float]]:
        """Return (observation_id, cosine_similarity) for the top-k matches.

        Returns ``[]`` if no table exists for ``encoder_id`` — cross-encoder
        callers must raise ``NoCompatibleEncoderError`` at a higher layer.
        """

        table_name = self._encoder_tables.get(encoder_id)
        if table_name is None:
            return []
        table = self._db.open_table(table_name)
        if table.count_rows() == 0:
            return []
        results = table.search(list(vector)).distance_type("cosine").limit(k).to_arrow()
        pairs: list[tuple[str, float]] = []
        for row in results.to_pylist():
            # LanceDB returns cosine *distance* in ``_distance``; similarity is
            # ``1 - distance``.
            distance = float(row["_distance"])
            similarity = 1.0 - distance
            pairs.append((row["observation_id"], similarity))
        return pairs

    # ---- erasure -----------------------------------------------------------

    def delete_entity_cascade(self, entity_id: str) -> dict[str, int]:
        """Hard-delete per spec §4.5. Returns per-table counts removed."""

        counts: dict[str, int] = {
            "observations": 0,
            "embeddings": 0,
            "bindings": 0,
            "facts": 0,
            "events": 0,
            "relationships": 0,
        }

        # 1. Observations bound only to this entity → delete obs + embeddings.
        bindings_table = self._db.open_table(BINDINGS_TABLE)
        bound = (
            bindings_table.search()
            .where(f"entity_id = '{entity_id}' AND polarity = '{Polarity.POSITIVE.value}'")
            .to_arrow()
        )
        bound_obs_ids = {r["observation_id"] for r in bound.to_pylist()}
        obs_table = self._db.open_table(OBSERVATIONS_TABLE)
        for obs_id in bound_obs_ids:
            others = (
                bindings_table.search()
                .where(
                    f"observation_id = '{obs_id}' "
                    f"AND entity_id != '{entity_id}' "
                    f"AND polarity = '{Polarity.POSITIVE.value}' "
                    "AND valid_to IS NULL"
                )
                .limit(1)
                .to_arrow()
            )
            if others.num_rows > 0:
                continue  # multi-bound observation survives
            # Delete observation.
            existing_obs = obs_table.search().where(f"id = '{obs_id}'").limit(1).to_arrow()
            if existing_obs.num_rows > 0:
                obs_table.delete(f"id = '{obs_id}'")
                counts["observations"] += 1
            # Delete embeddings across every encoder table.
            for table_name in self._encoder_tables.values():
                emb_table = self._db.open_table(table_name)
                matches = emb_table.search().where(f"observation_id = '{obs_id}'").to_arrow()
                if matches.num_rows > 0:
                    counts["embeddings"] += matches.num_rows
                    emb_table.delete(f"observation_id = '{obs_id}'")

        # 2. Delete all bindings (positive + negative) for this entity.
        before = bindings_table.count_rows()
        bindings_table.delete(f"entity_id = '{entity_id}'")
        counts["bindings"] = before - bindings_table.count_rows()

        # 3. Facts.
        facts_table = self._db.open_table(FACTS_TABLE)
        before = facts_table.count_rows()
        facts_table.delete(f"entity_id = '{entity_id}'")
        counts["facts"] = before - facts_table.count_rows()

        # 4. Events.
        events_table = self._db.open_table(EVENTS_TABLE)
        before = events_table.count_rows()
        events_table.delete(f"entity_id = '{entity_id}'")
        counts["events"] = before - events_table.count_rows()

        # 5. Relationships (either endpoint).
        rel_table = self._db.open_table(RELATIONSHIPS_TABLE)
        before = rel_table.count_rows()
        rel_table.delete(f"from_entity_id = '{entity_id}' OR to_entity_id = '{entity_id}'")
        counts["relationships"] = before - rel_table.count_rows()

        return counts

    def prune_versions(self, older_than: datetime) -> None:
        """Physically drop LanceDB version history for every relevant table.

        Calling ``table.optimize(cleanup_older_than=delta)`` asks LanceDB to
        remove on-disk version files whose write time is older than
        ``now - delta``. Passing a negative / zero ``delta`` (i.e.
        ``older_than >= now``) effectively means "prune everything older than
        right-now," which wipes the historical fragments that hold the deleted
        rows. This is the load-bearing GDPR Art. 17 step (spec §4.5).

        ``delete_unverified=True`` ensures even partially-written version
        files from a crashed optimize are cleared.
        """

        # Normalize to UTC and compute "how long ago" from now.
        older_than_utc = _to_utc(older_than)
        assert older_than_utc is not None
        now = datetime.now(UTC)
        # If caller asks to prune "older than future-time", treat as "prune
        # everything" — LanceDB wants a non-negative duration measured from
        # now backwards.
        delta = now - older_than_utc
        if delta < timedelta(0):
            delta = timedelta(0)

        for name in self._db.table_names():
            table = self._db.open_table(name)
            table.optimize(cleanup_older_than=delta, delete_unverified=True)

    # ---- debug / introspection --------------------------------------------

    def _table(self, name: str) -> Any:
        """Raw access for white-box tests only (not part of the Protocol)."""

        return self._db.open_table(name)


# ---- row → dataclass converters -------------------------------------------


def _row_to_observation(row: dict[str, Any]) -> Observation:
    bbox = row["bbox"]
    return Observation(
        id=row["id"],
        source_uri=row["source_uri"],
        source_hash=row["source_hash"],
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
        detector_id=row["detector_id"],
        modality=Modality(row["modality"]),
        detected_at=_as_utc(row["detected_at"]),
        source_ts=_as_utc(row["source_ts"]) if row["source_ts"] is not None else None,
        source_frame=row["source_frame"],
    )


def _entity_to_row(entity: Entity) -> dict[str, Any]:
    return {
        "id": entity.id,
        "kind": entity.kind.value,
        "name": entity.name,
        "modality": entity.modality.value,
        "status": entity.status.value,
        "created_at": _to_utc(entity.created_at),
        "last_seen": _to_utc(entity.last_seen),
        "aliases": list(entity.aliases),
        "merged_into_id": entity.merged_into_id,
    }


def _row_to_entity(row: dict[str, Any]) -> Entity:
    return Entity(
        id=row["id"],
        kind=Kind(row["kind"]),
        name=row["name"],
        modality=Modality(row["modality"]),
        status=Status(row["status"]),
        created_at=_as_utc(row["created_at"]),
        last_seen=_as_utc(row["last_seen"]),
        aliases=tuple(row["aliases"] or ()),
        merged_into_id=row["merged_into_id"],
    )


def _binding_to_row(binding: Binding) -> dict[str, Any]:
    return {
        "id": binding.id,
        "observation_id": binding.observation_id,
        "entity_id": binding.entity_id,
        "confidence": binding.confidence,
        "method": binding.method.value,
        "polarity": binding.polarity.value,
        "valid_from": _to_utc(binding.valid_from),
        "valid_to": _to_utc(binding.valid_to),
        "recorded_at": _to_utc(binding.recorded_at),
        "actor": binding.actor,
    }


def _row_to_binding(row: dict[str, Any]) -> Binding:
    return Binding(
        id=row["id"],
        observation_id=row["observation_id"],
        entity_id=row["entity_id"],
        confidence=row["confidence"],
        method=Method(row["method"]),
        polarity=Polarity(row["polarity"]),
        valid_from=_as_utc(row["valid_from"]),
        valid_to=_as_utc(row["valid_to"]) if row["valid_to"] is not None else None,
        recorded_at=_as_utc(row["recorded_at"]),
        actor=row["actor"],
    )


def _fact_to_row(fact: Fact) -> dict[str, Any]:
    return {
        "id": fact.id,
        "entity_id": fact.entity_id,
        "content": fact.content,
        "source": fact.source.value,
        "actor": fact.actor,
        "valid_from": _to_utc(fact.valid_from),
        "valid_to": _to_utc(fact.valid_to),
        "recorded_at": _to_utc(fact.recorded_at),
        "provenance_entity_id": fact.provenance_entity_id,
    }


def _row_to_fact(row: dict[str, Any]) -> Fact:
    return Fact(
        id=row["id"],
        entity_id=row["entity_id"],
        content=row["content"],
        source=Source(row["source"]),
        actor=row["actor"],
        valid_from=_as_utc(row["valid_from"]),
        valid_to=_as_utc(row["valid_to"]) if row["valid_to"] is not None else None,
        recorded_at=_as_utc(row["recorded_at"]),
        provenance_entity_id=row["provenance_entity_id"],
    )


def _event_to_row(event: Event) -> dict[str, Any]:
    return {
        "id": event.id,
        "entity_id": event.entity_id,
        "content": event.content,
        "source": event.source.value,
        "occurred_at": _to_utc(event.occurred_at),
        "recorded_at": _to_utc(event.recorded_at),
        "provenance_entity_id": event.provenance_entity_id,
    }


def _row_to_event(row: dict[str, Any]) -> Event:
    return Event(
        id=row["id"],
        entity_id=row["entity_id"],
        content=row["content"],
        source=Source(row["source"]),
        occurred_at=_as_utc(row["occurred_at"]),
        recorded_at=_as_utc(row["recorded_at"]),
        provenance_entity_id=row["provenance_entity_id"],
    )


def _relationship_to_row(rel: Relationship) -> dict[str, Any]:
    return {
        "id": rel.id,
        "from_entity_id": rel.from_entity_id,
        "to_entity_id": rel.to_entity_id,
        "relation_type": rel.relation_type,
        "source": rel.source.value,
        "valid_from": _to_utc(rel.valid_from),
        "valid_to": _to_utc(rel.valid_to),
        "recorded_at": _to_utc(rel.recorded_at),
        "provenance_from_id": rel.provenance_from_id,
        "provenance_to_id": rel.provenance_to_id,
    }


def _row_to_relationship(row: dict[str, Any]) -> Relationship:
    return Relationship(
        id=row["id"],
        from_entity_id=row["from_entity_id"],
        to_entity_id=row["to_entity_id"],
        relation_type=row["relation_type"],
        source=Source(row["source"]),
        valid_from=_as_utc(row["valid_from"]),
        valid_to=_as_utc(row["valid_to"]) if row["valid_to"] is not None else None,
        recorded_at=_as_utc(row["recorded_at"]),
        provenance_from_id=row["provenance_from_id"],
        provenance_to_id=row["provenance_to_id"],
    )


def _row_to_event_log(row: dict[str, Any]) -> EventLog:
    payload_raw = row.get("payload_json", "{}")
    payload = cast(dict[str, Any], json.loads(payload_raw))
    return EventLog(
        id=row["id"],
        op_type=row["op_type"],
        payload=payload,
        actor=row["actor"],
        affected_entity_ids=tuple(row["affected_entity_ids"] or ()),
        at=_as_utc(row["at"]),
        reversible_until=(
            _as_utc(row["reversible_until"]) if row["reversible_until"] is not None else None
        ),
        reversed_by=row["reversed_by"],
    )


def _as_utc(ts: datetime) -> datetime:
    """PyArrow gives back UTC-aware datetimes for tz=UTC columns; defensive cast."""

    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


# Schema sanity re-export so tests can reach the TS type if needed.
__all__ = ["TS", "LanceDBStore"]
