"""PyArrow schemas for the LanceDB backend (Wave 2B, schema v1).

Each table mirrors one dataclass from :mod:`vemem.core.types`. The schemas are
intentionally explicit (no auto-inference) so a LanceDB upgrade or a fresh
install produces byte-identical column layouts — important for the prune step
of ``forget()`` which relies on version-file compatibility across reads.

Embeddings live in **one table per encoder_id** (see DONE.md rationale). The
vector column is a ``FixedSizeList<float32, dim>``; dim is fixed by the encoder.
This keeps the ANN-search fast path homogeneous and makes per-encoder cascade
delete + prune a single call.

All timestamps are ``timestamp[us, tz=UTC]`` — the core types use timezone-aware
``datetime`` (UTC), and pyarrow refuses naive↔aware cross-inserts.
"""

from __future__ import annotations

import re

import pyarrow as pa

# ---- shared helpers --------------------------------------------------------

TS = pa.timestamp("us", tz="UTC")


def sanitize_encoder_id(encoder_id: str) -> str:
    """Return a filesystem/table-name-safe slug for an encoder id.

    LanceDB table names allow ``[A-Za-z0-9_]``. Encoder ids like
    ``insightface/arcface@0.7.3`` contain ``/``, ``@``, and ``.`` — we replace
    each with ``_`` and lowercase. Collisions are extremely unlikely in
    practice but the table-registration path (``meta`` table) stores the
    original ``encoder_id`` as a column so the mapping is reversible.
    """

    return "embeddings__" + re.sub(r"[^A-Za-z0-9_]", "_", encoder_id).lower()


# ---- table schemas ---------------------------------------------------------


OBSERVATIONS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("source_uri", pa.string(), nullable=False),
        pa.field("source_hash", pa.string(), nullable=False),
        pa.field("bbox", pa.list_(pa.int64(), 4), nullable=False),
        pa.field("detector_id", pa.string(), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("detected_at", TS, nullable=False),
        pa.field("source_ts", TS, nullable=True),
        pa.field("source_frame", pa.int64(), nullable=True),
    ]
)


def embeddings_schema(dim: int) -> pa.Schema:
    """Schema for a per-encoder embeddings table. ``dim`` fixes the vector width."""

    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("observation_id", pa.string(), nullable=False),
            pa.field("encoder_id", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dim), nullable=False),
            pa.field("dim", pa.int64(), nullable=False),
            pa.field("created_at", TS, nullable=False),
            pa.field("key_id", pa.string(), nullable=True),
        ]
    )


ENTITIES_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("created_at", TS, nullable=False),
        pa.field("last_seen", TS, nullable=False),
        pa.field("aliases", pa.list_(pa.string()), nullable=False),
        pa.field("merged_into_id", pa.string(), nullable=True),
    ]
)


BINDINGS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("observation_id", pa.string(), nullable=False),
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("method", pa.string(), nullable=False),
        pa.field("polarity", pa.string(), nullable=False),
        pa.field("valid_from", TS, nullable=False),
        pa.field("valid_to", TS, nullable=True),
        pa.field("recorded_at", TS, nullable=False),
        pa.field("actor", pa.string(), nullable=False),
    ]
)


FACTS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("actor", pa.string(), nullable=False),
        pa.field("valid_from", TS, nullable=False),
        pa.field("valid_to", TS, nullable=True),
        pa.field("recorded_at", TS, nullable=False),
        pa.field("provenance_entity_id", pa.string(), nullable=True),
    ]
)


EVENTS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("occurred_at", TS, nullable=False),
        pa.field("recorded_at", TS, nullable=False),
        pa.field("provenance_entity_id", pa.string(), nullable=True),
    ]
)


RELATIONSHIPS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("from_entity_id", pa.string(), nullable=False),
        pa.field("to_entity_id", pa.string(), nullable=False),
        pa.field("relation_type", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("valid_from", TS, nullable=False),
        pa.field("valid_to", TS, nullable=True),
        pa.field("recorded_at", TS, nullable=False),
        pa.field("provenance_from_id", pa.string(), nullable=True),
        pa.field("provenance_to_id", pa.string(), nullable=True),
    ]
)


EVENT_LOG_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("op_type", pa.string(), nullable=False),
        pa.field("payload_json", pa.string(), nullable=False),
        pa.field("actor", pa.string(), nullable=False),
        pa.field("affected_entity_ids", pa.list_(pa.string()), nullable=False),
        pa.field("at", TS, nullable=False),
        pa.field("reversible_until", TS, nullable=True),
        pa.field("reversed_by", pa.int64(), nullable=True),
    ]
)


# ``meta`` is a key-value settings table. Payload-as-JSON keeps it future-proof.
META_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("key", pa.string(), nullable=False),
        pa.field("value_json", pa.string(), nullable=False),
    ]
)


# ``encoders`` tracks each encoder_id that has an embeddings table created for
# it, so ``search_embeddings`` can answer "do we have any embeddings for this
# encoder?" without scanning every table.
ENCODERS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("encoder_id", pa.string(), nullable=False),
        pa.field("table_name", pa.string(), nullable=False),
        pa.field("dim", pa.int64(), nullable=False),
        pa.field("created_at", TS, nullable=False),
    ]
)


# ---- table names -----------------------------------------------------------


OBSERVATIONS_TABLE = "observations"
ENTITIES_TABLE = "entities"
BINDINGS_TABLE = "bindings"
FACTS_TABLE = "facts"
EVENTS_TABLE = "events"
RELATIONSHIPS_TABLE = "relationships"
EVENT_LOG_TABLE = "event_log"
META_TABLE = "meta"
ENCODERS_TABLE = "encoders"


ALL_FIXED_TABLES: tuple[tuple[str, pa.Schema], ...] = (
    (OBSERVATIONS_TABLE, OBSERVATIONS_SCHEMA),
    (ENTITIES_TABLE, ENTITIES_SCHEMA),
    (BINDINGS_TABLE, BINDINGS_SCHEMA),
    (FACTS_TABLE, FACTS_SCHEMA),
    (EVENTS_TABLE, EVENTS_SCHEMA),
    (RELATIONSHIPS_TABLE, RELATIONSHIPS_SCHEMA),
    (EVENT_LOG_TABLE, EVENT_LOG_SCHEMA),
    (META_TABLE, META_SCHEMA),
    (ENCODERS_TABLE, ENCODERS_SCHEMA),
)
