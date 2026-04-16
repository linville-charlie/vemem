"""JSON-serializable dict representations of core domain dataclasses.

The MCP wire format requires tool inputs and outputs to be JSON; core types
are frozen dataclasses with enums and ``datetime`` fields that don't survive
``json.dumps`` without help. Centralizing serialization here keeps tool
handlers in ``tools.py`` focused on op invocation rather than field shuffling,
and means a schema change only touches one module.

These helpers intentionally mirror the ``_*_dict`` helpers in ``core.ops``
but live in the MCP layer so the core module keeps its zero-dependency shape
(no json, no pydantic). Callers that want a structured Pydantic schema should
use the ``*Payload`` models in ``tools.py``; this module is for the pure-dict
representation the SDK serializes.
"""

from __future__ import annotations

from typing import Any

from vemem.core.ops import RecallSnapshot
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
)


def entity_to_dict(e: Entity) -> dict[str, Any]:
    return {
        "id": e.id,
        "kind": e.kind.value,
        "name": e.name,
        "aliases": list(e.aliases),
        "modality": e.modality.value,
        "status": e.status.value,
        "merged_into_id": e.merged_into_id,
        "created_at": e.created_at.isoformat(),
        "last_seen": e.last_seen.isoformat(),
    }


def observation_to_dict(o: Observation) -> dict[str, Any]:
    return {
        "id": o.id,
        "source_uri": o.source_uri,
        "source_hash": o.source_hash,
        "bbox": list(o.bbox),
        "detector_id": o.detector_id,
        "modality": o.modality.value,
        "detected_at": o.detected_at.isoformat(),
        "source_ts": o.source_ts.isoformat() if o.source_ts else None,
        "source_frame": o.source_frame,
    }


def embedding_to_dict(em: Embedding) -> dict[str, Any]:
    return {
        "id": em.id,
        "observation_id": em.observation_id,
        "encoder_id": em.encoder_id,
        "vector": list(em.vector),
        "dim": em.dim,
        "created_at": em.created_at.isoformat(),
        "key_id": em.key_id,
    }


def binding_to_dict(b: Binding) -> dict[str, Any]:
    return {
        "id": b.id,
        "observation_id": b.observation_id,
        "entity_id": b.entity_id,
        "polarity": b.polarity.value,
        "confidence": b.confidence,
        "method": b.method.value,
        "valid_from": b.valid_from.isoformat(),
        "valid_to": b.valid_to.isoformat() if b.valid_to else None,
        "recorded_at": b.recorded_at.isoformat(),
        "actor": b.actor,
    }


def fact_to_dict(f: Fact) -> dict[str, Any]:
    return {
        "id": f.id,
        "entity_id": f.entity_id,
        "content": f.content,
        "source": f.source.value,
        "actor": f.actor,
        "valid_from": f.valid_from.isoformat(),
        "valid_to": f.valid_to.isoformat() if f.valid_to else None,
        "recorded_at": f.recorded_at.isoformat(),
        "provenance_entity_id": f.provenance_entity_id,
    }


def event_to_dict(ev: Event) -> dict[str, Any]:
    return {
        "id": ev.id,
        "entity_id": ev.entity_id,
        "content": ev.content,
        "source": ev.source.value,
        "occurred_at": ev.occurred_at.isoformat(),
        "recorded_at": ev.recorded_at.isoformat(),
        "provenance_entity_id": ev.provenance_entity_id,
    }


def relationship_to_dict(r: Relationship) -> dict[str, Any]:
    return {
        "id": r.id,
        "from_entity_id": r.from_entity_id,
        "to_entity_id": r.to_entity_id,
        "relation_type": r.relation_type,
        "source": r.source.value,
        "valid_from": r.valid_from.isoformat(),
        "valid_to": r.valid_to.isoformat() if r.valid_to else None,
        "recorded_at": r.recorded_at.isoformat(),
        "provenance_from_id": r.provenance_from_id,
        "provenance_to_id": r.provenance_to_id,
    }


def event_log_to_dict(e: EventLog) -> dict[str, Any]:
    return {
        "id": e.id,
        "op_type": e.op_type,
        "payload": e.payload,
        "actor": e.actor,
        "affected_entity_ids": list(e.affected_entity_ids),
        "at": e.at.isoformat(),
        "reversible_until": e.reversible_until.isoformat() if e.reversible_until else None,
        "reversed_by": e.reversed_by,
    }


def candidate_to_dict(c: Candidate) -> dict[str, Any]:
    return {
        "entity": entity_to_dict(c.entity),
        "confidence": c.confidence,
        "matched_observation_ids": list(c.matched_observation_ids),
        "method": c.method.value,
        "facts": [fact_to_dict(f) for f in c.facts],
    }


def recall_snapshot_to_dict(snap: RecallSnapshot) -> dict[str, Any]:
    return {
        "entity": entity_to_dict(snap.entity),
        "facts": [fact_to_dict(f) for f in snap.facts],
    }
