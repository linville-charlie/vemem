"""Frozen domain types — the shapes from spec §3.

Types are ``@dataclass(frozen=True, slots=True)`` everywhere. Mutation is a
storage concern; the in-memory objects are immutable values. Embeddings use
``tuple[float, ...]`` rather than ``numpy.ndarray`` so the core module has no
third-party dependencies and all types remain hashable.

Each type's docstring names the spec section it implements — update together.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from vemem.core.enums import Kind, Method, Modality, Polarity, Source, Status


def observation_id_for(
    source_hash: str,
    bbox: tuple[int, int, int, int],
    detector_id: str,
) -> str:
    """Return the deterministic content-hash id for an observation (§3.1).

    Re-observing the same ``(image, bbox, detector)`` returns the same id, which
    is what makes observation writes idempotent.
    """
    h = hashlib.sha256()
    h.update(source_hash.encode("utf-8"))
    h.update(b"|")
    h.update(repr(bbox).encode("utf-8"))
    h.update(b"|")
    h.update(detector_id.encode("utf-8"))
    return "obs_" + h.hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class Observation:
    """Spec §3.1 — one detection in one image at one region by one detector.

    Observations are immutable evidence. The library stores the image hash
    (not the bytes); callers own the image itself (§3.1b).
    """

    id: str
    source_uri: str
    source_hash: str
    bbox: tuple[int, int, int, int]
    detector_id: str
    modality: Modality
    detected_at: datetime
    source_ts: datetime | None = None
    source_frame: int | None = None


@dataclass(frozen=True, slots=True)
class Embedding:
    """Spec §3.1a — one vector per (observation, encoder) pair, append-only.

    Encoder version is part of ``encoder_id``; an encoder upgrade produces new
    rows, never overwrites. Vectors are expected L2-normalized at write time.
    """

    id: str
    observation_id: str
    encoder_id: str
    vector: tuple[float, ...]
    dim: int
    created_at: datetime
    key_id: str | None = None


@dataclass(frozen=True, slots=True)
class Entity:
    """Spec §3.2 — a stable-id interpretation over observations."""

    id: str
    kind: Kind
    name: str
    modality: Modality
    status: Status
    created_at: datetime
    last_seen: datetime
    aliases: tuple[str, ...] = ()
    merged_into_id: str | None = None


@dataclass(frozen=True, slots=True)
class Binding:
    """Spec §3.3 — bi-temporal observation → entity claim.

    ``valid_from`` / ``valid_to`` is the belief timeline; ``recorded_at`` is
    the system timeline. A binding's ``polarity`` may be positive ("obs IS
    entity") or negative ("obs is NOT entity").
    """

    id: str
    observation_id: str
    entity_id: str
    confidence: float
    method: Method
    valid_from: datetime
    recorded_at: datetime
    actor: str
    polarity: Polarity = Polarity.POSITIVE
    valid_to: datetime | None = None


@dataclass(frozen=True, slots=True)
class Fact:
    """Spec §3.4 — bi-temporal statement about an entity."""

    id: str
    entity_id: str
    content: str
    source: Source
    actor: str
    valid_from: datetime
    recorded_at: datetime
    valid_to: datetime | None = None
    provenance_entity_id: str | None = None


@dataclass(frozen=True, slots=True)
class Event:
    """Spec §3.5 — a timestamped occurrence involving an entity."""

    id: str
    entity_id: str
    content: str
    source: Source
    occurred_at: datetime
    recorded_at: datetime
    provenance_entity_id: str | None = None


@dataclass(frozen=True, slots=True)
class Relationship:
    """Spec §3.6 — a directed edge between two entities."""

    id: str
    from_entity_id: str
    to_entity_id: str
    relation_type: str
    source: Source
    valid_from: datetime
    recorded_at: datetime
    valid_to: datetime | None = None
    provenance_from_id: str | None = None
    provenance_to_id: str | None = None


@dataclass(frozen=True, slots=True)
class EventLog:
    """Spec §3.7 — audit trail + undo stack.

    Never stores embeddings or image bytes. ``affected_entity_ids`` is the
    denormalized index that makes "show me everything that touched Charlie"
    queryable without scanning payloads.
    """

    id: int
    op_type: str
    payload: dict[str, Any]
    actor: str
    affected_entity_ids: tuple[str, ...]
    at: datetime
    reversible_until: datetime | None = None
    reversed_by: int | None = None


@dataclass(frozen=True, slots=True)
class Candidate:
    """Return value of ``identify()`` — a ranked match against the gallery."""

    entity: Entity
    confidence: float
    matched_observation_ids: tuple[str, ...]
    method: Method
    facts: tuple[Fact, ...] = field(default_factory=tuple)
