"""Structural types for the pluggable layers.

These are ``typing.Protocol``s, not ABCs — implementations satisfy them by
shape, not inheritance. The real implementations live in
``vemem.storage`` and ``vemem.encoders``; an in-memory fake lives under
``tests/support`` for fast ops-level tests.

Keep these surfaces as small as the ops truly need. Additions here ripple into
every backend.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class Clock(Protocol):
    """Abstracts ``datetime.now()`` so tests can freeze time."""

    def now(self) -> datetime: ...


@runtime_checkable
class Encoder(Protocol):
    """Produces an embedding vector from an image crop.

    ``id`` must include encoder version — e.g. ``insightface/arcface@0.7.3`` —
    because encoder version is part of identity-of-evidence (spec §3.1a).
    """

    id: str
    dim: int

    def embed(self, image_crop: bytes) -> tuple[float, ...]: ...


@runtime_checkable
class Detector(Protocol):
    """Detects entity candidates in an image and returns bboxes.

    Detection and encoding are separate concerns: a detector finds regions,
    an encoder describes them.
    """

    id: str

    def detect(self, image_bytes: bytes) -> list[tuple[int, int, int, int]]: ...


class Store(Protocol):
    """The storage layer contract.

    Implementations: ``vemem.storage.lancedb_store.LanceDBStore`` (production),
    ``tests.support.fake_store.FakeStore`` (in-memory, test-only).

    Methods are grouped by concern: write (idempotent appends), read (queries
    into the current believed state), lifecycle (forget, restrict, prune).
    The signatures here are the minimal surface ops need — they are NOT the
    public API ops themselves (those live in ``vemem.core.ops``).
    """

    # ---- schema / lifecycle ----
    def schema_version(self) -> int: ...
    def close(self) -> None: ...

    # ---- write: observations / embeddings (§3.1, §3.1a) ----
    def put_observation(self, obs: Observation) -> None: ...
    def put_embedding(self, emb: Embedding) -> None: ...
    def get_observation(self, observation_id: str) -> Observation | None: ...

    # ---- write: entities (§3.2) ----
    def put_entity(self, entity: Entity) -> None: ...
    def get_entity(self, entity_id: str) -> Entity | None: ...
    def find_entity_by_name(self, name: str) -> Entity | None: ...

    # ---- write: bindings (§3.3, bi-temporal append-only) ----
    def append_binding(self, binding: Binding) -> None: ...
    def close_binding(self, binding_id: str, at: datetime) -> None: ...
    def current_positive_bindings(self, observation_id: str) -> list[Binding]: ...
    def bindings_for_entity(
        self, entity_id: str, *, include_negative: bool = False
    ) -> list[Binding]: ...

    # ---- write: facts / events / relationships (§3.4 - §3.6) ----
    def put_fact(self, fact: Fact) -> None: ...
    def retract_fact(self, fact_id: str, at: datetime) -> None: ...
    def facts_for_entity(self, entity_id: str, *, active_only: bool = True) -> list[Fact]: ...

    def put_event(self, event: Event) -> None: ...
    def events_for_entity(self, entity_id: str) -> list[Event]: ...

    def put_relationship(self, rel: Relationship) -> None: ...
    def relationships_for_entity(
        self, entity_id: str, *, active_only: bool = True
    ) -> list[Relationship]: ...

    # ---- write: event log (§3.7) ----
    def append_event_log(self, event: EventLog) -> EventLog:
        """Persist an event and return it with the store-assigned id.

        The ``event.id`` field of the input is ignored; the store assigns a
        monotonic id and returns the full row so ops can reference the new id
        without a follow-up fetch.
        """
        ...

    def get_event_log(self, event_id: int) -> EventLog | None: ...
    def events_affecting_entity(self, entity_id: str) -> list[EventLog]: ...

    # ---- read: vector search (§4.0) ----
    def search_embeddings(
        self, *, encoder_id: str, vector: tuple[float, ...], k: int
    ) -> list[tuple[str, float]]:
        """Return ``[(observation_id, cosine_similarity), ...]``.

        Implementations that need index refresh before search MUST handle it
        here; callers should not see stale results for freshly-written rows.
        """
        ...

    # ---- lifecycle: erasure (§4.5) ----
    def delete_entity_cascade(self, entity_id: str) -> dict[str, int]:
        """Hard delete everything tied to this entity. Returns counts."""
        ...

    def prune_versions(self, older_than: datetime) -> None:
        """Physically remove historical rows older than ``older_than``.

        Required for GDPR-compliant forget — without it, old versions of
        deleted rows remain recoverable.
        """
        ...
