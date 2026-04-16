"""In-memory Store implementation used by ops-level tests.

Lives under ``tests/`` so it is not shipped in the published package. Covers
the full ``vemem.core.protocols.Store`` Protocol with plain dicts/lists and
pure-Python cosine similarity — correctness over performance.

Used by ``tests/core/test_ops_*.py`` and by Wave 3 tests (MCP, CLI, bridge).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from math import sqrt

from vemem.core.enums import Polarity, Status
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


class FakeStore:
    """In-memory Store. Not thread-safe; not performant; correct."""

    def __init__(self) -> None:
        self._observations: dict[str, Observation] = {}
        self._embeddings: dict[str, Embedding] = {}
        self._entities: dict[str, Entity] = {}
        self._bindings: list[Binding] = []
        self._facts: dict[str, Fact] = {}
        self._events: dict[str, Event] = {}
        self._relationships: dict[str, Relationship] = {}
        self._event_log: list[EventLog] = []
        self._next_event_id: int = 1
        self._closed: bool = False

    # ---- schema / lifecycle ----

    def schema_version(self) -> int:
        return 1

    def close(self) -> None:
        self._closed = True

    # ---- observations / embeddings ----

    def put_observation(self, obs: Observation) -> None:
        self._observations[obs.id] = obs

    def put_embedding(self, emb: Embedding) -> None:
        self._embeddings[emb.id] = emb

    def get_observation(self, observation_id: str) -> Observation | None:
        return self._observations.get(observation_id)

    def embeddings_for_observation(self, observation_id: str) -> list[Embedding]:
        return [e for e in self._embeddings.values() if e.observation_id == observation_id]

    # ---- entities ----

    def put_entity(self, entity: Entity) -> None:
        self._entities[entity.id] = entity

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def find_entity_by_name(self, name: str) -> Entity | None:
        for e in self._entities.values():
            if e.status is not Status.ACTIVE:
                continue
            if e.name == name or name in e.aliases:
                return e
        return None

    # ---- bindings ----

    def append_binding(self, binding: Binding) -> None:
        self._bindings.append(binding)

    def close_binding(self, binding_id: str, at: datetime) -> None:
        for i, b in enumerate(self._bindings):
            if b.id == binding_id and b.valid_to is None:
                self._bindings[i] = replace(b, valid_to=at)
                return
        raise KeyError(f"binding {binding_id} not open or not found")

    def get_binding(self, binding_id: str) -> Binding | None:
        for b in self._bindings:
            if b.id == binding_id:
                return b
        return None

    def current_positive_bindings(self, observation_id: str) -> list[Binding]:
        return [
            b
            for b in self._bindings
            if b.observation_id == observation_id
            and b.polarity is Polarity.POSITIVE
            and b.valid_to is None
        ]

    def bindings_for_entity(
        self, entity_id: str, *, include_negative: bool = False
    ) -> list[Binding]:
        return [
            b
            for b in self._bindings
            if b.entity_id == entity_id and (include_negative or b.polarity is Polarity.POSITIVE)
        ]

    # ---- facts / events / relationships ----

    def put_fact(self, fact: Fact) -> None:
        self._facts[fact.id] = fact

    def get_fact(self, fact_id: str) -> Fact | None:
        return self._facts.get(fact_id)

    def retract_fact(self, fact_id: str, at: datetime) -> None:
        fact = self._facts.get(fact_id)
        if fact is None:
            raise KeyError(f"fact {fact_id} not found")
        if fact.valid_to is not None:
            return
        self._facts[fact_id] = replace(fact, valid_to=at)

    def facts_for_entity(self, entity_id: str, *, active_only: bool = True) -> list[Fact]:
        return [
            f
            for f in self._facts.values()
            if f.entity_id == entity_id and (not active_only or f.valid_to is None)
        ]

    def put_event(self, event: Event) -> None:
        self._events[event.id] = event

    def events_for_entity(self, entity_id: str) -> list[Event]:
        return [e for e in self._events.values() if e.entity_id == entity_id]

    def put_relationship(self, rel: Relationship) -> None:
        self._relationships[rel.id] = rel

    def get_relationship(self, relationship_id: str) -> Relationship | None:
        return self._relationships.get(relationship_id)

    def relationships_for_entity(
        self, entity_id: str, *, active_only: bool = True
    ) -> list[Relationship]:
        return [
            r
            for r in self._relationships.values()
            if (r.from_entity_id == entity_id or r.to_entity_id == entity_id)
            and (not active_only or r.valid_to is None)
        ]

    # ---- event log ----

    def append_event_log(self, event: EventLog) -> EventLog:
        assigned = replace(event, id=self._next_event_id)
        self._event_log.append(assigned)
        self._next_event_id += 1
        return assigned

    def get_event_log(self, event_id: int) -> EventLog | None:
        for e in self._event_log:
            if e.id == event_id:
                return e
        return None

    def events_affecting_entity(self, entity_id: str) -> list[EventLog]:
        return [e for e in self._event_log if entity_id in e.affected_entity_ids]

    def list_events(
        self, *, actor: str | None = None, since: datetime | None = None
    ) -> list[EventLog]:
        return [
            e
            for e in self._event_log
            if (actor is None or e.actor == actor) and (since is None or e.at >= since)
        ]

    # ---- vector search ----

    def search_embeddings(
        self, *, encoder_id: str, vector: tuple[float, ...], k: int
    ) -> list[tuple[str, float]]:
        matches: list[tuple[str, float]] = []
        for emb in self._embeddings.values():
            if emb.encoder_id != encoder_id or len(emb.vector) != len(vector):
                continue
            matches.append((emb.observation_id, _cosine(emb.vector, vector)))
        matches.sort(key=lambda pair: -pair[1])
        return matches[:k]

    # ---- erasure ----

    def delete_entity_cascade(self, entity_id: str) -> dict[str, int]:
        counts = {
            "observations": 0,
            "embeddings": 0,
            "bindings": 0,
            "facts": 0,
            "events": 0,
            "relationships": 0,
        }

        bound_obs: set[str] = {
            b.observation_id
            for b in self._bindings
            if b.entity_id == entity_id and b.polarity is Polarity.POSITIVE
        }

        for obs_id in bound_obs:
            other_positive = [
                b
                for b in self._bindings
                if b.observation_id == obs_id
                and b.polarity is Polarity.POSITIVE
                and b.entity_id != entity_id
                and b.valid_to is None
            ]
            if other_positive:
                continue  # multi-bound observation survives
            if self._observations.pop(obs_id, None) is not None:
                counts["observations"] += 1
            emb_ids = [eid for eid, e in self._embeddings.items() if e.observation_id == obs_id]
            for eid in emb_ids:
                del self._embeddings[eid]
                counts["embeddings"] += 1

        before = len(self._bindings)
        self._bindings = [b for b in self._bindings if b.entity_id != entity_id]
        counts["bindings"] = before - len(self._bindings)

        fact_ids = [fid for fid, f in self._facts.items() if f.entity_id == entity_id]
        for fid in fact_ids:
            del self._facts[fid]
            counts["facts"] += 1

        event_ids = [eid for eid, e in self._events.items() if e.entity_id == entity_id]
        for eid in event_ids:
            del self._events[eid]
            counts["events"] += 1

        rel_ids = [
            rid
            for rid, r in self._relationships.items()
            if r.from_entity_id == entity_id or r.to_entity_id == entity_id
        ]
        for rid in rel_ids:
            del self._relationships[rid]
            counts["relationships"] += 1

        return counts

    def prune_versions(self, older_than: datetime) -> None:
        # FakeStore has no version history; no-op (matches spec — real stores
        # physically remove old versions here).
        return None


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
