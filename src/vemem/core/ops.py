"""Top-level operations on a visual entity memory store.

Implements the semantics in ``docs/spec/identity-semantics.md`` §4. Each op is
a module-level function that takes its dependencies (Store, Clock, actor)
explicitly — no global state, no hidden singletons — so ops are trivially
testable against ``FakeStore`` and swap backends without code changes.

Conventions:
- Writes always close old bindings with ``valid_to = clock.now()`` before
  appending new ones (bi-temporal, §3.3).
- Cross-table operations emit the EventLog entry LAST, after all data writes
  have succeeded (§6 crash recovery hooks).
- All errors inherit from ``vemem.core.errors.VemError``.

Ops in this file: ``identify``, ``label``, ``remember``, ``recall``. The
``relabel``, ``merge``, ``split``, ``forget``, ``restrict``, ``unrestrict``,
``export``, and ``undo`` ops land in the next commit (Wave 2A.2).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from typing import TYPE_CHECKING

from vemem.core.enums import Kind, Method, Modality, OpType, Polarity, Source, Status
from vemem.core.errors import EntityUnavailableError, ModalityMismatchError
from vemem.core.ids import new_id
from vemem.core.types import Binding, Candidate, Entity, EventLog, Fact

if TYPE_CHECKING:
    from vemem.core.protocols import Clock, Store

DEFAULT_UNDO_WINDOW = timedelta(days=30)
DEFAULT_MIN_CONFIDENCE = 0.5


@dataclass(frozen=True, slots=True)
class RecallSnapshot:
    """Return value of ``recall()`` — an entity plus its active knowledge."""

    entity: Entity
    facts: tuple[Fact, ...]


# ---------- internal helpers ----------


def _resolve_entity(store: Store, name_or_id: str) -> Entity | None:
    """Look up an entity by id first, then by name/alias."""
    by_id = store.get_entity(name_or_id)
    if by_id is not None:
        return by_id
    return store.find_entity_by_name(name_or_id)


def _require_active(entity: Entity) -> None:
    if entity.status is not Status.ACTIVE:
        raise EntityUnavailableError(f"entity {entity.id} is {entity.status.value}, not active")


def _observation_modalities(store: Store, observation_ids: list[str]) -> set[Modality]:
    modalities: set[Modality] = set()
    for oid in observation_ids:
        obs = store.get_observation(oid)
        if obs is not None:
            modalities.add(obs.modality)
    return modalities


def _make_entity_id() -> str:
    return "ent_" + new_id()


def _make_binding_id() -> str:
    return "bnd_" + new_id()


def _make_fact_id() -> str:
    return "fact_" + new_id()


# ---------- identify (§4.0) ----------


def identify(
    store: Store,
    *,
    encoder_id: str,
    vector: tuple[float, ...],
    k: int = 5,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    prefer: str = "instance",
) -> list[Candidate]:
    """Return ranked candidate entities that match the query vector.

    Read-only. Uses the store's ANN search, filters by ``min_confidence``,
    respects negative bindings (an entity with a negative binding against the
    *exact observation* we matched is excluded), and resolves each matched
    observation to its current positive binding's entity.

    ``prefer`` biases ranking when both instance and type entities match —
    ``"instance"`` (default) / ``"type"`` / ``"both"``.
    """
    raw = store.search_embeddings(encoder_id=encoder_id, vector=vector, k=max(k * 4, k))

    # Aggregate by entity: max confidence across its matched observations.
    per_entity: dict[str, dict[str, object]] = {}
    for observation_id, similarity in raw:
        if similarity < min_confidence:
            continue
        for binding in store.current_positive_bindings(observation_id):
            # Negative binding check — if the SAME observation has a negative
            # binding against the same entity, it would already be absent from
            # current_positive_bindings. But the entity could also be
            # negative-bound via an explicit user reject on this observation
            # even without a positive binding. Guard explicitly.
            if _observation_rejects_entity(store, observation_id, binding.entity_id):
                continue
            entity = store.get_entity(binding.entity_id)
            if entity is None or entity.status is not Status.ACTIVE:
                continue
            bucket = per_entity.setdefault(
                entity.id,
                {
                    "entity": entity,
                    "confidence": similarity,
                    "obs": [observation_id],
                    "method": binding.method,
                },
            )
            if similarity > bucket["confidence"]:  # type: ignore[operator]
                bucket["confidence"] = similarity
            obs_list: list[str] = bucket["obs"]  # type: ignore[assignment]
            if observation_id not in obs_list:
                obs_list.append(observation_id)

    candidates: list[Candidate] = []
    for data in per_entity.values():
        entity = data["entity"]  # type: ignore[assignment]
        assert isinstance(entity, Entity)
        candidates.append(
            Candidate(
                entity=entity,
                confidence=float(data["confidence"]),  # type: ignore[arg-type]
                matched_observation_ids=tuple(data["obs"]),  # type: ignore[arg-type]
                method=data["method"],  # type: ignore[arg-type]
            )
        )

    kind_preference = {
        "instance": (Kind.INSTANCE, Kind.TYPE),
        "type": (Kind.TYPE, Kind.INSTANCE),
        "both": None,
    }.get(prefer)

    def sort_key(c: Candidate) -> tuple[int, float]:
        if kind_preference is None:
            return (0, -c.confidence)
        rank = kind_preference.index(c.entity.kind)
        return (rank, -c.confidence)

    candidates.sort(key=sort_key)
    return candidates[:k]


def _observation_rejects_entity(store: Store, observation_id: str, entity_id: str) -> bool:
    """True if ``observation_id`` carries an active negative binding to ``entity_id``."""
    for b in store.bindings_for_entity(entity_id, include_negative=True):
        if (
            b.observation_id == observation_id
            and b.polarity is Polarity.NEGATIVE
            and b.valid_to is None
        ):
            return True
    return False


# ---------- label (§4.1) ----------


def label(
    store: Store,
    observation_ids: list[str],
    entity_name_or_id: str,
    *,
    clock: Clock,
    actor: str,
) -> Entity:
    """Commit a user-authoritative positive binding.

    Creates the entity if ``entity_name_or_id`` is new. Closes any prior
    current positive binding on the named observations (the supersede step
    from §3.3). Emits an EventLog entry.
    """
    now = clock.now()

    modalities = _observation_modalities(store, observation_ids)
    if len(modalities) > 1:
        raise ModalityMismatchError(f"observations span modalities {[m.value for m in modalities]}")

    existing = _resolve_entity(store, entity_name_or_id)
    entity_was_created = False
    if existing is None:
        modality = next(iter(modalities), Modality.FACE)
        entity = Entity(
            id=_make_entity_id(),
            kind=Kind.INSTANCE,
            name=entity_name_or_id,
            modality=modality,
            status=Status.ACTIVE,
            created_at=now,
            last_seen=now,
        )
        store.put_entity(entity)
        entity_was_created = True
    else:
        _require_active(existing)
        if modalities and existing.modality not in modalities:
            raise ModalityMismatchError(
                f"entity {existing.id} is {existing.modality.value}; "
                f"observations are {[m.value for m in modalities]}"
            )
        entity = existing

    closed_binding_ids: list[str] = []
    new_binding_ids: list[str] = []
    for obs_id in observation_ids:
        for prior in store.current_positive_bindings(obs_id):
            store.close_binding(prior.id, at=now)
            closed_binding_ids.append(prior.id)
        binding = Binding(
            id=_make_binding_id(),
            observation_id=obs_id,
            entity_id=entity.id,
            confidence=1.0,
            method=Method.USER_LABEL,
            valid_from=now,
            recorded_at=now,
            actor=actor,
            polarity=Polarity.POSITIVE,
            valid_to=None,
        )
        store.append_binding(binding)
        new_binding_ids.append(binding.id)

    # last_seen update on the entity
    if entity.last_seen != now:
        entity = replace(entity, last_seen=now)
        store.put_entity(entity)

    store.append_event_log(
        EventLog(
            id=0,
            op_type=OpType.LABEL.value,
            payload={
                "entity_id": entity.id,
                "entity_was_created": entity_was_created,
                "observation_ids": list(observation_ids),
                "closed_binding_ids": closed_binding_ids,
                "new_binding_ids": new_binding_ids,
            },
            actor=actor,
            affected_entity_ids=(entity.id,),
            at=now,
            reversible_until=now + DEFAULT_UNDO_WINDOW,
        )
    )

    return entity


# ---------- remember / recall (§3.4, §4.9 knowledge layer) ----------


def remember(
    store: Store,
    *,
    entity_id: str,
    content: str,
    source: Source,
    clock: Clock,
    actor: str,
) -> Fact:
    """Attach a fact to an entity. Bi-temporal: stays valid until retracted."""
    now = clock.now()
    entity = store.get_entity(entity_id)
    if entity is None:
        raise EntityUnavailableError(f"entity {entity_id} not found")
    _require_active(entity)

    fact = Fact(
        id=_make_fact_id(),
        entity_id=entity_id,
        content=content,
        source=source,
        actor=actor,
        valid_from=now,
        recorded_at=now,
    )
    store.put_fact(fact)

    store.append_event_log(
        EventLog(
            id=0,
            op_type=OpType.REMEMBER.value,
            payload={"entity_id": entity_id, "fact_id": fact.id},
            actor=actor,
            affected_entity_ids=(entity_id,),
            at=now,
            reversible_until=now + DEFAULT_UNDO_WINDOW,
        )
    )
    return fact


def recall(store: Store, *, entity_id: str) -> RecallSnapshot:
    """Return the entity plus its currently-active facts."""
    entity = store.get_entity(entity_id)
    if entity is None:
        raise EntityUnavailableError(f"entity {entity_id} not found")

    facts = store.facts_for_entity(entity_id, active_only=True)
    return RecallSnapshot(entity=entity, facts=tuple(facts))
