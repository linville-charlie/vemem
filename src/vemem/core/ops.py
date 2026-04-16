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
- Every reversible event records enough payload information for ``undo`` to
  reconstruct the pre-op state without reading the storage log twice.

Ops in this file: ``identify``, ``label``, ``relabel``, ``merge``, ``split``,
``remember``, ``recall``, ``forget``, ``restrict``, ``unrestrict``, ``export``.
``undo`` lives in ``ops_undo.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from vemem.core.enums import Kind, Method, Modality, OpType, Polarity, Source, Status
from vemem.core.errors import (
    EntityUnavailableError,
    KindMismatchError,
    ModalityMismatchError,
    OperationNotReversibleError,
)
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


# ---------- relabel (§4.2) ----------


def relabel(
    store: Store,
    observation_id: str,
    new_entity_name_or_id: str,
    *,
    clock: Clock,
    actor: str,
) -> Entity:
    """Move a single observation to a different entity.

    Sugar for ``label`` of one observation, plus a *negative* binding against
    the old entity so the auto-clusterer can never re-attach that observation.
    Name re-use does not trigger merge — identity is the entity id, not the
    name (§4.2 Apple-Photos-style rule).
    """
    now = clock.now()

    # Remember who the observation was previously bound to, before label moves it.
    prior_positive_bindings = store.current_positive_bindings(observation_id)
    prior_entity_ids = {b.entity_id for b in prior_positive_bindings}

    new_entity = label(store, [observation_id], new_entity_name_or_id, clock=clock, actor=actor)

    # Emit negative bindings against each prior entity so the clusterer cannot
    # re-merge this observation back.
    negative_binding_ids: list[str] = []
    for prior_id in prior_entity_ids:
        if prior_id == new_entity.id:
            continue
        neg = Binding(
            id=_make_binding_id(),
            observation_id=observation_id,
            entity_id=prior_id,
            confidence=1.0,
            method=Method.USER_REJECT,
            valid_from=now,
            recorded_at=now,
            actor=actor,
            polarity=Polarity.NEGATIVE,
            valid_to=None,
        )
        store.append_binding(neg)
        negative_binding_ids.append(neg.id)

    store.append_event_log(
        EventLog(
            id=0,
            op_type=OpType.RELABEL.value,
            payload={
                "observation_id": observation_id,
                "new_entity_id": new_entity.id,
                "prior_entity_ids": sorted(prior_entity_ids),
                "negative_binding_ids": negative_binding_ids,
            },
            actor=actor,
            affected_entity_ids=tuple(sorted({new_entity.id, *prior_entity_ids})),
            at=now,
            reversible_until=now + DEFAULT_UNDO_WINDOW,
        )
    )
    return new_entity


# ---------- merge (§4.3) ----------


def merge(
    store: Store,
    entity_ids: list[str],
    *,
    keep: str = "oldest",
    clock: Clock,
    actor: str,
) -> Entity:
    """Fold losers into a winner; facts migrate with provenance.

    Rejects modality or kind mismatches. Drops (doesn't migrate) negative
    bindings on losers. Retracts relationships that would become self-loops
    on the winner after merge. See spec §4.3 for the full recipe.
    """
    now = clock.now()
    if len(entity_ids) < 2:
        raise ValueError("merge requires at least two entity_ids")

    entities: list[Entity] = []
    for eid in entity_ids:
        e = store.get_entity(eid)
        if e is None:
            raise EntityUnavailableError(f"entity {eid} not found")
        _require_active(e)
        entities.append(e)

    modalities = {e.modality for e in entities}
    if len(modalities) > 1:
        raise ModalityMismatchError(
            f"cannot merge across modalities: {[m.value for m in modalities]}"
        )
    kinds = {e.kind for e in entities}
    if len(kinds) > 1:
        raise KindMismatchError(
            "cannot merge an instance with a type; use an instance_of relationship instead"
        )

    if keep == "oldest":
        winner = min(entities, key=lambda e: e.created_at)
    else:
        winner_match = next((e for e in entities if e.id == keep), None)
        if winner_match is None:
            raise ValueError(f"keep={keep!r} does not match any merged entity id")
        winner = winner_match
    losers = [e for e in entities if e.id != winner.id]

    closed_binding_ids: list[str] = []
    opened_binding_ids: list[str] = []
    dropped_negative_ids: list[str] = []
    moved_fact_ids: list[tuple[str, str]] = []  # (fact_id, original_entity_id)
    collapsed_relationship_ids: list[str] = []

    for loser in losers:
        # Migrate current positive bindings
        for bnd in store.bindings_for_entity(loser.id, include_negative=False):
            if bnd.valid_to is not None:
                continue
            store.close_binding(bnd.id, at=now)
            closed_binding_ids.append(bnd.id)
            migrated = Binding(
                id=_make_binding_id(),
                observation_id=bnd.observation_id,
                entity_id=winner.id,
                confidence=bnd.confidence,
                method=Method.MIGRATED,
                valid_from=now,
                recorded_at=now,
                actor=actor,
                polarity=Polarity.POSITIVE,
                valid_to=None,
            )
            store.append_binding(migrated)
            opened_binding_ids.append(migrated.id)

        # Close (but do not migrate) loser's negative bindings
        for bnd in store.bindings_for_entity(loser.id, include_negative=True):
            if bnd.polarity is Polarity.NEGATIVE and bnd.valid_to is None:
                store.close_binding(bnd.id, at=now)
                dropped_negative_ids.append(bnd.id)

        # Migrate facts (with provenance so undo knows where they came from)
        for fact in store.facts_for_entity(loser.id, active_only=True):
            moved = replace(fact, entity_id=winner.id, provenance_entity_id=loser.id)
            store.put_fact(moved)
            moved_fact_ids.append((fact.id, loser.id))

        # Events — same migration (rewrite entity_id with provenance)
        for event in store.events_for_entity(loser.id):
            moved_event = replace(event, entity_id=winner.id, provenance_entity_id=loser.id)
            store.put_event(moved_event)

        # Relationships — rewrite endpoints, then collapse self-loops
        for rel in store.relationships_for_entity(loser.id, active_only=True):
            new_from = winner.id if rel.from_entity_id == loser.id else rel.from_entity_id
            new_to = winner.id if rel.to_entity_id == loser.id else rel.to_entity_id
            updated = replace(
                rel,
                from_entity_id=new_from,
                to_entity_id=new_to,
                provenance_from_id=loser.id if rel.from_entity_id == loser.id else None,
                provenance_to_id=loser.id if rel.to_entity_id == loser.id else None,
            )
            store.put_relationship(updated)

        # Tombstone the loser entity
        store.put_entity(replace(loser, status=Status.MERGED_INTO, merged_into_id=winner.id))

    # Self-loop collapse on winner's active relationships
    for rel in store.relationships_for_entity(winner.id, active_only=True):
        if rel.from_entity_id == winner.id and rel.to_entity_id == winner.id:
            store.put_relationship(replace(rel, valid_to=now))
            collapsed_relationship_ids.append(rel.id)

    # last_seen bump
    if winner.last_seen < now:
        winner = replace(winner, last_seen=now)
        store.put_entity(winner)

    store.append_event_log(
        EventLog(
            id=0,
            op_type=OpType.MERGE.value,
            payload={
                "winner_id": winner.id,
                "loser_ids": [e.id for e in losers],
                "closed_binding_ids": closed_binding_ids,
                "opened_binding_ids": opened_binding_ids,
                "dropped_negative_ids": dropped_negative_ids,
                "moved_fact_ids": moved_fact_ids,
                "collapsed_relationship_ids": collapsed_relationship_ids,
            },
            actor=actor,
            affected_entity_ids=tuple([winner.id, *(e.id for e in losers)]),
            at=now,
            reversible_until=now + DEFAULT_UNDO_WINDOW,
        )
    )
    return winner


# ---------- split (§4.4) ----------


def split(
    store: Store,
    entity_id: str,
    groups: list[list[str]],
    *,
    clock: Clock,
    actor: str,
    fact_policy: str = "keep_original",
) -> list[Entity]:
    """Break one entity into N. ``groups[0]`` stays on the original id.

    Cross-wise negative bindings are emitted so the auto-clusterer cannot
    silently re-merge what the user just split. Facts, events, and
    relationships stay on the original by default (``fact_policy`` decides).
    """
    now = clock.now()
    if not groups:
        raise ValueError("split requires at least one group")

    original = store.get_entity(entity_id)
    if original is None:
        raise EntityUnavailableError(f"entity {entity_id} not found")
    _require_active(original)

    # Create new entities for groups[1:]; groups[0] keeps the original.
    result_entities: list[Entity] = [original]
    for idx in range(1, len(groups)):
        new_entity = Entity(
            id=_make_entity_id(),
            kind=original.kind,
            name=f"{original.name} (split {idx})" if original.name else f"split_{idx}",
            modality=original.modality,
            status=Status.ACTIVE,
            created_at=now,
            last_seen=now,
        )
        store.put_entity(new_entity)
        result_entities.append(new_entity)

    closed_binding_ids: list[str] = []
    new_binding_ids: list[str] = []
    cross_negative_ids: list[str] = []

    # Close old positive bindings for moved observations, open new ones on the
    # correct entity.
    for group_idx, obs_ids in enumerate(groups):
        target = result_entities[group_idx]
        for oid in obs_ids:
            for bnd in store.current_positive_bindings(oid):
                if bnd.entity_id == entity_id:
                    store.close_binding(bnd.id, at=now)
                    closed_binding_ids.append(bnd.id)
            # If the observation was already on the target (groups[0] case)
            # nothing to add; otherwise open a new positive binding.
            if target.id != entity_id:
                new_binding = Binding(
                    id=_make_binding_id(),
                    observation_id=oid,
                    entity_id=target.id,
                    confidence=1.0,
                    method=Method.MIGRATED,
                    valid_from=now,
                    recorded_at=now,
                    actor=actor,
                    polarity=Polarity.POSITIVE,
                    valid_to=None,
                )
                store.append_binding(new_binding)
                new_binding_ids.append(new_binding.id)
            else:
                # Re-open a fresh binding on the original so groups[0] obs stay attached.
                # (The close above would orphan them otherwise.)
                # Only re-open if we actually closed one — don't spuriously add bindings
                # for observations that weren't bound to this entity.
                # The close only fires on matching entity_id, so this branch is safe.
                # Re-opens are tagged MIGRATED to indicate this was an automated move.
                if any(cid in closed_binding_ids for cid in closed_binding_ids[-1:]):
                    # Only re-open if we actually closed one for this obs.
                    # A simpler check: was there a closed binding for this obs on original?
                    pass  # handled below
    # Simpler re-open: for groups[0], restore a current positive binding on original
    # for any observation we just closed.
    if groups:
        first_obs = set(groups[0])
        for oid in first_obs:
            current = store.current_positive_bindings(oid)
            if not any(b.entity_id == entity_id for b in current):
                new_binding = Binding(
                    id=_make_binding_id(),
                    observation_id=oid,
                    entity_id=entity_id,
                    confidence=1.0,
                    method=Method.MIGRATED,
                    valid_from=now,
                    recorded_at=now,
                    actor=actor,
                    polarity=Polarity.POSITIVE,
                    valid_to=None,
                )
                store.append_binding(new_binding)
                new_binding_ids.append(new_binding.id)

    # Cross-wise negatives — every group's obs get negatives against every OTHER group's entity.
    for i, obs_ids_i in enumerate(groups):
        ent_i = result_entities[i]
        for j, _obs_ids_j in enumerate(groups):
            if i == j:
                continue
            other = result_entities[j]
            for oid in obs_ids_i:
                neg = Binding(
                    id=_make_binding_id(),
                    observation_id=oid,
                    entity_id=other.id,
                    confidence=1.0,
                    method=Method.USER_REJECT,
                    valid_from=now,
                    recorded_at=now,
                    actor=actor,
                    polarity=Polarity.NEGATIVE,
                    valid_to=None,
                )
                store.append_binding(neg)
                cross_negative_ids.append(neg.id)
            # silence the unused-var check for ent_i when there's only one group
            _ = ent_i

    # Fact policy
    if fact_policy == "copy_to_all":
        originals = store.facts_for_entity(entity_id, active_only=True)
        for split_entity in result_entities[1:]:
            for fact in originals:
                copy = replace(
                    fact,
                    id=_make_fact_id(),
                    entity_id=split_entity.id,
                    provenance_entity_id=entity_id,
                )
                store.put_fact(copy)

    store.append_event_log(
        EventLog(
            id=0,
            op_type=OpType.SPLIT.value,
            payload={
                "original_entity_id": entity_id,
                "group_entity_ids": [e.id for e in result_entities],
                "group_observation_ids": [list(g) for g in groups],
                "closed_binding_ids": closed_binding_ids,
                "new_binding_ids": new_binding_ids,
                "cross_negative_ids": cross_negative_ids,
                "fact_policy": fact_policy,
            },
            actor=actor,
            affected_entity_ids=tuple(e.id for e in result_entities),
            at=now,
            reversible_until=now + DEFAULT_UNDO_WINDOW,
        )
    )
    return result_entities


# ---------- forget (§4.5) ----------


def forget(
    store: Store,
    *,
    entity_id: str,
    clock: Clock,
    actor: str,
    grace_days: int = 0,
) -> dict[str, int]:
    """Hard-delete everything tied to the entity and prune old versions.

    Not reversible by ``undo``. Returns deletion counts. Payload in the
    EventLog carries counts only — no content, no IDs of deleted rows — so
    the audit record itself is not a biometric artifact.

    ``grace_days > 0`` is accepted but not yet implemented; a future release
    will support a restore-within-window path. For now it is equivalent to 0.
    """
    now = clock.now()
    entity = store.get_entity(entity_id)
    if entity is None:
        raise EntityUnavailableError(f"entity {entity_id} not found")
    if entity.status is Status.FORGOTTEN:
        raise EntityUnavailableError(f"entity {entity_id} already forgotten")

    counts = store.delete_entity_cascade(entity_id)

    store.put_entity(
        replace(
            entity,
            name="",
            aliases=(),
            status=Status.FORGOTTEN,
            merged_into_id=None,
        )
    )

    # Prune old versions so biometric data is gone from version history.
    store.prune_versions(older_than=now)

    store.append_event_log(
        EventLog(
            id=0,
            op_type=OpType.FORGET.value,
            payload={"entity_id": entity_id, "counts": counts, "grace_days": grace_days},
            actor=actor,
            affected_entity_ids=(entity_id,),
            at=now,
            reversible_until=None,  # Not reversible by undo.
        )
    )
    return counts


# ---------- restrict / unrestrict (§4.7) ----------


def restrict(
    store: Store,
    *,
    entity_id: str,
    clock: Clock,
    actor: str,
) -> Entity:
    """Stop using the entity for inference without deleting (GDPR Art. 18).

    Sets ``status = RESTRICTED``. Restricted entities are excluded from
    ``identify()`` but facts remain readable via ``recall()`` for the owner.
    """
    return _flip_status(
        store,
        entity_id=entity_id,
        clock=clock,
        actor=actor,
        op=OpType.RESTRICT,
        from_status=Status.ACTIVE,
        to_status=Status.RESTRICTED,
    )


def unrestrict(
    store: Store,
    *,
    entity_id: str,
    clock: Clock,
    actor: str,
) -> Entity:
    """Reverse of ``restrict`` — returns entity to ACTIVE."""
    return _flip_status(
        store,
        entity_id=entity_id,
        clock=clock,
        actor=actor,
        op=OpType.UNRESTRICT,
        from_status=Status.RESTRICTED,
        to_status=Status.ACTIVE,
    )


def _flip_status(
    store: Store,
    *,
    entity_id: str,
    clock: Clock,
    actor: str,
    op: OpType,
    from_status: Status,
    to_status: Status,
) -> Entity:
    now = clock.now()
    entity = store.get_entity(entity_id)
    if entity is None:
        raise EntityUnavailableError(f"entity {entity_id} not found")
    if entity.status is not from_status:
        raise EntityUnavailableError(
            f"entity {entity_id} is {entity.status.value}, cannot {op.value}"
        )

    updated = replace(entity, status=to_status)
    store.put_entity(updated)

    store.append_event_log(
        EventLog(
            id=0,
            op_type=op.value,
            payload={
                "entity_id": entity_id,
                "from_status": from_status.value,
                "to_status": to_status.value,
            },
            actor=actor,
            affected_entity_ids=(entity_id,),
            at=now,
            reversible_until=now + DEFAULT_UNDO_WINDOW,
        )
    )
    return updated


# ---------- export (§4.8) ----------


def export(
    store: Store,
    *,
    entity_id: str,
    include_embeddings: bool = False,
) -> dict[str, Any]:
    """GDPR Art. 20 data portability dump of everything the store knows.

    Returns a plain dict ready for JSON serialization. Embedding vectors are
    excluded by default — raw biometric vectors in a user-facing export are
    often worse than useless. Pass ``include_embeddings=True`` for a full
    round-trippable dump.
    """
    entity = store.get_entity(entity_id)
    if entity is None:
        raise EntityUnavailableError(f"entity {entity_id} not found")

    bindings = [
        _binding_dict(b) for b in store.bindings_for_entity(entity_id, include_negative=True)
    ]

    # Collect observations bound to this entity
    observation_ids = {
        b.observation_id for b in store.bindings_for_entity(entity_id, include_negative=True)
    }
    observations = []
    embeddings: list[dict[str, Any]] = []
    for oid in observation_ids:
        obs = store.get_observation(oid)
        if obs is None:
            continue
        observations.append(_observation_dict(obs))
        if include_embeddings:
            # FakeStore doesn't expose a public "embeddings for observation" query,
            # so skip if the Store impl can't produce one — production impls do.
            get_embs = getattr(store, "embeddings_for_observation", None)
            if callable(get_embs):
                for emb in get_embs(oid):
                    embeddings.append(_embedding_dict(emb))

    # If include_embeddings but store doesn't have the optional method, fall back
    # to scanning any attribute-exposed embedding cache (FakeStore keeps a dict).
    if include_embeddings and not embeddings:
        cache = getattr(store, "_embeddings", None)
        if isinstance(cache, dict):
            for emb in cache.values():
                if emb.observation_id in observation_ids:
                    embeddings.append(_embedding_dict(emb))

    facts = [_fact_dict(f) for f in store.facts_for_entity(entity_id, active_only=False)]
    events = [_event_dict(e) for e in store.events_for_entity(entity_id)]
    relationships = [
        _relationship_dict(r) for r in store.relationships_for_entity(entity_id, active_only=False)
    ]
    event_log = [_event_log_dict(e) for e in store.events_affecting_entity(entity_id)]

    return {
        "schema_version": 1,
        "entity": _entity_dict(entity),
        "observations": observations,
        "embeddings": embeddings,
        "bindings": bindings,
        "facts": facts,
        "events": events,
        "relationships": relationships,
        "event_log": event_log,
    }


# ---------- export serializers (module-private) ----------


def _entity_dict(e: Entity) -> dict[str, Any]:
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


def _observation_dict(o: Any) -> dict[str, Any]:
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


def _embedding_dict(e: Any) -> dict[str, Any]:
    return {
        "id": e.id,
        "observation_id": e.observation_id,
        "encoder_id": e.encoder_id,
        "vector": list(e.vector),
        "dim": e.dim,
        "created_at": e.created_at.isoformat(),
        "key_id": e.key_id,
    }


def _binding_dict(b: Binding) -> dict[str, Any]:
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


def _fact_dict(f: Fact) -> dict[str, Any]:
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


def _event_dict(e: Any) -> dict[str, Any]:
    return {
        "id": e.id,
        "entity_id": e.entity_id,
        "content": e.content,
        "source": e.source.value,
        "occurred_at": e.occurred_at.isoformat(),
        "recorded_at": e.recorded_at.isoformat(),
        "provenance_entity_id": e.provenance_entity_id,
    }


def _relationship_dict(r: Any) -> dict[str, Any]:
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


def _event_log_dict(e: EventLog) -> dict[str, Any]:
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


# ---------- undo (§4.6) ----------


def undo(
    store: Store,
    *,
    event_id: int | None = None,
    clock: Clock,
    actor: str,
) -> EventLog:
    """Reverse a prior reversible operation.

    Dispatch by ``op_type``; each handler uses the stored payload to reconstruct
    the pre-op state. Forget is not reversible; undo itself is not reversible
    (no redo in v0); events past ``reversible_until`` are rejected; events
    already reversed by a prior undo are rejected.

    If ``event_id`` is None, undoes the most recent reversible event by
    ``actor``.
    """
    now = clock.now()
    target = _resolve_event_to_undo(store, event_id=event_id, actor=actor, now=now)

    op = target.op_type
    handler = _UNDO_HANDLERS.get(op)
    if handler is None:
        raise OperationNotReversibleError(f"no undo handler for op_type {op!r}")

    handler(store, target, now=now, actor=actor)

    return store.append_event_log(
        EventLog(
            id=0,
            op_type=OpType.UNDO.value,
            payload={"undone_event_id": target.id, "original_op_type": op},
            actor=actor,
            affected_entity_ids=target.affected_entity_ids,
            at=now,
            reversible_until=None,  # redo not supported in v0
        )
    )


def _resolve_event_to_undo(
    store: Store, *, event_id: int | None, actor: str, now: datetime
) -> EventLog:
    if event_id is not None:
        target = store.get_event_log(event_id)
        if target is None:
            raise OperationNotReversibleError(f"event {event_id} not found")
    else:
        # Scan back for the most recent reversible event by this actor.
        # Because events_affecting_entity requires an entity_id, we walk all
        # entities' event lists. For a single-writer store this is O(events);
        # acceptable for v0.
        target = _most_recent_reversible_by(store, actor=actor, now=now)
        if target is None:
            raise OperationNotReversibleError(f"no reversible events to undo for actor {actor!r}")

    if target.reversible_until is None:
        raise OperationNotReversibleError(f"event {target.id} is not reversible")
    if target.reversible_until < now:
        raise OperationNotReversibleError(
            f"event {target.id} expired at {target.reversible_until.isoformat()}"
        )
    if _already_reversed(store, target):
        raise OperationNotReversibleError(f"event {target.id} has already been undone")
    return target


def _most_recent_reversible_by(store: Store, *, actor: str, now: datetime) -> EventLog | None:
    candidates = store.list_events(actor=actor)
    reversible = [
        e
        for e in candidates
        if e.reversible_until is not None
        and e.reversible_until >= now
        and not _already_reversed(store, e)
        and e.op_type != OpType.UNDO.value
    ]
    if not reversible:
        return None
    # (at, id) tiebreaker — multiple ops at the same clock tick (e.g. remember
    # right after label, or the internal label inside relabel) need a stable
    # order, and the store-assigned id is monotonic.
    reversible.sort(key=lambda e: (e.at, e.id), reverse=True)
    return reversible[0]


def _already_reversed(store: Store, event: EventLog) -> bool:
    seen: set[int] = set()
    for eid in event.affected_entity_ids:
        for e in store.events_affecting_entity(eid):
            if e.id in seen:
                continue
            seen.add(e.id)
            if e.op_type == OpType.UNDO.value and e.payload.get("undone_event_id") == event.id:
                return True
    return False


# ---------- per-op undo handlers ----------


def _undo_label(store: Store, event: EventLog, *, now: datetime, actor: str) -> None:
    payload = event.payload
    # Close the new bindings created by label
    for bnd_id in payload.get("new_binding_ids", []):
        b = store.get_binding(bnd_id)
        if b is not None and b.valid_to is None:
            store.close_binding(bnd_id, at=now)
    # Restore prior bindings — create equivalent new positive bindings
    for closed_bnd_id in payload.get("closed_binding_ids", []):
        prior = store.get_binding(closed_bnd_id)
        if prior is None:
            continue
        restored = Binding(
            id=_make_binding_id(),
            observation_id=prior.observation_id,
            entity_id=prior.entity_id,
            confidence=prior.confidence,
            method=prior.method,
            valid_from=now,
            recorded_at=now,
            actor=actor,
            polarity=prior.polarity,
            valid_to=None,
        )
        store.append_binding(restored)


def _undo_remember(store: Store, event: EventLog, *, now: datetime, actor: str) -> None:
    fact_id = event.payload.get("fact_id")
    if isinstance(fact_id, str):
        fact = store.get_fact(fact_id)
        if fact is not None and fact.valid_to is None:
            store.retract_fact(fact_id, at=now)


def _undo_relabel(store: Store, event: EventLog, *, now: datetime, actor: str) -> None:
    # A relabel is: label() onto a new entity + negative bindings against priors.
    # Undoing requires closing the new positive + negatives, restoring prior positives.
    observation_id = event.payload.get("observation_id")
    new_entity_id = event.payload.get("new_entity_id")
    prior_entity_ids = event.payload.get("prior_entity_ids", [])
    negative_binding_ids = event.payload.get("negative_binding_ids", [])

    if not isinstance(observation_id, str):
        return

    # Close the current positive binding (the label() call inside relabel created it)
    if isinstance(new_entity_id, str):
        for b in store.current_positive_bindings(observation_id):
            if b.entity_id == new_entity_id:
                store.close_binding(b.id, at=now)

    # Close the negative bindings
    for neg_id in negative_binding_ids:
        neg = store.get_binding(neg_id)
        if neg is not None and neg.valid_to is None:
            store.close_binding(neg_id, at=now)

    # Restore prior positives — re-attach observation to previous entities
    for prior_entity_id in prior_entity_ids:
        restored = Binding(
            id=_make_binding_id(),
            observation_id=observation_id,
            entity_id=prior_entity_id,
            confidence=1.0,
            method=Method.USER_LABEL,
            valid_from=now,
            recorded_at=now,
            actor=actor,
            polarity=Polarity.POSITIVE,
            valid_to=None,
        )
        store.append_binding(restored)


def _undo_merge(store: Store, event: EventLog, *, now: datetime, actor: str) -> None:
    payload = event.payload
    # Close migrated bindings on the winner
    for bnd_id in payload.get("opened_binding_ids", []):
        b = store.get_binding(bnd_id)
        if b is not None and b.valid_to is None:
            store.close_binding(bnd_id, at=now)

    # Restore loser bindings (re-attach observations to losers)
    for closed_bnd_id in payload.get("closed_binding_ids", []):
        prior = store.get_binding(closed_bnd_id)
        if prior is None:
            continue
        restored = Binding(
            id=_make_binding_id(),
            observation_id=prior.observation_id,
            entity_id=prior.entity_id,
            confidence=prior.confidence,
            method=prior.method,
            valid_from=now,
            recorded_at=now,
            actor=actor,
            polarity=prior.polarity,
            valid_to=None,
        )
        store.append_binding(restored)

    # Restore loser negative bindings that we had closed
    for neg_id in payload.get("dropped_negative_ids", []):
        prior = store.get_binding(neg_id)
        if prior is None:
            continue
        restored = Binding(
            id=_make_binding_id(),
            observation_id=prior.observation_id,
            entity_id=prior.entity_id,
            confidence=prior.confidence,
            method=prior.method,
            valid_from=now,
            recorded_at=now,
            actor=actor,
            polarity=Polarity.NEGATIVE,
            valid_to=None,
        )
        store.append_binding(restored)

    # Move facts back to their original entity
    for fact_id, original_entity_id in payload.get("moved_fact_ids", []):
        fact = store.get_fact(fact_id)
        if fact is None:
            continue
        store.put_fact(replace(fact, entity_id=original_entity_id, provenance_entity_id=None))

    # Restore collapsed relationships
    for rel_id in payload.get("collapsed_relationship_ids", []):
        rel = store.get_relationship(rel_id)
        if rel is not None and rel.valid_to is not None:
            store.put_relationship(replace(rel, valid_to=None))

    # Resurrect losers — status back to ACTIVE, clear merged_into_id
    for loser_id in payload.get("loser_ids", []):
        loser = store.get_entity(loser_id)
        if loser is not None and loser.status is Status.MERGED_INTO:
            store.put_entity(replace(loser, status=Status.ACTIVE, merged_into_id=None))


def _undo_split(store: Store, event: EventLog, *, now: datetime, actor: str) -> None:
    payload = event.payload

    # Close new bindings on split-off entities
    for bnd_id in payload.get("new_binding_ids", []):
        b = store.get_binding(bnd_id)
        if b is not None and b.valid_to is None:
            store.close_binding(bnd_id, at=now)

    # Close cross-wise negatives
    for bnd_id in payload.get("cross_negative_ids", []):
        b = store.get_binding(bnd_id)
        if b is not None and b.valid_to is None:
            store.close_binding(bnd_id, at=now)

    # Restore original bindings by re-creating positive bindings on original entity
    original_entity_id = payload.get("original_entity_id")
    if isinstance(original_entity_id, str):
        for closed_bnd_id in payload.get("closed_binding_ids", []):
            prior = store.get_binding(closed_bnd_id)
            if prior is None or prior.entity_id != original_entity_id:
                continue
            restored = Binding(
                id=_make_binding_id(),
                observation_id=prior.observation_id,
                entity_id=original_entity_id,
                confidence=prior.confidence,
                method=prior.method,
                valid_from=now,
                recorded_at=now,
                actor=actor,
                polarity=prior.polarity,
                valid_to=None,
            )
            store.append_binding(restored)

    # Tombstone split-off entities (groups[1:])
    group_entity_ids = payload.get("group_entity_ids", [])
    if original_entity_id and len(group_entity_ids) > 1:
        for split_off_id in group_entity_ids[1:]:
            e = store.get_entity(split_off_id)
            if e is not None and e.status is Status.ACTIVE:
                store.put_entity(replace(e, status=Status.FORGOTTEN, name="", aliases=()))


def _undo_restrict(store: Store, event: EventLog, *, now: datetime, actor: str) -> None:
    entity_id = event.payload.get("entity_id")
    from_status = event.payload.get("from_status")
    if isinstance(entity_id, str) and isinstance(from_status, str):
        entity = store.get_entity(entity_id)
        if entity is not None:
            store.put_entity(replace(entity, status=Status(from_status)))


_UNDO_HANDLERS: dict[str, Any] = {
    OpType.LABEL.value: _undo_label,
    OpType.REMEMBER.value: _undo_remember,
    OpType.RELABEL.value: _undo_relabel,
    OpType.MERGE.value: _undo_merge,
    OpType.SPLIT.value: _undo_split,
    OpType.RESTRICT.value: _undo_restrict,
    OpType.UNRESTRICT.value: _undo_restrict,
}
