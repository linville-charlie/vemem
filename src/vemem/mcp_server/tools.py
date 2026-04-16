"""Tool handlers — one per MCP tool.

Each handler is a plain function that accepts a :class:`ServerContext` plus
typed kwargs and returns a JSON-serializable dict. FastMCP generates the tool
schema from the function signature, and the server module in ``server.py``
wraps each handler in a thin closure that supplies the shared context.

Keeping handlers as top-level functions (rather than methods) makes them
trivially unit-testable: the tests in ``tests/mcp_server/test_tools.py`` pass
an in-memory :class:`FakeStore` and a stub encoder/detector directly.

Design notes:

- **Actor attribution** — every write op takes an optional ``actor``
  (default ``"mcp:unknown"``). Claude Desktop and most MCP clients don't
  forward a principal; a savvier client can override per-call.
- **Image inputs** — base64-encoded bytes decoded to ``bytes`` here. The
  detector produces bboxes; the encoder produces embeddings. Each detected
  face becomes one Observation + one Embedding row in the store.
- **Error surface** — domain errors (``ModalityMismatchError`` etc.) bubble
  up to the MCP layer as ``ToolError`` with the exception message. Missing
  encoder surfaces as a ``RuntimeError`` with the startup diagnostic, so
  the caller gets an actionable message rather than a generic 500.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Any

from vemem.core import ops
from vemem.core.enums import Modality, Source
from vemem.core.ids import new_id
from vemem.core.protocols import Clock, Detector, Encoder, Store
from vemem.core.types import Embedding, Observation, observation_id_for
from vemem.mcp_server.serialization import (
    candidate_to_dict,
    entity_to_dict,
    event_log_to_dict,
    fact_to_dict,
    recall_snapshot_to_dict,
)

DEFAULT_ACTOR = "mcp:unknown"


@dataclass
class ServerContext:
    """Shared dependencies passed to every tool handler.

    Bundling these keeps the handler signatures tight (just the tool's own
    arguments) and lets the integration test swap real components for fakes
    without reaching into the server's globals.
    """

    store: Store
    clock: Clock
    encoder: Encoder | None
    detector: Detector | None
    encoder_error: str | None = None

    def require_image_pipeline(self) -> tuple[Encoder, Detector]:
        """Return (encoder, detector) or raise a RuntimeError with the startup error.

        Image-dependent tools call this before decoding bytes so the caller
        gets a useful error instead of an AttributeError on ``None``.
        """
        if self.encoder is None or self.detector is None:
            msg = self.encoder_error or "encoder/detector not initialized"
            raise RuntimeError(msg)
        return self.encoder, self.detector


# ---------- image helpers ----------


def _decode_image(image_b64: str) -> bytes:
    """Decode a base64 string to raw bytes, with a helpful error on bad input."""
    try:
        return base64.b64decode(image_b64, validate=True)
    except Exception as exc:
        raise ValueError(f"image_base64 is not valid base64: {exc}") from exc


def _ingest_image(
    ctx: ServerContext,
    *,
    image_bytes: bytes,
    source_uri: str,
    modality: Modality,
) -> list[Observation]:
    """Detect faces in ``image_bytes``, persist Observation + Embedding per face.

    Returns the list of Observation records that were created or matched
    (observation ids are content-addressed, so re-ingesting the same image is
    idempotent). Embeddings are appended once per (observation, encoder) pair.
    """
    encoder, detector = ctx.require_image_pipeline()
    now = ctx.clock.now()
    source_hash = hashlib.sha256(image_bytes).hexdigest()

    bboxes = detector.detect(image_bytes)
    observations: list[Observation] = []
    for bbox in bboxes:
        obs_id = observation_id_for(source_hash, bbox, detector.id)
        existing = ctx.store.get_observation(obs_id)
        if existing is None:
            obs = Observation(
                id=obs_id,
                source_uri=source_uri,
                source_hash=source_hash,
                bbox=bbox,
                detector_id=detector.id,
                modality=modality,
                detected_at=now,
            )
            ctx.store.put_observation(obs)
        else:
            obs = existing
        observations.append(obs)

        # Embed the full image for now — cropping is a detector refinement.
        # InsightFace's pipeline does its own crop + align internally when we
        # hand it the full frame.
        vector = encoder.embed(image_bytes)
        emb = Embedding(
            id="emb_" + new_id(),
            observation_id=obs.id,
            encoder_id=encoder.id,
            vector=vector,
            dim=encoder.dim,
            created_at=now,
        )
        ctx.store.put_embedding(emb)

    return observations


# ---------- core identity ----------


def observe_image(
    ctx: ServerContext,
    *,
    image_base64: str,
    source_uri: str = "mcp://inline",
    modality: str = "face",
) -> dict[str, Any]:
    """Detect and persist observations for every entity found in the image.

    Returns the list of observation ids and their bboxes so a follow-up
    ``label`` call can target them. Idempotent: re-observing the same image
    returns the same ids (§3.1).
    """
    image_bytes = _decode_image(image_base64)
    observations = _ingest_image(
        ctx,
        image_bytes=image_bytes,
        source_uri=source_uri,
        modality=Modality(modality),
    )
    return {
        "observations": [
            {"id": o.id, "bbox": list(o.bbox), "detector_id": o.detector_id} for o in observations
        ],
        "source_hash": observations[0].source_hash if observations else None,
    }


def identify_image(
    ctx: ServerContext,
    *,
    image_base64: str,
    k: int = 5,
    min_confidence: float = 0.5,
    prefer: str = "instance",
) -> dict[str, Any]:
    """Identify entities in an image without mutating state.

    Runs the detector + encoder on the image, queries the gallery via
    :func:`vemem.core.ops.identify`, and returns ranked :class:`Candidate`
    entities per detected face.
    """
    encoder, detector = ctx.require_image_pipeline()
    image_bytes = _decode_image(image_base64)

    bboxes = detector.detect(image_bytes)
    results: list[dict[str, Any]] = []
    for bbox in bboxes:
        vector = encoder.embed(image_bytes)
        candidates = ops.identify(
            ctx.store,
            encoder_id=encoder.id,
            vector=vector,
            k=k,
            min_confidence=min_confidence,
            prefer=prefer,
        )
        results.append(
            {
                "bbox": list(bbox),
                "candidates": [candidate_to_dict(c) for c in candidates],
            }
        )
    return {"detections": results, "encoder_id": encoder.id}


def identify_by_name(
    ctx: ServerContext,
    *,
    entity_name_or_id: str,
) -> dict[str, Any]:
    """Resolve an entity by name or id and return its recall snapshot.

    Convenience wrapper combining ``find_entity_by_name`` + :func:`recall`.
    """
    entity = ctx.store.get_entity(entity_name_or_id) or ctx.store.find_entity_by_name(
        entity_name_or_id
    )
    if entity is None:
        raise ValueError(f"no entity found for {entity_name_or_id!r}")
    snap = ops.recall(ctx.store, entity_id=entity.id)
    return recall_snapshot_to_dict(snap)


# ---------- state changes ----------


def label_tool(
    ctx: ServerContext,
    *,
    observation_ids: list[str],
    entity_name_or_id: str,
    actor: str = DEFAULT_ACTOR,
) -> dict[str, Any]:
    """Commit a user-authoritative positive binding (§4.1)."""
    entity = ops.label(
        ctx.store,
        observation_ids,
        entity_name_or_id,
        clock=ctx.clock,
        actor=actor,
    )
    return entity_to_dict(entity)


def relabel_tool(
    ctx: ServerContext,
    *,
    observation_id: str,
    new_entity_name_or_id: str,
    actor: str = DEFAULT_ACTOR,
) -> dict[str, Any]:
    """Move a single observation to a different entity (§4.2)."""
    entity = ops.relabel(
        ctx.store,
        observation_id,
        new_entity_name_or_id,
        clock=ctx.clock,
        actor=actor,
    )
    return entity_to_dict(entity)


def merge_tool(
    ctx: ServerContext,
    *,
    entity_ids: list[str],
    keep: str = "oldest",
    actor: str = DEFAULT_ACTOR,
) -> dict[str, Any]:
    """Fold losers into a winner (§4.3). Facts migrate with provenance."""
    entity = ops.merge(
        ctx.store,
        entity_ids,
        keep=keep,
        clock=ctx.clock,
        actor=actor,
    )
    return entity_to_dict(entity)


def split_tool(
    ctx: ServerContext,
    *,
    entity_id: str,
    groups: list[list[str]],
    fact_policy: str = "keep_original",
    actor: str = DEFAULT_ACTOR,
) -> dict[str, Any]:
    """Break one entity into N (§4.4). ``groups[0]`` stays on the original id."""
    result = ops.split(
        ctx.store,
        entity_id,
        groups,
        clock=ctx.clock,
        actor=actor,
        fact_policy=fact_policy,
    )
    return {"entities": [entity_to_dict(e) for e in result]}


def forget_tool(
    ctx: ServerContext,
    *,
    entity_id: str,
    grace_days: int = 0,
    actor: str = DEFAULT_ACTOR,
) -> dict[str, Any]:
    """Hard-delete everything tied to the entity (§4.5). Not reversible."""
    counts = ops.forget(
        ctx.store,
        entity_id=entity_id,
        clock=ctx.clock,
        actor=actor,
        grace_days=grace_days,
    )
    return {"entity_id": entity_id, "counts": counts}


def restrict_tool(
    ctx: ServerContext,
    *,
    entity_id: str,
    actor: str = DEFAULT_ACTOR,
) -> dict[str, Any]:
    """Stop using the entity for inference (§4.7)."""
    entity = ops.restrict(ctx.store, entity_id=entity_id, clock=ctx.clock, actor=actor)
    return entity_to_dict(entity)


def unrestrict_tool(
    ctx: ServerContext,
    *,
    entity_id: str,
    actor: str = DEFAULT_ACTOR,
) -> dict[str, Any]:
    """Reverse of :func:`restrict_tool`."""
    entity = ops.unrestrict(ctx.store, entity_id=entity_id, clock=ctx.clock, actor=actor)
    return entity_to_dict(entity)


# ---------- knowledge ----------


def remember_tool(
    ctx: ServerContext,
    *,
    entity_id: str,
    content: str,
    source: str = "user",
    actor: str = DEFAULT_ACTOR,
) -> dict[str, Any]:
    """Attach a fact to an entity (§3.4, §4.x knowledge layer)."""
    fact = ops.remember(
        ctx.store,
        entity_id=entity_id,
        content=content,
        source=Source(source),
        clock=ctx.clock,
        actor=actor,
    )
    return fact_to_dict(fact)


def recall_tool(
    ctx: ServerContext,
    *,
    entity_id: str,
) -> dict[str, Any]:
    """Return the entity plus its currently-active facts (§4.9)."""
    snap = ops.recall(ctx.store, entity_id=entity_id)
    return recall_snapshot_to_dict(snap)


# ---------- audit ----------


def undo_tool(
    ctx: ServerContext,
    *,
    event_id: int | None = None,
    actor: str = DEFAULT_ACTOR,
) -> dict[str, Any]:
    """Reverse a prior reversible operation (§4.6).

    When ``event_id`` is omitted, undoes the most recent reversible event by
    ``actor``. Forget is not reversible by undo — an explicit
    ``OperationNotReversibleError`` bubbles up.
    """
    new_event = ops.undo(ctx.store, event_id=event_id, clock=ctx.clock, actor=actor)
    return event_log_to_dict(new_event)


def export_tool(
    ctx: ServerContext,
    *,
    entity_id: str,
    include_embeddings: bool = False,
) -> dict[str, Any]:
    """Data portability dump for one entity (§4.8)."""
    return ops.export(ctx.store, entity_id=entity_id, include_embeddings=include_embeddings)
