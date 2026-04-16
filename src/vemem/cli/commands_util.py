"""Small helpers used by multiple CLI commands.

The image-ingest path (``observe``, ``identify``) shares the same detector →
encoder → persist pipeline as the MCP server. Factoring it here keeps
``app.py`` focused on Typer wiring while guaranteeing both surfaces behave
identically: the same observation ids, the same idempotency rules, the same
embeddings row per (observation, encoder) pair.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from vemem.core.enums import Modality
from vemem.core.ids import new_id
from vemem.core.types import Embedding, Observation, observation_id_for

if TYPE_CHECKING:
    from vemem.cli.context import CliContext


def ingest_image(
    ctx: CliContext,
    *,
    image_bytes: bytes,
    source_uri: str,
    modality: Modality = Modality.FACE,
) -> list[str]:
    """Detect + embed + persist observations for one image, returning obs ids.

    Raises ``RuntimeError`` if the encoder/detector is not loaded — callers
    should surface that as a user-visible install hint.
    """
    if ctx.encoder is None or ctx.detector is None:
        raise RuntimeError(
            ctx.encoder_error or "encoder/detector not initialized",
        )

    now = ctx.clock.now()
    source_hash = hashlib.sha256(image_bytes).hexdigest()
    bboxes = ctx.detector.detect(image_bytes)

    obs_ids: list[str] = []
    for bbox in bboxes:
        obs_id = observation_id_for(source_hash, bbox, ctx.detector.id)
        existing = ctx.store.get_observation(obs_id)
        if existing is None:
            obs = Observation(
                id=obs_id,
                source_uri=source_uri,
                source_hash=source_hash,
                bbox=bbox,
                detector_id=ctx.detector.id,
                modality=modality,
                detected_at=now,
            )
            ctx.store.put_observation(obs)
        obs_ids.append(obs_id)

        vector = ctx.encoder.embed(image_bytes)
        emb = Embedding(
            id="emb_" + new_id(),
            observation_id=obs_id,
            encoder_id=ctx.encoder.id,
            vector=vector,
            dim=ctx.encoder.dim,
            created_at=now,
        )
        ctx.store.put_embedding(emb)

    return obs_ids
