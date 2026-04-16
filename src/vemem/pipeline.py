"""High-level image ingestion helpers.

``vemem.core.ops`` has no opinions about images — it operates on already-built
``Observation`` + ``Embedding`` domain objects. Real callers (the MCP server,
the CLI, the bridge example) all follow the same recipe to turn raw bytes
into those objects: hash the image, run the detector for bboxes, run the
encoder for a vector, assemble an ``Observation`` with a deterministic
content-hash id, attach the embedding, and persist.

This module is that recipe in one place so all three callers don't drift.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import TYPE_CHECKING

from vemem.core.enums import Modality
from vemem.core.ids import new_id
from vemem.core.types import Embedding, Observation, observation_id_for

if TYPE_CHECKING:
    from vemem.core.protocols import Clock, Detector, Encoder, Store


def observe_image(
    store: Store,
    *,
    image_bytes: bytes,
    detector: Detector,
    encoder: Encoder,
    clock: Clock,
    modality: Modality = Modality.FACE,
    source_uri: str | None = None,
    source_ts: datetime | None = None,
    source_frame: int | None = None,
) -> list[Observation]:
    """Detect, embed, and persist observations from a single image.

    Returns one ``Observation`` per detector bbox, in the order the detector
    emitted them. Observation ids are content-addressed (spec §3.1) so
    re-observing identical bytes returns existing rows idempotently.

    **Encoder input**: the current face-first encoders (InsightFace) run
    internal detection + alignment on whatever bytes they get, so we hand
    them the FULL image and let them re-find the face rather than pre-cropping
    per bbox. A strict Encoder impl that expects a pre-cropped face would
    need to crop first; left as an encoder-level policy.

    ``source_uri`` is an opaque reference the library never fetches (spec
    §3.1b); if ``None``, we fall back to ``hash:<sha256>``.
    """
    now = clock.now()
    source_hash = hashlib.sha256(image_bytes).hexdigest()
    bboxes = detector.detect(image_bytes)
    observations: list[Observation] = []

    for bbox in bboxes:
        obs_id = observation_id_for(source_hash, bbox, detector.id)
        existing = store.get_observation(obs_id)
        if existing is None:
            obs = Observation(
                id=obs_id,
                source_uri=source_uri or f"hash:{source_hash}",
                source_hash=source_hash,
                bbox=bbox,
                detector_id=detector.id,
                modality=modality,
                detected_at=now,
                source_ts=source_ts,
                source_frame=source_frame,
            )
            store.put_observation(obs)
        else:
            obs = existing

        vector = encoder.embed(image_bytes)
        store.put_embedding(
            Embedding(
                id="emb_" + new_id(),
                observation_id=obs.id,
                encoder_id=encoder.id,
                vector=tuple(vector),
                dim=len(vector),
                created_at=now,
            )
        )
        observations.append(obs)

    return observations
