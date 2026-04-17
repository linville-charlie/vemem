"""High-level image ingestion helpers.

``vemem.core.ops`` has no opinions about images — it operates on already-built
``Observation`` + ``Embedding`` domain objects. Real callers (the MCP server,
the CLI, the bridge example) all follow the same recipe to turn raw bytes
into those objects: hash the image, run the detector for bboxes, **crop each
bbox region**, run the encoder on the crop, assemble an ``Observation`` with
a deterministic content-hash id, attach the embedding, and persist.

This module is that recipe in one place so all three callers don't drift.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import TYPE_CHECKING

from vemem.core.enums import Modality
from vemem.core.ids import new_id
from vemem.core.types import Embedding, Observation, observation_id_for
from vemem.encoders.crop import crop_image

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

    The pipeline dispatches on the encoder's shape (see Encoder Protocol):

    - If the encoder implements ``embed_frame(image_bytes, bbox)``, we hand
      it the full frame and let it own detection/alignment. This is what
      InsightFace wants — it runs its own detector + landmark alignment
      internally, and a tight bbox crop would starve it of context.
    - Otherwise we crop to the bbox and call ``embed(crop)``. This is what
      CLIP / DINOv3 / SigLIP and similar "give me pre-cropped input" encoders
      expect; handing them the full frame would silently embed the whole
      image.

    Either way the encoder sees exactly the region the detector flagged, just
    via the contract it actually wants.

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

        # Two encoder input modes (see Encoder Protocol docstring):
        #   - embed_frame(full image, bbox) — encoders that do their own
        #     detection + landmark alignment internally (InsightFace). A
        #     tight bbox crop removes the context their internal detector
        #     needs, so we hand them the full frame and let them pick.
        #   - embed(crop) — encoders that expect a pre-cropped region
        #     (CLIP, DINOv3, …). The pipeline crops for them.
        if hasattr(encoder, "embed_frame"):
            vector = encoder.embed_frame(image_bytes, bbox)
        else:
            crop_bytes = crop_image(image_bytes, bbox)
            vector = encoder.embed(crop_bytes)
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
