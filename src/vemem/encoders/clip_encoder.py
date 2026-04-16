"""CLIP encoder adapter (experimental in v0).

Uses ``open_clip_torch`` to produce general-purpose image embeddings. Intended
as a scaffold: v0 ships face recognition via InsightFace as the production
path; CLIP is here so the :class:`Encoder` Protocol has a second exercised
implementation and so object / scene entities have a path forward in v0.1
without a rewrite.

Thresholds, identify defaults, and ANN parameters are NOT tuned around CLIP in
v0 — treat this adapter as "runs end-to-end, produces normalized vectors of
the advertised dimension" only.

Defaults to ``ViT-B-32`` pretrained on ``laion2b_s34b_b79k`` — 512-d embeddings
out of the box. The 768-d variant (``ViT-L-14``) is available via constructor
args if the caller wants a bigger model. The ``dim`` attribute is set
dynamically from the loaded model so callers can't drift out of sync with the
advertised size.
"""

from __future__ import annotations

import io
from typing import Any

import open_clip
import torch
from PIL import Image


class CLIPEncoder:
    """EXPERIMENTAL: general-purpose CLIP image embeddings.

    This is a v0 scaffold. It satisfies the :class:`Encoder` Protocol and is
    safe to run end-to-end, but the library's identify/recall defaults are
    tuned for face recognition. Use with caution for anything that commits to
    a confidence threshold.
    """

    def __init__(
        self,
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model_name = model_name
        self._pretrained = pretrained

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained,
                device=self._device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load open_clip model '{model_name}' pretrained "
                f"'{pretrained}'. On first use open_clip downloads weights "
                "via HuggingFace Hub. Ensure network access or pre-populate "
                "the cache."
            ) from exc

        model.eval()
        self._model = model
        self._preprocess = preprocess

        # Probe the embedding dimension from the model rather than trusting
        # a hard-coded lookup table — variants and fine-tunes drift.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self._device)
            feat = model.encode_image(dummy)
        self.dim = int(feat.shape[-1])

        self.id = f"open_clip/{model_name}:{pretrained}@{open_clip.__version__}"

    def embed(self, image_crop: bytes) -> tuple[float, ...]:
        """Embed ``image_crop`` and return an L2-normalized tuple of length ``dim``."""

        with Image.open(io.BytesIO(image_crop)) as img:
            rgb = img.convert("RGB")
            tensor: Any = self._preprocess(rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            feat = self._model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        vec = feat.squeeze(0).cpu().tolist()
        return tuple(float(x) for x in vec)
