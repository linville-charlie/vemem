"""InsightFace ArcFace encoder — 512-d face embeddings.

Uses the ``buffalo_l`` model pack (SCRFD-10GF detector + ArcFace recognition
head). On first use, InsightFace downloads the model pack (~200MB) into
``~/.insightface/models/buffalo_l/``. Tests that need real weights are gated
behind ``VEMEM_RUN_INTEGRATION=1`` to keep the default ``pytest`` run fast and
offline-friendly.

Design notes:

- ``Encoder.id`` includes the insightface package version so an upgrade
  produces a different encoder_id — the spec (§3.1a) requires this so new
  embeddings live in separate rows, not overwrites.
- Output is L2-normalized. InsightFace already exposes ``Face.normed_embedding``
  which is unit-norm; we re-normalize defensively in case a future version
  changes that.
- Output is ``tuple[float, ...]`` so the core module stays numpy-free.
- No module-level model cache. The Wave 4 facade can add caching if it proves
  necessary; at encoder construction time the cost is explicit.
"""

from __future__ import annotations

import io
from typing import Any

import insightface
import numpy as np
from PIL import Image


class InsightFaceEncoder:
    """Produces 512-d L2-normalized face embeddings via InsightFace ArcFace.

    The encoder detects the most prominent face in the crop and returns its
    ArcFace embedding. If no face is detected, ``embed`` raises ``RuntimeError``
    — the caller should use an explicit :class:`InsightFaceDetector` to find
    regions before handing crops to ``embed``.
    """

    dim: int = 512

    def __init__(
        self,
        *,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        providers: list[str] | None = None,
    ) -> None:
        self.id = f"insightface/arcface@{insightface.__version__}"
        self._model_name = model_name
        self._app = self._load_app(model_name, providers)
        self._app.prepare(ctx_id=-1, det_size=det_size)

    @staticmethod
    def _load_app(model_name: str, providers: list[str] | None) -> Any:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("insightface is not installed; add it to your dependencies") from exc

        try:
            return FaceAnalysis(
                name=model_name,
                providers=providers or ["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load InsightFace model '{model_name}'. "
                "On first use InsightFace downloads the model pack to "
                "~/.insightface/models/. Ensure network access is available, or "
                "pre-populate the model directory."
            ) from exc

    def embed(self, image_crop: bytes) -> tuple[float, ...]:
        """Embed ``image_crop`` and return a 512-d L2-normalized tuple.

        Raises ``RuntimeError`` if no face is detected in the crop.
        """

        arr = _decode_bgr(image_crop)
        faces = self._app.get(arr)
        if not faces:
            raise RuntimeError("no face detected in crop")

        # pick the highest-scoring detection
        face = max(faces, key=lambda f: float(f.det_score))

        emb = np.asarray(face.normed_embedding, dtype=np.float32)
        # defensive re-normalization
        norm = float(np.linalg.norm(emb))
        if norm == 0.0:
            raise RuntimeError("encoder produced a zero-norm embedding")
        emb = emb / norm

        return tuple(float(x) for x in emb.tolist())


def _decode_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode ``image_bytes`` to a BGR ``HxWx3`` numpy array.

    InsightFace's OpenCV-based pipeline expects BGR; PIL gives us RGB, so we
    flip the last axis. Any PIL-readable format works on the way in.
    """

    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    # RGB -> BGR
    return rgb[:, :, ::-1].copy()
