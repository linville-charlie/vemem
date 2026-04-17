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

        InsightFace's recognition pipeline does its own detection + landmark
        alignment internally. If the caller passes a tight bbox crop with no
        context around the face, that internal detector will often fail — the
        crop is essentially all-face-no-frame, which the SCRFD detector wasn't
        trained on.

        For callers that already have a bbox and the full image, use
        :meth:`embed_frame` instead; it sidesteps this class of failure.

        Raises ``RuntimeError`` if no face is detected in the crop.
        """

        return self._embed_numpy(_decode_bgr(image_crop))

    def embed_frame(
        self,
        image_bytes: bytes,
        bbox: tuple[int, int, int, int],
    ) -> tuple[float, ...]:
        """Embed the face inside ``bbox`` on the full frame.

        Pipeline calls this when it has a full image + a detector-emitted
        bbox. We run InsightFace's own detect+align on the full frame (which
        is what InsightFace expects — it does the crop and alignment itself
        from landmarks), then pick the detected face whose bbox best overlaps
        the caller's bbox by IoU. That way the pipeline's detector decides
        *which* face to embed, even when InsightFace's internal detector
        finds several.

        Raises ``RuntimeError`` if InsightFace finds no faces or the best
        match has zero overlap with the requested bbox.
        """

        arr = _decode_bgr(image_bytes)
        faces = self._app.get(arr)
        if not faces:
            raise RuntimeError(f"no face detected in frame for bbox {bbox}")

        target = _select_face_for_bbox(faces, bbox)
        if target is None:
            raise RuntimeError(
                f"no detected face overlaps requested bbox {bbox}; "
                f"insightface found {len(faces)} face(s) elsewhere"
            )
        return _normalize(target.normed_embedding)

    def _embed_numpy(self, arr: np.ndarray) -> tuple[float, ...]:
        faces = self._app.get(arr)
        if not faces:
            raise RuntimeError("no face detected in crop")

        # pick the highest-scoring detection
        face = max(faces, key=lambda f: float(f.det_score))
        return _normalize(face.normed_embedding)


def _select_face_for_bbox(
    faces: list[Any],
    target_bbox: tuple[int, int, int, int],
    *,
    min_iou: float = 0.1,
) -> Any | None:
    """Return the face whose InsightFace bbox best overlaps ``target_bbox``.

    Accepts any positive overlap above a small floor so a slightly-different
    bounding algorithm between the pipeline's detector and InsightFace's
    internal SCRFD doesn't cause a spurious miss. Returns ``None`` if no face
    clears the floor.
    """

    tx, ty, tw, th = target_bbox
    target_xyxy = (tx, ty, tx + tw, ty + th)
    best: tuple[float, Any] | None = None
    for face in faces:
        # InsightFace returns face.bbox as a numpy [x1, y1, x2, y2]
        fb = [float(v) for v in face.bbox]
        iou = _iou(target_xyxy, (fb[0], fb[1], fb[2], fb[3]))
        if iou >= min_iou and (best is None or iou > best[0]):
            best = (iou, face)
    return best[1] if best else None


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _normalize(embedding: np.ndarray) -> tuple[float, ...]:
    """L2-normalize ``embedding`` and return as a plain float tuple."""

    emb = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    if norm == 0.0:
        raise RuntimeError("encoder produced a zero-norm embedding")
    return tuple(float(x) for x in (emb / norm).tolist())


def _decode_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode ``image_bytes`` to a BGR ``HxWx3`` numpy array.

    InsightFace's OpenCV-based pipeline expects BGR; PIL gives us RGB, so we
    flip the last axis. Any PIL-readable format works on the way in.
    """

    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    # RGB -> BGR
    return rgb[:, :, ::-1].copy()
