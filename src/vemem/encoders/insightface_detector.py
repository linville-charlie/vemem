"""InsightFace face detector — produces ``(x, y, w, h)`` bboxes.

Wraps InsightFace's ``FaceAnalysis`` pipeline (SCRFD-10GF detector from
``buffalo_l``). The detector and encoder are intentionally separate adapters
(both satisfying :class:`vemem.core.protocols.Detector` /
:class:`vemem.core.protocols.Encoder`) even though InsightFace naturally fuses
them. Keeping them split:

- lets callers drop in a different detector (e.g. YOLO-face) without touching
  the encoder,
- keeps the :class:`Detector` Protocol small and modality-agnostic,
- matches the spec's separation of Observation (detected region) from
  Embedding (one vector per encoder) in §3.1 / §3.1a.

If callers want the fused fast path, they can simply run both adapters on the
same ``FaceAnalysis`` instance — a ``detect_and_embed`` fused Protocol method
is an open question (see ``DONE.md``).
"""

from __future__ import annotations

from typing import Any

import insightface

from vemem.encoders.insightface_encoder import _decode_bgr


class InsightFaceDetector:
    """Detects faces and returns ``(x, y, w, h)`` integer bboxes in pixel space."""

    def __init__(
        self,
        *,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        providers: list[str] | None = None,
    ) -> None:
        self.id = f"insightface/{model_name}@{insightface.__version__}"
        self._model_name = model_name
        self._app = self._load_app(model_name, providers)
        self._app.prepare(ctx_id=-1, det_size=det_size, det_thresh=det_thresh)

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

    def detect(self, image_bytes: bytes) -> list[tuple[int, int, int, int]]:
        """Return ``(x, y, w, h)`` bboxes for all detected faces."""

        arr = _decode_bgr(image_bytes)
        faces = self._app.get(arr)
        bboxes: list[tuple[int, int, int, int]] = []
        for face in faces:
            # InsightFace's bbox is (x1, y1, x2, y2)
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            x = round(x1)
            y = round(y1)
            w = round(x2 - x1)
            h = round(y2 - y1)
            if w > 0 and h > 0:
                bboxes.append((x, y, w, h))
        return bboxes
