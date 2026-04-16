"""Encoder Protocol + concrete adapters.

Encoders produce embeddings from image crops. Encoder version is part of the
encoder_id; upgrading an encoder produces new rows, never overwrites — see
docs/spec/identity-semantics.md §3.1a.
"""

from vemem.encoders.clip_encoder import CLIPEncoder
from vemem.encoders.crop import crop_image
from vemem.encoders.insightface_detector import InsightFaceDetector
from vemem.encoders.insightface_encoder import InsightFaceEncoder

__all__ = [
    "CLIPEncoder",
    "InsightFaceDetector",
    "InsightFaceEncoder",
    "crop_image",
]
