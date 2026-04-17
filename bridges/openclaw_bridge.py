"""openclaw vemem bridge — vemem identity + optional VLM scene → thinking LLM.

Images are NEVER sent to the thinking LLM directly. The only ways visual
information reaches the LLM are:
  1. vemem (InsightFace → identify → recall) — ALWAYS on
  2. VLM scene description (qwen3.5:cloud) — OPT-IN via --use-vlm

    image ─► InsightFace detect/embed ─► vemem identify + recall ──┐
       │                                                            │
       └─► [optional, --use-vlm]  qwen3.5:cloud scene text ─────────┤
                                                                    ▼
                                                            TEXT-ONLY context
                                                                    │
                                                                    ▼
                                                         glm-5.1:cloud (thinking)

No image bytes ever reach the thinking LLM. _assert_text_only() guards the
outbound message and raises RuntimeError if anything tries.

Usage:

    # Label a face + fact (one-time)
    uv run python bridges/openclaw_bridge.py label charlie.jpg Charlie \\
        --fact "training for Boston Marathon"

    # Ask — vemem only (safest, what Ella uses by default)
    uv run python bridges/openclaw_bridge.py ask photo.jpg "How's training going?"

    # Ask — vemem + VLM scene (for full testing)
    uv run python bridges/openclaw_bridge.py ask photo.jpg "What's happening?" \\
        --use-vlm

Env vars:
    VEMEM_HOME     — where LanceDB lives (default ~/.vemem)
    OLLAMA_HOST    — ollama endpoint (default http://localhost:11434)
    LLM_MODEL      — thinking model (default glm-5.1:cloud)
    VLM_MODEL      — vision model used only when --use-vlm (default qwen3.5:cloud)
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from ollama import Client
except ImportError as _exc:  # optional dep — keep import error actionable
    raise SystemExit(
        "openclaw_bridge requires the `ollama` Python client.\n"
        "Install with:  uv pip install ollama\n"
        f"(original error: {_exc})"
    ) from _exc

from vemem.core import ops
from vemem.core.enums import Source
from vemem.encoders.insightface_detector import InsightFaceDetector
from vemem.encoders.insightface_encoder import InsightFaceEncoder
from vemem.pipeline import observe_image
from vemem.storage.lancedb_store import LanceDBStore

LLM_MODEL = os.environ.get("LLM_MODEL", "glm-5.1:cloud")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3.5:cloud")
VEMEM_HOME = os.environ.get("VEMEM_HOME", str(Path.home() / ".vemem"))


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(tz=UTC)


def _assert_text_only(messages: list[dict]) -> None:
    """Guard: no image payloads must ever reach the thinking LLM."""
    for m in messages:
        if m.get("images"):
            raise RuntimeError(
                "Bridge invariant violated: attempted to send images to the "
                "thinking LLM. Images must flow only through vemem."
            )
        content = m.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image", "image_url"):
                    raise RuntimeError(
                        "Bridge invariant violated: image content-part found in LLM message."
                    )


def think(client: Client, user_msg: str, context: str) -> tuple[str, str]:
    """Thinking LLM: reasons over vemem-assembled TEXT context only."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. The user is asking about a photo. "
                "A local face-recognition system (vemem) has processed the image "
                "and extracted the identity + facts below. You do NOT see the image; "
                "reason only from this text context."
                f"\n\nvemem context:\n{context}"
            ),
        },
        {"role": "user", "content": user_msg},
    ]
    _assert_text_only(messages)
    result = client.chat(model=LLM_MODEL, messages=messages)
    msg = result.get("message", {})
    return msg.get("content", "").strip(), msg.get("thinking", "").strip()


def describe_scene(client: Client, image_bytes: bytes) -> str:
    """VLM: one-sentence scene description. Contract: bytes → str.

    Called ONLY when --use-vlm is set. The resulting string is incorporated
    into the text context; the image bytes themselves never reach the LLM.
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")
    result = client.chat(
        model=VLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Describe the scene in one sentence. Focus on who is visible "
                    "and what they're doing."
                ),
                "images": [b64],
            }
        ],
    )
    return result["message"]["content"].strip()


def build_context(
    store,
    detector,
    encoder,
    image_bytes: bytes,
    *,
    vlm_client: Client | None = None,
) -> str:
    """Build text context from vemem (always) and VLM (if vlm_client given).

    The return value is the ONLY thing that reaches the thinking LLM. No
    image bytes are ever embedded in it.
    """
    lines: list[str] = []
    if vlm_client is not None:
        scene = describe_scene(vlm_client, image_bytes)
        lines.append(f"Scene (VLM): {scene}")
    bboxes = detector.detect(image_bytes)
    if not bboxes:
        lines.append("vemem: no faces detected in image.")
        return "\n".join(lines)

    lines.append(f"vemem: {len(bboxes)} face(s) detected.")

    named: list[str] = []
    unknown = 0
    for bbox in bboxes:
        # Prefer embed_frame (encoders with internal detect+align, e.g.
        # InsightFace) over embed(full) — otherwise every face in a
        # multi-face image gets the same embedding.
        if hasattr(encoder, "embed_frame"):
            vector = encoder.embed_frame(image_bytes, bbox)
        else:
            vector = encoder.embed(image_bytes)
        candidates = ops.identify(
            store,
            encoder_id=encoder.id,
            vector=vector,
            k=3,
            min_confidence=0.5,
        )
        if not candidates:
            unknown += 1
            continue
        top = max(candidates, key=lambda c: c.confidence)
        snap = ops.recall(store, entity_id=top.entity.id)
        facts = "; ".join(f.content for f in snap.facts) if snap.facts else "no facts yet"
        named.append(f"{top.entity.name} (conf {top.confidence:.2f}). Known facts: [{facts}]")

    if named:
        lines.append("People recognized: " + "; ".join(named))
    if unknown:
        lines.append(f"Unrecognized faces: {unknown} (not yet labeled).")
    return "\n".join(lines)


def cmd_label(args):
    """Observe + label an image, optionally attaching a fact."""
    image_bytes = Path(args.image).read_bytes()
    store = LanceDBStore(path=VEMEM_HOME)
    detector = InsightFaceDetector()
    encoder = InsightFaceEncoder()
    clock = SystemClock()

    observations = observe_image(
        store,
        image_bytes=image_bytes,
        detector=detector,
        encoder=encoder,
        clock=clock,
        source_uri=f"file://{Path(args.image).resolve()}",
    )
    if not observations:
        print("No face detected in image.", file=sys.stderr)
        return 1

    entity = ops.label(
        store,
        [observations[0].id],
        args.name,
        clock=clock,
        actor=f"user:{os.environ.get('USER', 'unknown')}",
    )
    print(f"Labeled {len(observations)} face(s) as {entity.name} ({entity.id})")

    if args.fact:
        ops.remember(
            store,
            entity_id=entity.id,
            content=args.fact,
            source=Source.USER,
            clock=clock,
            actor=f"user:{os.environ.get('USER', 'unknown')}",
        )
        print(f"Remembered: {args.fact}")
    return 0


def cmd_ask(args):
    """vemem identify/recall (+ optional VLM scene) → TEXT context → thinking LLM."""
    image_bytes = Path(args.image).read_bytes()
    store = LanceDBStore(path=VEMEM_HOME)
    detector = InsightFaceDetector()
    encoder = InsightFaceEncoder()
    client = Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    vlm_client = client if args.use_vlm else None

    context = build_context(store, detector, encoder, image_bytes, vlm_client=vlm_client)
    header = "--- vemem + VLM context ---" if args.use_vlm else "--- vemem-only context ---"
    print(f"{header}  (only text below reaches the LLM)")
    print(context)

    answer, trace = think(client, args.question, context)
    if trace and args.show_thinking:
        print("\n--- Thinking trace ---")
        print(trace)
    print("\n--- Answer ---")
    print(answer)
    return 0


def cmd_observe(args):
    """Just observe: detect + embed + persist. No labeling, no LLM."""
    image_bytes = Path(args.image).read_bytes()
    store = LanceDBStore(path=VEMEM_HOME)
    detector = InsightFaceDetector()
    encoder = InsightFaceEncoder()
    clock = SystemClock()

    observations = observe_image(
        store,
        image_bytes=image_bytes,
        detector=detector,
        encoder=encoder,
        clock=clock,
        source_uri=f"file://{Path(args.image).resolve()}",
    )
    for obs in observations:
        print(f"{obs.id}  bbox={obs.bbox}")
    print(f"\n{len(observations)} face(s) persisted to {VEMEM_HOME}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    p_obs = sub.add_parser("observe", help="detect + embed + persist a face")
    p_obs.add_argument("image", type=str)
    p_obs.set_defaults(fn=cmd_observe)

    p_lbl = sub.add_parser("label", help="label a face + optional fact")
    p_lbl.add_argument("image", type=str)
    p_lbl.add_argument("name", type=str)
    p_lbl.add_argument("--fact", type=str, default=None)
    p_lbl.set_defaults(fn=cmd_label)

    p_ask = sub.add_parser("ask", help="ask the thinking LLM about an image")
    p_ask.add_argument("image", type=str)
    p_ask.add_argument("question", type=str)
    p_ask.add_argument(
        "--use-vlm",
        action="store_true",
        help="also feed qwen3.5:cloud's scene description into the text context",
    )
    p_ask.add_argument("--show-thinking", action="store_true")
    p_ask.set_defaults(fn=cmd_ask)

    args = p.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
