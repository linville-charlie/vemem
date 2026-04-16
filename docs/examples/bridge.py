"""Reference bridge between a vision model (VLM) and a text LLM, using vemem.

Why this file exists
--------------------

A text LLM has no persistent identity for the *people and things* it sees.
Every session restarts from scratch: ``"who was that in the photo?"`` has to be
answered fresh every time, and any knowledge you told the assistant about the
person ("Charlie is training for Boston") rots the moment the conversation
ends. That's the problem vemem solves.

The minimal mental model is::

    image  ─►  VLM  ─►  "a person in indoor lighting"           (scene text)
       │
       └──►  vemem.observe()  ─►  detect + embed + persist      (evidence)
                    │
                    ├──►  ops.label()     (user or agent binds obs → entity)
                    ├──►  ops.remember()  (attach a fact to the entity)
                    └──►  ops.identify()  (later image: who is this?)
                                 │
                                 └──►  ops.recall() per candidate
                                           │
                                           └──►  prompt-ready context
                                                       │
                                                       └──►  LLM chat

This script wires that together end-to-end. The VLM and the LLM are replaced
with tiny, deterministic Python stubs so the file runs with zero network calls
or model weights — swap in Qwen2-VL / LLaVA / Ollama / Claude / GPT and the
shape stays identical.

Full spec context: ``docs/spec/identity-semantics.md`` §1 (purpose), §4 (ops).

Run it
------

::

    uv run python docs/examples/bridge.py

See ``docs/examples/README.md`` for how to plug in real models.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from vemem.core import ops
from vemem.core.enums import Modality, Source
from vemem.core.protocols import Clock, Detector, Encoder, Store
from vemem.core.types import Candidate, Observation

# ---------------------------------------------------------------------------
# Fake VLM — describes an image in natural language.
#
# Real replacement: Qwen2-VL / LLaVA / GPT-4o vision / Claude vision. The
# contract is simple: ``bytes -> str``. Everything else in this file stays the
# same when you swap the implementation.
# ---------------------------------------------------------------------------

VlmDescribe = Callable[[bytes], str]


def describe_scene(image_bytes: bytes) -> str:
    """Return a one-line description of what's in the image.

    The canned mapping keeps the demo deterministic. A real VLM call looks
    like::

        def describe_scene(image_bytes: bytes) -> str:
            return qwen_vl.generate(
                images=[image_bytes],
                prompt="Describe the scene in one sentence.",
            )
    """
    # Hash-based "describe" so the demo produces consistent output without a
    # real model. Real VLMs look at pixels; we look at bytes.
    digest = hashlib.sha256(image_bytes).hexdigest()
    bucket = int(digest[:2], 16) % 3
    return (
        "a person smiling in warm indoor lighting",
        "a person outdoors near a running track",
        "two people standing close at an event",
    )[bucket]


# ---------------------------------------------------------------------------
# Fake LLM — takes a user message + context block, returns a reply.
#
# Real replacement: Ollama / Claude / GPT. The ``context`` argument is the
# vemem-assembled prompt text; your real LLM sees it as part of the system or
# user prompt. See ``docs/examples/README.md`` for concrete snippets.
# ---------------------------------------------------------------------------

LlmChat = Callable[[str, str], str]


def chat(user_msg: str, context: str) -> str:
    """Canned conversational reply that echoes back what it saw in context.

    The real LLM version would look like::

        def chat(user_msg: str, context: str) -> str:
            return ollama.chat(
                model="llama3.1",
                messages=[
                    {"role": "system", "content": f"Visual context:\\n{context}"},
                    {"role": "user", "content": user_msg},
                ],
            )
    """
    if "Charlie" in context:
        return f"Based on what I see, Charlie is here. {user_msg.rstrip('?.')} — {context}"
    if "Unknown faces" in context:
        return f"I see an unfamiliar face. {user_msg.rstrip('?.')} — please label them first."
    return f"No visual context to ground this. {user_msg.rstrip('?.')}"


# ---------------------------------------------------------------------------
# Bridge — the glue. One class so it is easy to show in isolation and to test.
# ---------------------------------------------------------------------------


@dataclass
class Bridge:
    """Wires a VLM + vemem + LLM into one observe/identify/chat loop.

    Write path: ``observe()`` runs the detector + encoder over the image and
    persists observations (immutable evidence, spec §3.1). It also asks the
    VLM for a free-form scene description and stores it as an un-bound *Event*
    for later auditing — not attached to any entity yet, because we don't
    know who's in the photo until a human labels it.

    Read path: ``identify_and_recall()`` runs ``ops.identify`` against the
    stored gallery, pulls ``ops.recall`` for each confident match, and
    assembles a short text block ready to paste into an LLM prompt.
    """

    store: Store
    clock: Clock
    encoder: Encoder
    detector: Detector
    vlm: VlmDescribe = field(default=describe_scene)
    llm: LlmChat = field(default=chat)
    actor: str = "bridge:example"
    min_confidence: float = 0.5

    # --- write path --------------------------------------------------------

    def observe(
        self,
        image_bytes: bytes,
        *,
        source_uri: str = "bridge://inline",
        modality: Modality = Modality.FACE,
    ) -> list[Observation]:
        """Detect, embed, and persist. Also asks the VLM for a scene note.

        Thin wrapper around :func:`vemem.pipeline.observe_image` — that helper
        is the shared recipe the MCP server and CLI use too. Observations are
        content-addressed (spec §3.1), so re-observing the same bytes returns
        the same ids.
        """
        from vemem.pipeline import observe_image

        observations = observe_image(
            self.store,
            image_bytes=image_bytes,
            detector=self.detector,
            encoder=self.encoder,
            clock=self.clock,
            modality=modality,
            source_uri=source_uri,
        )

        # Free-text VLM note. Stored as a bridge-level "seen" record in-memory
        # on the Bridge — we intentionally do NOT attach it to an entity,
        # because we don't know which entity this scene belongs to until a
        # user labels the observations. Real integrations often route this
        # through a separate notes table; the library's Event type is tied to
        # an entity_id, so we keep it out of vemem's core tables here.
        scene = self.vlm(image_bytes)
        self._last_scene = scene
        return observations

    # --- read path ---------------------------------------------------------

    def identify_and_recall(self, image_bytes: bytes) -> str:
        """Run identify + recall, return a prompt-ready context string.

        Shape::

            Scene (VLM): <one-line description>
            People visible: <Name> (conf 0.94). Known facts: [fact; fact; ...]
            Unknown faces: <N>
        """
        scene = self.vlm(image_bytes)
        lines: list[str] = [f"Scene (VLM): {scene}."]

        bboxes = self.detector.detect(image_bytes)
        if not bboxes:
            lines.append("No faces detected.")
            return "\n".join(lines)

        named: list[str] = []
        unknown = 0
        for _bbox in bboxes:
            vector = self.encoder.embed(image_bytes)
            candidates = ops.identify(
                self.store,
                encoder_id=self.encoder.id,
                vector=vector,
                k=3,
                min_confidence=self.min_confidence,
            )
            top = _pick_top(candidates)
            if top is None:
                unknown += 1
                continue
            snap = ops.recall(self.store, entity_id=top.entity.id)
            fact_blob = _format_facts(snap.facts)
            named.append(
                f"{top.entity.name} (conf {top.confidence:.2f}). Known facts: [{fact_blob}]"
            )

        if named:
            lines.append("People visible: " + "; ".join(named))
        if unknown:
            lines.append(f"Unknown faces: {unknown} (not yet labeled).")
        return "\n".join(lines)

    # --- glue --------------------------------------------------------------

    def chat_about(self, image_bytes: bytes, user_msg: str) -> str:
        """Assemble visual context and hand it to the LLM."""
        context = self.identify_and_recall(image_bytes)
        return self.llm(user_msg, context)


# --- helpers ---------------------------------------------------------------


def _pick_top(candidates: list[Candidate]) -> Candidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.confidence)


def _format_facts(facts: tuple[object, ...]) -> str:
    # facts is tuple[Fact, ...] but we keep the type loose so the example reads
    # well without importing Fact.
    return "; ".join(getattr(f, "content", str(f)) for f in facts) if facts else "no facts yet"


# ---------------------------------------------------------------------------
# Scripted demo — run ``uv run python docs/examples/bridge.py``.
# ---------------------------------------------------------------------------


class _SystemClock:
    """Real-time clock for the demo."""

    def now(self) -> datetime:
        return datetime.now(tz=UTC)


class _DemoDetector:
    """Stand-in face detector — returns a single bbox for any non-empty image."""

    id: str = "bridge-demo/detector@0"

    def detect(self, image_bytes: bytes) -> list[tuple[int, int, int, int]]:
        return [(10, 10, 100, 100)] if image_bytes else []


class _DemoEncoder:
    """Stand-in face encoder — 8-d hash projection, deterministic."""

    id: str = "bridge-demo/encoder@0"
    dim: int = 8

    def embed(self, image_crop: bytes) -> tuple[float, ...]:
        digest = hashlib.sha256(image_crop).digest()
        raw = [(b - 128) / 128.0 for b in digest[: self.dim]]
        norm = math.sqrt(sum(x * x for x in raw))
        if norm == 0.0:
            return tuple([1.0] + [0.0] * (self.dim - 1))
        return tuple(x / norm for x in raw)


def main() -> None:
    """Run a tiny two-session scenario end-to-end.

    Session 1: we see a stranger; no identity yet, we just observe.
    Session 2: a user labels that observation as "Charlie" and remembers a fact.
    Session 3: a new day, a new photo of the same person — the LLM now has
               persistent context without the user re-introducing Charlie.
    """
    # These imports live inside main() so that importing this module as a
    # library does NOT require FakeStore (which lives under tests/).
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from tests.support.fake_store import FakeStore

    store = FakeStore()
    clock = _SystemClock()
    bridge = Bridge(
        store=store,
        clock=clock,
        encoder=_DemoEncoder(),
        detector=_DemoDetector(),
    )

    # The demo uses deterministic byte strings as stand-ins for image payloads.
    charlie_day_1 = b"fake-image-bytes::charlie-at-meetup"
    charlie_day_2 = b"fake-image-bytes::charlie-at-meetup"  # same person, same "photo"

    print("=== Session 1 — VLM sees, vemem records, no entity yet ===")
    obs_list = bridge.observe(charlie_day_1, source_uri="file:///photos/day1.jpg")
    print(f"VLM says: {bridge._last_scene}")
    print(f"Observations persisted: {[o.id for o in obs_list]}")

    print("\n=== User labels the observation as Charlie + remembers a fact ===")
    entity = ops.label(
        store,
        [obs_list[0].id],
        "Charlie",
        clock=clock,
        actor="user:alice",
    )
    ops.remember(
        store,
        entity_id=entity.id,
        content="training for Boston Marathon",
        source=Source.USER,
        clock=clock,
        actor="user:alice",
    )
    print(f"Entity: {entity.name} ({entity.id})")

    print("\n=== Session 2 — new day, same person; LLM now has persistent context ===")
    reply = bridge.chat_about(charlie_day_2, "How's training going?")
    print(f"Assistant reply: {reply}")


if __name__ == "__main__":
    main()
