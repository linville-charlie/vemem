"""End-to-end tests for ``docs/examples/bridge.py``.

The example is a script, not a package module, so we load it via
``importlib`` (see ``conftest.py``) and exercise the ``Bridge`` class with a
FakeStore + stub detector/encoder + stub VLM/LLM. Real model weights are never
loaded.

What we're proving:
- The write path (``observe()``) persists Observations + Embeddings.
- Re-observing the same bytes is idempotent.
- The read path (``identify_and_recall()``) surfaces labeled entities by name
  and the facts attached to them.
- Unknown faces are reported as ``Unknown faces: N``.
- The glue (``chat_about()``) hands the assembled context to the LLM.
- The scripted ``main()`` runs without exceptions and prints the expected
  session markers.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from types import ModuleType

from tests.examples.conftest import (
    FixedClock,
    RecordingLLM,
    RecordingVLM,
    StubDetector,
    StubEncoder,
)
from tests.support.fake_store import FakeStore
from vemem.core import ops
from vemem.core.enums import Source


def _make_bridge(
    bridge_module: ModuleType,
    store: FakeStore,
    clock: FixedClock,
    encoder: StubEncoder,
    detector: StubDetector,
    vlm: RecordingVLM,
    llm: RecordingLLM,
):
    return bridge_module.Bridge(
        store=store,
        clock=clock,
        encoder=encoder,
        detector=detector,
        vlm=vlm,
        llm=llm,
        actor="test:bridge",
    )


# ---------- write path ----------


def test_observe_persists_observation_and_embedding(
    bridge_module: ModuleType,
    store: FakeStore,
    clock: FixedClock,
    encoder: StubEncoder,
    detector: StubDetector,
    vlm: RecordingVLM,
    llm: RecordingLLM,
) -> None:
    bridge = _make_bridge(bridge_module, store, clock, encoder, detector, vlm, llm)

    observations = bridge.observe(b"image-bytes-for-charlie")

    assert len(observations) == 1
    obs = observations[0]
    assert obs.id.startswith("obs_")
    # One observation row, one embedding row.
    assert store.get_observation(obs.id) is obs
    assert len(store._embeddings) == 1
    # VLM got consulted during observe() (stored scene description).
    assert vlm.calls == 1


def test_observe_is_idempotent_on_same_bytes(
    bridge_module: ModuleType,
    store: FakeStore,
    clock: FixedClock,
    encoder: StubEncoder,
    detector: StubDetector,
    vlm: RecordingVLM,
    llm: RecordingLLM,
) -> None:
    bridge = _make_bridge(bridge_module, store, clock, encoder, detector, vlm, llm)

    first = bridge.observe(b"same-image-bytes")
    second = bridge.observe(b"same-image-bytes")

    # Same image -> same observation id (content-addressed per §3.1).
    assert first[0].id == second[0].id
    assert len(store._observations) == 1
    # Both calls appended an embedding (per spec §3.1a one per call is fine;
    # the bridge doesn't dedupe embeddings, matching the MCP tool behavior).
    assert len(store._embeddings) == 2


# ---------- read path ----------


def test_identify_and_recall_surfaces_labeled_entity(
    bridge_module: ModuleType,
    store: FakeStore,
    clock: FixedClock,
    encoder: StubEncoder,
    detector: StubDetector,
    vlm: RecordingVLM,
    llm: RecordingLLM,
) -> None:
    bridge = _make_bridge(bridge_module, store, clock, encoder, detector, vlm, llm)

    obs = bridge.observe(b"charlie-image-bytes")[0]
    entity = ops.label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")
    ops.remember(
        store,
        entity_id=entity.id,
        content="training for Boston Marathon",
        source=Source.USER,
        clock=clock,
        actor="user:alice",
    )

    context = bridge.identify_and_recall(b"charlie-image-bytes")

    assert "Charlie" in context
    assert "training for Boston Marathon" in context
    assert "conf 1.00" in context  # user_label confidence


def test_identify_and_recall_flags_unknown_faces(
    bridge_module: ModuleType,
    store: FakeStore,
    clock: FixedClock,
    encoder: StubEncoder,
    detector: StubDetector,
    vlm: RecordingVLM,
    llm: RecordingLLM,
) -> None:
    bridge = _make_bridge(bridge_module, store, clock, encoder, detector, vlm, llm)

    # Never labeled; empty gallery => no candidates => unknown.
    context = bridge.identify_and_recall(b"stranger-image-bytes")

    assert "Unknown faces: 1" in context
    assert "Charlie" not in context


def test_identify_and_recall_reports_no_faces_when_detector_empty(
    bridge_module: ModuleType,
    store: FakeStore,
    clock: FixedClock,
    encoder: StubEncoder,
    detector: StubDetector,
    vlm: RecordingVLM,
    llm: RecordingLLM,
) -> None:
    bridge = _make_bridge(bridge_module, store, clock, encoder, detector, vlm, llm)

    # Empty bytes -> StubDetector returns zero bboxes.
    context = bridge.identify_and_recall(b"")

    assert "No faces detected." in context


# ---------- glue ----------


def test_chat_about_passes_context_to_llm(
    bridge_module: ModuleType,
    store: FakeStore,
    clock: FixedClock,
    encoder: StubEncoder,
    detector: StubDetector,
    vlm: RecordingVLM,
    llm: RecordingLLM,
) -> None:
    bridge = _make_bridge(bridge_module, store, clock, encoder, detector, vlm, llm)

    obs = bridge.observe(b"charlie-image-bytes")[0]
    entity = ops.label(store, [obs.id], "Charlie", clock=clock, actor="user:alice")
    ops.remember(
        store,
        entity_id=entity.id,
        content="met at OSS conference",
        source=Source.USER,
        clock=clock,
        actor="user:alice",
    )
    llm.reply = "stub-reply"

    reply = bridge.chat_about(b"charlie-image-bytes", "who is this?")

    assert reply == "stub-reply"
    assert llm.last_user_msg == "who is this?"
    assert llm.last_context is not None
    assert "Charlie" in llm.last_context
    assert "met at OSS conference" in llm.last_context


def test_chat_about_on_unknown_entity_still_invokes_llm(
    bridge_module: ModuleType,
    store: FakeStore,
    clock: FixedClock,
    encoder: StubEncoder,
    detector: StubDetector,
    vlm: RecordingVLM,
    llm: RecordingLLM,
) -> None:
    bridge = _make_bridge(bridge_module, store, clock, encoder, detector, vlm, llm)
    llm.reply = "stub-reply-for-unknown"

    reply = bridge.chat_about(b"stranger-image-bytes", "who is this?")

    assert reply == "stub-reply-for-unknown"
    assert llm.last_context is not None
    assert "Unknown faces: 1" in llm.last_context


# ---------- scripted demo ----------


def test_main_demo_runs_end_to_end(bridge_module: ModuleType) -> None:
    """The ``main()`` scripted scenario must run without exceptions.

    This is the canonical output a user sees after ``uv run python
    docs/examples/bridge.py``. We capture stdout and sanity-check the session
    markers — if the demo breaks, this test is the fast signal.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        bridge_module.main()

    output = buf.getvalue()
    assert "Session 1" in output
    assert "Charlie" in output  # entity got labeled
    assert "Session 2" in output
    assert "Assistant reply" in output
