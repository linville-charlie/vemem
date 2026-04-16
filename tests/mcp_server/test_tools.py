"""Unit tests for every MCP tool handler.

These tests exercise the handler functions directly, with an in-memory
FakeStore + deterministic stub encoder/detector injected via the ``ctx``
fixture. They are fast (no subprocess, no network) and cover happy paths plus
at least one error path per tool.

The goal is NOT to re-test the semantics of the core ops (those live in
``tests/core/test_ops_*.py``) — it's to catch wire-level issues: argument
routing, serialization of return values, error surfacing at the MCP layer.
"""

from __future__ import annotations

import base64

import pytest

from tests.mcp_server.conftest import FixedClock
from tests.support.fake_store import FakeStore
from vemem.core.errors import (
    EntityUnavailableError,
    OperationNotReversibleError,
)
from vemem.mcp_server import tools
from vemem.mcp_server.tools import ServerContext

# ---------- helpers ----------


IMAGE_BYTES = b"fake-image-bytes-for-tests"
IMAGE_B64 = base64.b64encode(IMAGE_BYTES).decode("ascii")


def _seed_label(ctx: ServerContext, name: str = "Charlie") -> tuple[str, str]:
    """Run observe + label and return (entity_id, observation_id)."""
    observed = tools.observe_image(ctx, image_base64=IMAGE_B64)
    obs_id = observed["observations"][0]["id"]
    entity_dict = tools.label_tool(
        ctx,
        observation_ids=[obs_id],
        entity_name_or_id=name,
        actor="test:labeler",
    )
    return entity_dict["id"], obs_id


# ---------- observe_image ----------


def test_observe_image_creates_observations_and_embeddings(ctx: ServerContext) -> None:
    result = tools.observe_image(ctx, image_base64=IMAGE_B64)

    assert len(result["observations"]) == 1
    obs_entry = result["observations"][0]
    assert obs_entry["id"].startswith("obs_")
    assert obs_entry["bbox"] == [10, 10, 100, 100]
    assert obs_entry["detector_id"] == "test-stub/detector@0"
    assert isinstance(ctx.store, FakeStore)
    assert len(ctx.store._embeddings) == 1


def test_observe_image_is_idempotent_on_same_bytes(ctx: ServerContext) -> None:
    first = tools.observe_image(ctx, image_base64=IMAGE_B64)
    second = tools.observe_image(ctx, image_base64=IMAGE_B64)

    # Same image → same observation id (content-addressed per §3.1).
    assert first["observations"][0]["id"] == second["observations"][0]["id"]
    # Only one observation stored, but both calls appended embeddings.
    assert isinstance(ctx.store, FakeStore)
    assert len(ctx.store._observations) == 1


def test_observe_image_rejects_bad_base64(ctx: ServerContext) -> None:
    with pytest.raises(ValueError, match="valid base64"):
        tools.observe_image(ctx, image_base64="not=valid=base64==!!!")


def test_observe_image_errors_when_encoder_missing(
    ctx_without_encoder: ServerContext,
) -> None:
    with pytest.raises(RuntimeError, match="insightface weights not found"):
        tools.observe_image(ctx_without_encoder, image_base64=IMAGE_B64)


# ---------- identify_image ----------


def test_identify_image_returns_candidates_for_seeded_entity(ctx: ServerContext) -> None:
    # Seed a labeled Charlie against the stub encoder's vector for IMAGE_BYTES.
    _seed_label(ctx, name="Charlie")

    result = tools.identify_image(ctx, image_base64=IMAGE_B64)

    assert result["encoder_id"] == "test-stub/encoder@0"
    assert len(result["detections"]) == 1
    candidates = result["detections"][0]["candidates"]
    assert any(c["entity"]["name"] == "Charlie" for c in candidates)


def test_identify_image_empty_when_gallery_empty(ctx: ServerContext) -> None:
    result = tools.identify_image(ctx, image_base64=IMAGE_B64)
    assert result["detections"][0]["candidates"] == []


def test_identify_image_errors_when_encoder_missing(
    ctx_without_encoder: ServerContext,
) -> None:
    with pytest.raises(RuntimeError, match="insightface weights not found"):
        tools.identify_image(ctx_without_encoder, image_base64=IMAGE_B64)


# ---------- identify_by_name ----------


def test_identify_by_name_returns_recall_snapshot(ctx: ServerContext) -> None:
    entity_id, _obs = _seed_label(ctx, name="Dana")
    tools.remember_tool(
        ctx,
        entity_id=entity_id,
        content="wears red shoes",
        actor="test:narrator",
    )

    snap = tools.identify_by_name(ctx, entity_name_or_id="Dana")

    assert snap["entity"]["name"] == "Dana"
    assert len(snap["facts"]) == 1
    assert snap["facts"][0]["content"] == "wears red shoes"


def test_identify_by_name_rejects_unknown_name(ctx: ServerContext) -> None:
    with pytest.raises(ValueError, match="no entity found"):
        tools.identify_by_name(ctx, entity_name_or_id="Nobody")


# ---------- label ----------


def test_label_creates_entity_and_binding(ctx: ServerContext) -> None:
    observed = tools.observe_image(ctx, image_base64=IMAGE_B64)
    obs_id = observed["observations"][0]["id"]

    result = tools.label_tool(
        ctx,
        observation_ids=[obs_id],
        entity_name_or_id="Elena",
        actor="test:editor",
    )

    assert result["name"] == "Elena"
    assert result["status"] == "active"
    assert result["id"].startswith("ent_")


# ---------- relabel ----------


def test_relabel_moves_observation(ctx: ServerContext) -> None:
    entity_id, obs_id = _seed_label(ctx, name="Frank")

    result = tools.relabel_tool(
        ctx,
        observation_id=obs_id,
        new_entity_name_or_id="Gail",
        actor="test:editor",
    )

    assert result["name"] == "Gail"
    assert result["id"] != entity_id


# ---------- merge ----------


def test_merge_folds_losers_into_winner(ctx: ServerContext, clock: FixedClock) -> None:
    # Seed two entities with different observations.
    e1, _ = _seed_label(ctx, name="Henry")
    # Second image has different bytes → different observation id
    second_b64 = base64.b64encode(b"different-image-bytes").decode("ascii")
    second = tools.observe_image(ctx, image_base64=second_b64)
    e2_entity = tools.label_tool(
        ctx,
        observation_ids=[second["observations"][0]["id"]],
        entity_name_or_id="Henry2",
    )
    e2 = e2_entity["id"]

    winner = tools.merge_tool(ctx, entity_ids=[e1, e2], keep="oldest")

    assert winner["id"] == e1
    assert winner["status"] == "active"


def test_merge_rejects_single_entity(ctx: ServerContext) -> None:
    e1, _ = _seed_label(ctx, name="Ian")
    with pytest.raises(ValueError, match="at least two"):
        tools.merge_tool(ctx, entity_ids=[e1])


# ---------- split ----------


def test_split_returns_entity_list(ctx: ServerContext) -> None:
    entity_id, obs_id = _seed_label(ctx, name="Jamie")
    # Add a second observation to split off
    second_b64 = base64.b64encode(b"split-second").decode("ascii")
    second = tools.observe_image(ctx, image_base64=second_b64)
    obs2 = second["observations"][0]["id"]
    tools.label_tool(
        ctx,
        observation_ids=[obs2],
        entity_name_or_id=entity_id,  # same entity
    )

    result = tools.split_tool(
        ctx,
        entity_id=entity_id,
        groups=[[obs_id], [obs2]],
    )

    assert len(result["entities"]) == 2
    assert result["entities"][0]["id"] == entity_id


# ---------- forget ----------


def test_forget_returns_counts(ctx: ServerContext) -> None:
    entity_id, _obs = _seed_label(ctx, name="Karen")

    result = tools.forget_tool(ctx, entity_id=entity_id, actor="test:eraser")

    assert result["entity_id"] == entity_id
    assert result["counts"]["observations"] >= 1
    assert result["counts"]["bindings"] >= 1


def test_forget_rejects_unknown_entity(ctx: ServerContext) -> None:
    with pytest.raises(EntityUnavailableError):
        tools.forget_tool(ctx, entity_id="ent_does_not_exist", actor="test:eraser")


# ---------- restrict / unrestrict ----------


def test_restrict_then_unrestrict_roundtrip(ctx: ServerContext) -> None:
    entity_id, _ = _seed_label(ctx, name="Liam")

    restricted = tools.restrict_tool(ctx, entity_id=entity_id, actor="test:privacy")
    assert restricted["status"] == "restricted"

    active = tools.unrestrict_tool(ctx, entity_id=entity_id, actor="test:privacy")
    assert active["status"] == "active"


def test_restrict_rejects_already_restricted(ctx: ServerContext) -> None:
    entity_id, _ = _seed_label(ctx, name="Mia")
    tools.restrict_tool(ctx, entity_id=entity_id, actor="test:privacy")

    with pytest.raises(EntityUnavailableError, match="restrict"):
        tools.restrict_tool(ctx, entity_id=entity_id, actor="test:privacy")


# ---------- remember / recall ----------


def test_remember_and_recall(ctx: ServerContext) -> None:
    entity_id, _ = _seed_label(ctx, name="Noah")

    fact = tools.remember_tool(
        ctx,
        entity_id=entity_id,
        content="works at Acme",
        actor="test:narrator",
    )
    assert fact["content"] == "works at Acme"
    assert fact["source"] == "user"

    snap = tools.recall_tool(ctx, entity_id=entity_id)
    assert snap["entity"]["id"] == entity_id
    assert len(snap["facts"]) == 1


def test_remember_rejects_unknown_entity(ctx: ServerContext) -> None:
    with pytest.raises(EntityUnavailableError):
        tools.remember_tool(
            ctx,
            entity_id="ent_nope",
            content="phantom",
            actor="test:narrator",
        )


# ---------- undo ----------


def test_undo_rejects_when_no_reversible_events(ctx: ServerContext) -> None:
    with pytest.raises(OperationNotReversibleError):
        tools.undo_tool(ctx, actor="test:editor")


def test_undo_reverts_last_reversible_op_by_actor(ctx: ServerContext) -> None:
    entity_id, _ = _seed_label(ctx, name="Olivia")
    tools.remember_tool(
        ctx,
        entity_id=entity_id,
        content="loves opera",
        actor="test:narrator",
    )

    # The most recent reversible event by "test:narrator" is the remember.
    new_event = tools.undo_tool(ctx, actor="test:narrator")

    assert new_event["op_type"] == "undo"
    assert new_event["payload"]["original_op_type"] == "remember"


def test_undo_rejects_forget(ctx: ServerContext) -> None:
    entity_id, _ = _seed_label(ctx, name="Pat")
    tools.forget_tool(ctx, entity_id=entity_id, actor="test:eraser")

    with pytest.raises(OperationNotReversibleError):
        tools.undo_tool(ctx, actor="test:eraser")


# ---------- export ----------


def test_export_returns_structured_dump(ctx: ServerContext) -> None:
    entity_id, obs_id = _seed_label(ctx, name="Quinn")
    tools.remember_tool(
        ctx,
        entity_id=entity_id,
        content="quiet",
        actor="test:narrator",
    )

    dump = tools.export_tool(ctx, entity_id=entity_id, include_embeddings=False)

    assert dump["entity"]["id"] == entity_id
    assert dump["observations"][0]["id"] == obs_id
    assert dump["facts"][0]["content"] == "quiet"
    assert dump["embeddings"] == []  # include_embeddings=False


def test_export_rejects_unknown_entity(ctx: ServerContext) -> None:
    with pytest.raises(EntityUnavailableError):
        tools.export_tool(ctx, entity_id="ent_none")
