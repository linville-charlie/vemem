"""CliRunner-based tests for every ``vm`` command.

These exercise the Typer app in-process against a FakeStore + stub
encoder/detector (installed via the ``install_ctx`` fixture). No subprocess,
no real LanceDB, no model weights — the goal is to pin the CLI wiring, argument
routing, and output format rather than re-test core op semantics.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from typer.testing import CliRunner

from tests.support.fake_store import FakeStore
from vemem.cli.app import app
from vemem.cli.context import CliContext
from vemem.core import ops
from vemem.core.enums import Source

IMAGE_BYTES = b"fake-image-bytes-for-cli-tests"


def _write_image(tmp_path: Path, name: str = "face.png") -> Path:
    path = tmp_path / name
    path.write_bytes(IMAGE_BYTES)
    return path


def _seed_entity(ctx: CliContext, name: str = "Charlie") -> tuple[str, str]:
    """Seed store with an observed + labeled entity, returning (entity_id, obs_id)."""
    from vemem.cli import commands_util

    obs_ids = commands_util.ingest_image(
        ctx,
        image_bytes=IMAGE_BYTES,
        source_uri="file:///fake.png",
    )
    entity = ops.label(ctx.store, obs_ids, name, clock=ctx.clock, actor=ctx.actor)
    return entity.id, obs_ids[0]


# ---------- root / help ----------


def test_root_help_lists_subcommands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Spot-check a handful of registered subcommands
    for cmd in [
        "observe",
        "identify",
        "label",
        "remember",
        "recall",
        "merge",
        "forget",
        "list",
        "inspect",
        "export",
        "export-tools",
    ]:
        assert cmd in result.stdout


# ---------- observe ----------


def test_observe_prints_observation_ids(
    runner: CliRunner,
    install_ctx: CliContext,
    tmp_path: Path,
) -> None:
    img = _write_image(tmp_path)
    result = runner.invoke(app, ["observe", str(img)])

    assert result.exit_code == 0
    assert "obs_" in result.stdout
    assert isinstance(install_ctx.store, FakeStore)
    assert len(install_ctx.store._observations) == 1


def test_observe_missing_encoder_prints_install_hint(
    runner: CliRunner,
    install_ctx_no_encoder: CliContext,
    tmp_path: Path,
) -> None:
    img = _write_image(tmp_path)
    result = runner.invoke(app, ["observe", str(img)])

    assert result.exit_code == 2
    assert "insightface" in result.stdout.lower()


def test_observe_missing_file_returns_user_error(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    result = runner.invoke(app, ["observe", "/nonexistent/path.png"])
    assert result.exit_code == 1


# ---------- identify ----------


def test_identify_returns_table_with_candidate(
    runner: CliRunner,
    install_ctx: CliContext,
    tmp_path: Path,
) -> None:
    _seed_entity(install_ctx, name="Charlie")
    img = _write_image(tmp_path)
    result = runner.invoke(app, ["identify", str(img)])

    assert result.exit_code == 0
    assert "Charlie" in result.stdout


def test_identify_json_format(
    runner: CliRunner,
    install_ctx: CliContext,
    tmp_path: Path,
) -> None:
    _seed_entity(install_ctx, name="Charlie")
    img = _write_image(tmp_path)
    result = runner.invoke(app, ["identify", str(img), "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "detections" in payload


# ---------- label / relabel ----------


def test_label_creates_new_entity(
    runner: CliRunner,
    install_ctx: CliContext,
    tmp_path: Path,
) -> None:
    img = _write_image(tmp_path)
    observe_result = runner.invoke(app, ["observe", str(img)])
    assert observe_result.exit_code == 0
    obs_id = next(
        line.split()[0]
        for line in observe_result.stdout.splitlines()
        if line.strip().startswith("obs_")
    )

    result = runner.invoke(app, ["label", obs_id, "--name", "Dana"])
    assert result.exit_code == 0
    assert "Dana" in result.stdout or "ent_" in result.stdout


def test_relabel_moves_observation(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    _entity_id, obs_id = _seed_entity(install_ctx, name="Charlie")
    result = runner.invoke(app, ["relabel", obs_id, "--name", "Dana"])

    assert result.exit_code == 0
    assert "Dana" in result.stdout or "ent_" in result.stdout


# ---------- remember / recall ----------


def test_remember_attaches_fact(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    _seed_entity(install_ctx, name="Charlie")
    result = runner.invoke(app, ["remember", "Charlie", "--fact", "likes pears"])

    assert result.exit_code == 0


def test_recall_returns_entity_and_facts(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    entity_id, _ = _seed_entity(install_ctx, name="Charlie")
    ops.remember(
        install_ctx.store,
        entity_id=entity_id,
        content="likes pears",
        source=Source.USER,
        clock=install_ctx.clock,
        actor=install_ctx.actor,
    )

    result = runner.invoke(app, ["recall", "Charlie"])
    assert result.exit_code == 0
    assert "pears" in result.stdout


def test_recall_json_format(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    entity_id, _ = _seed_entity(install_ctx, name="Charlie")
    ops.remember(
        install_ctx.store,
        entity_id=entity_id,
        content="likes pears",
        source=Source.USER,
        clock=install_ctx.clock,
        actor=install_ctx.actor,
    )

    result = runner.invoke(app, ["recall", "Charlie", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["entity"]["name"] == "Charlie"
    assert any(f["content"] == "likes pears" for f in payload["facts"])


def test_recall_unknown_entity_returns_user_error(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    result = runner.invoke(app, ["recall", "NobodyKnows"])
    assert result.exit_code == 1


# ---------- merge ----------


def test_merge_folds_losers_into_winner(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    entity_a, _ = _seed_entity(install_ctx, name="Charlie")
    # Seed a second entity with a distinct image so the FakeStore records a
    # different observation id.
    from vemem.cli import commands_util

    obs_ids = commands_util.ingest_image(
        install_ctx,
        image_bytes=IMAGE_BYTES + b"-other",
        source_uri="file:///other.png",
    )
    entity_b = ops.label(
        install_ctx.store, obs_ids, "CharlesBis", clock=install_ctx.clock, actor=install_ctx.actor
    )

    result = runner.invoke(app, ["merge", entity_a, entity_b.id, "--yes"])
    assert result.exit_code == 0


def test_merge_aborts_on_prompt_reject(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    entity_a, _ = _seed_entity(install_ctx, name="Charlie")
    from vemem.cli import commands_util

    obs_ids = commands_util.ingest_image(
        install_ctx,
        image_bytes=IMAGE_BYTES + b"-other",
        source_uri="file:///other.png",
    )
    entity_b = ops.label(
        install_ctx.store, obs_ids, "Bis", clock=install_ctx.clock, actor=install_ctx.actor
    )

    result = runner.invoke(app, ["merge", entity_a, entity_b.id], input="n\n")
    assert result.exit_code == 1
    # Ensure neither entity got merged
    assert install_ctx.store.get_entity(entity_a).status.value == "active"
    assert install_ctx.store.get_entity(entity_b.id).status.value == "active"


# ---------- forget ----------


def test_forget_with_yes_flag_hard_deletes(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    entity_id, _ = _seed_entity(install_ctx, name="Charlie")
    result = runner.invoke(app, ["forget", "Charlie", "--yes"])

    assert result.exit_code == 0
    entity = install_ctx.store.get_entity(entity_id)
    assert entity is not None
    assert entity.status.value == "forgotten"


def test_forget_without_yes_prompts_and_aborts_on_n(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    entity_id, _ = _seed_entity(install_ctx, name="Charlie")
    result = runner.invoke(app, ["forget", "Charlie"], input="n\n")

    # Aborted by the user → exit code 1 (user error / explicit abort).
    assert result.exit_code == 1
    entity = install_ctx.store.get_entity(entity_id)
    assert entity is not None
    assert entity.status.value == "active"


def test_forget_without_yes_proceeds_on_y(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    entity_id, _ = _seed_entity(install_ctx, name="Charlie")
    result = runner.invoke(app, ["forget", "Charlie"], input="y\n")

    assert result.exit_code == 0
    assert install_ctx.store.get_entity(entity_id).status.value == "forgotten"


# ---------- restrict / unrestrict ----------


def test_restrict_flips_status(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    entity_id, _ = _seed_entity(install_ctx, name="Charlie")
    result = runner.invoke(app, ["restrict", "Charlie"])
    assert result.exit_code == 0
    assert install_ctx.store.get_entity(entity_id).status.value == "restricted"

    result = runner.invoke(app, ["unrestrict", entity_id])
    assert result.exit_code == 0
    assert install_ctx.store.get_entity(entity_id).status.value == "active"


# ---------- undo ----------


def test_undo_last_reversible_event(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    _seed_entity(install_ctx, name="Charlie")
    # Attach a fact; undo-by-actor should undo this most-recent reversible event.
    result_remember = runner.invoke(app, ["remember", "Charlie", "--fact", "likes pears"])
    assert result_remember.exit_code == 0

    result = runner.invoke(app, ["undo"])
    assert result.exit_code == 0


# ---------- list ----------


def test_list_shows_entities(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    _seed_entity(install_ctx, name="Charlie")
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "Charlie" in result.stdout


def test_list_json(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    _seed_entity(install_ctx, name="Charlie")
    result = runner.invoke(app, ["list", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert any(e["name"] == "Charlie" for e in payload["entities"])


# ---------- inspect ----------


def test_inspect_shows_entity_details(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    entity_id, _ = _seed_entity(install_ctx, name="Charlie")
    ops.remember(
        install_ctx.store,
        entity_id=entity_id,
        content="likes pears",
        source=Source.USER,
        clock=install_ctx.clock,
        actor=install_ctx.actor,
    )

    result = runner.invoke(app, ["inspect", "Charlie"])
    assert result.exit_code == 0
    assert "Charlie" in result.stdout
    assert "pears" in result.stdout


# ---------- export ----------


def test_export_writes_json_to_stdout(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    _seed_entity(install_ctx, name="Charlie")
    result = runner.invoke(app, ["export", "Charlie"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["entity"]["name"] == "Charlie"


def test_export_to_output_file(
    runner: CliRunner,
    install_ctx: CliContext,
    tmp_path: Path,
) -> None:
    _seed_entity(install_ctx, name="Charlie")
    out = tmp_path / "charlie.json"
    result = runner.invoke(app, ["export", "Charlie", "--output", str(out)])

    assert result.exit_code == 0
    payload = json.loads(out.read_text())
    assert payload["entity"]["name"] == "Charlie"


# ---------- export-tools ----------


def test_export_tools_to_stdout(
    runner: CliRunner,
    install_ctx: CliContext,
) -> None:
    result = runner.invoke(app, ["export-tools"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert len(payload) > 0


def test_export_tools_to_file(
    runner: CliRunner,
    install_ctx: CliContext,
    tmp_path: Path,
) -> None:
    out = tmp_path / "tools.json"
    result = runner.invoke(app, ["export-tools", "--output", str(out)])
    assert result.exit_code == 0
    payload = json.loads(out.read_text())
    assert isinstance(payload, list)


# ---------- migrate / repair stubs ----------


def test_migrate_is_stub(runner: CliRunner, install_ctx: CliContext) -> None:
    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 0
    assert "not implemented" in result.stdout.lower()


def test_repair_is_stub(runner: CliRunner, install_ctx: CliContext) -> None:
    result = runner.invoke(app, ["repair"])
    assert result.exit_code == 0
    assert "not implemented" in result.stdout.lower()


# ---------- module entry ----------


def test_module_entry_help() -> None:
    """`python -m vemem.cli --help` mirrors `vm --help`."""
    # Instead of spawning a subprocess (slow), verify the __main__ module
    # imports the same app object.
    from vemem.cli import __main__ as cli_main
    from vemem.cli.app import app as registered_app

    assert cli_main.app is registered_app


# ---------- serve-mcp ----------


def test_serve_mcp_help_hint(runner: CliRunner, install_ctx: CliContext) -> None:
    """--help should reference `python -m vemem.mcp_server` so users know the fallback."""
    result = runner.invoke(app, ["serve-mcp", "--help"])
    assert result.exit_code == 0
    assert "mcp_server" in result.stdout or "mcp-server" in result.stdout


# ---------- base64-encoded helper check ----------


def test_observe_b64_exists() -> None:
    """Sanity check: the test image base64 encodes without error."""
    assert base64.b64encode(IMAGE_BYTES).decode("ascii")
