"""The ``vm`` command-line interface.

Each subcommand wraps one or more core ops (:mod:`vemem.core.ops`) with
Typer + Rich for humans and a predictable ``--format json`` for scripts.
Heavy components (the LanceDB store, InsightFace encoder) are loaded lazily
via :func:`vemem.cli.context.build_cli_context` so a ``vm --help`` invocation
doesn't pay the model-load cost.

Exit codes:

- 0 — success
- 1 — user error (bad args, entity not found, prompt declined)
- 2 — internal error (encoder missing for an image command, unexpected failure)

Actor defaults to ``cli:{username}``; ``--actor`` overrides it per invocation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from vemem.cli import commands_util
from vemem.cli.context import CliContext, build_cli_context
from vemem.cli.output import (
    candidate_json,
    console,
    dump_json,
    inspect_json,
    list_json,
    print_candidates_table,
    print_entities_table,
    print_entity_detail,
    print_facts_table,
    recall_snapshot_json,
)
from vemem.core import ops
from vemem.core.enums import Kind, Source, Status
from vemem.core.errors import (
    EntityUnavailableError,
    VemError,
)

app = typer.Typer(
    name="vm",
    help=(
        "vemem — manual identity work and maintenance.\n\n"
        "Observe images, label entities, attach facts, and manage corrections. "
        "Uses the LanceDB store at $VEMEM_HOME (default ~/.vemem) unless "
        "overridden with --home."
    ),
    add_completion=False,
    no_args_is_help=True,
)


# ---------- shared options ----------

HomeOpt = Annotated[
    Path | None,
    typer.Option(
        "--home",
        help="Store directory (overrides $VEMEM_HOME). Default: ~/.vemem",
        show_default=False,
    ),
]

ActorOpt = Annotated[
    str | None,
    typer.Option(
        "--actor",
        help="Override the recorded actor (default: cli:{username}).",
        show_default=False,
    ),
]

FormatOpt = Annotated[
    str,
    typer.Option(
        "--format",
        help="Output format: 'table' (human) or 'json' (scripting).",
        case_sensitive=False,
    ),
]


def _load(home: Path | None, actor: str | None) -> CliContext:
    """Build the CLI context, turning any init failure into a clean exit."""
    try:
        return build_cli_context(home=home, actor=actor)
    except Exception as exc:  # pragma: no cover - defensive; LanceDB open is tested elsewhere
        console().print(f"[red]error:[/red] failed to open store: {exc}")
        raise typer.Exit(code=2) from exc


def _require_image_pipeline(ctx: CliContext) -> None:
    """Exit with a helpful message if the encoder/detector isn't loaded."""
    if ctx.encoder is None or ctx.detector is None:
        msg = ctx.encoder_error or "encoder/detector not initialized"
        console().print(f"[red]error:[/red] {msg}")
        raise typer.Exit(code=2)


def _resolve_entity_or_exit(ctx: CliContext, name_or_id: str) -> Any:
    """Look up an entity by id → name, or exit 1 with a clean message."""
    entity = ctx.store.get_entity(name_or_id) or ctx.store.find_entity_by_name(name_or_id)
    if entity is None:
        console().print(f"[red]error:[/red] no entity matches {name_or_id!r}")
        raise typer.Exit(code=1)
    return entity


# ---------- core identity ----------


@app.command()
def observe(
    path_or_uri: Annotated[str, typer.Argument(help="Path to an image file.")],
    modality: Annotated[
        str,
        typer.Option(
            "--modality",
            help="Modality for new observations (face is the only supported modality in v0).",
        ),
    ] = "face",
    home: HomeOpt = None,
    actor: ActorOpt = None,
) -> None:
    """Detect + embed + store observations from an image file."""
    ctx = _load(home, actor)
    _require_image_pipeline(ctx)

    try:
        image_bytes = Path(path_or_uri).read_bytes()
    except FileNotFoundError:
        console().print(f"[red]error:[/red] file not found: {path_or_uri}")
        raise typer.Exit(code=1) from None
    except OSError as exc:
        console().print(f"[red]error:[/red] cannot read {path_or_uri}: {exc}")
        raise typer.Exit(code=1) from exc

    from vemem.core.enums import Modality

    try:
        modality_enum = Modality(modality)
    except ValueError:
        console().print(f"[red]error:[/red] unknown modality {modality!r}")
        raise typer.Exit(code=1) from None

    obs_ids = commands_util.ingest_image(
        ctx, image_bytes=image_bytes, source_uri=str(path_or_uri), modality=modality_enum
    )
    if not obs_ids:
        console().print("[yellow]no faces detected in image[/yellow]")
        return
    for oid in obs_ids:
        typer.echo(oid)


@app.command()
def identify(
    path_or_uri: Annotated[str, typer.Argument(help="Path to an image file.")],
    k: Annotated[int, typer.Option("--k", help="Max candidates per detection.")] = 5,
    min_confidence: Annotated[
        float,
        typer.Option("--min-confidence", help="Lower bound on candidate similarity."),
    ] = 0.5,
    prefer: Annotated[
        str,
        typer.Option("--prefer", help="Ranking bias: instance | type | both."),
    ] = "instance",
    output_format: FormatOpt = "table",
    home: HomeOpt = None,
    actor: ActorOpt = None,
) -> None:
    """Return ranked candidate entities per detected face."""
    ctx = _load(home, actor)
    _require_image_pipeline(ctx)

    try:
        image_bytes = Path(path_or_uri).read_bytes()
    except FileNotFoundError:
        console().print(f"[red]error:[/red] file not found: {path_or_uri}")
        raise typer.Exit(code=1) from None

    assert ctx.encoder is not None  # guaranteed by _require_image_pipeline
    assert ctx.detector is not None
    bboxes = ctx.detector.detect(image_bytes)
    detections: list[dict[str, Any]] = []
    for bbox in bboxes:
        vector = ctx.encoder.embed(image_bytes)
        candidates = ops.identify(
            ctx.store,
            encoder_id=ctx.encoder.id,
            vector=vector,
            k=k,
            min_confidence=min_confidence,
            prefer=prefer,
        )
        detections.append(
            {
                "bbox": list(bbox),
                "candidates": [candidate_json(c) for c in candidates],
            }
        )

        if output_format.lower() == "table":
            console().print(f"[dim]bbox[/dim] {bbox}")
            print_candidates_table(candidates)

    if output_format.lower() == "json":
        typer.echo(dump_json({"detections": detections, "encoder_id": ctx.encoder.id}))


@app.command()
def label(
    observation_ids: Annotated[list[str], typer.Argument(help="Observation ids to bind.")],
    name: Annotated[str, typer.Option("--name", help="Target entity name or id.")],
    home: HomeOpt = None,
    actor: ActorOpt = None,
) -> None:
    """Bind observations to an entity by name (creates it if new)."""
    ctx = _load(home, actor)
    try:
        entity = ops.label(ctx.store, list(observation_ids), name, clock=ctx.clock, actor=ctx.actor)
    except VemError as exc:
        console().print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(entity.id)
    console().print(
        f"[green]labeled[/green] {len(observation_ids)} obs → [bold]{entity.name}[/bold]"
    )


@app.command()
def relabel(
    observation_id: Annotated[str, typer.Argument(help="Observation id.")],
    name: Annotated[str, typer.Option("--name", help="New entity name or id.")],
    home: HomeOpt = None,
    actor: ActorOpt = None,
) -> None:
    """Move a single observation to a different entity (sugar for label)."""
    ctx = _load(home, actor)
    try:
        entity = ops.relabel(ctx.store, observation_id, name, clock=ctx.clock, actor=ctx.actor)
    except VemError as exc:
        console().print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(entity.id)


# ---------- knowledge ----------


@app.command()
def remember(
    entity: Annotated[str, typer.Argument(help="Entity name or id.")],
    fact: Annotated[str, typer.Option("--fact", help="Free-text fact to attach.")],
    source: Annotated[str, typer.Option("--source", help="user | vlm | llm | import")] = "user",
    home: HomeOpt = None,
    actor: ActorOpt = None,
) -> None:
    """Attach a fact to an entity."""
    ctx = _load(home, actor)
    target = _resolve_entity_or_exit(ctx, entity)

    try:
        source_enum = Source(source)
    except ValueError:
        console().print(f"[red]error:[/red] unknown source {source!r}")
        raise typer.Exit(code=1) from None

    try:
        new_fact = ops.remember(
            ctx.store,
            entity_id=target.id,
            content=fact,
            source=source_enum,
            clock=ctx.clock,
            actor=ctx.actor,
        )
    except VemError as exc:
        console().print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console().print(
        f"[green]remembered[/green] fact {new_fact.id} on [bold]{target.name or target.id}[/bold]"
    )


@app.command()
def recall(
    entity: Annotated[str, typer.Argument(help="Entity name or id.")],
    output_format: FormatOpt = "table",
    home: HomeOpt = None,
    actor: ActorOpt = None,
) -> None:
    """Show the entity plus its currently-active facts."""
    ctx = _load(home, actor)
    target = _resolve_entity_or_exit(ctx, entity)

    snapshot = ops.recall(ctx.store, entity_id=target.id)

    if output_format.lower() == "json":
        typer.echo(dump_json(recall_snapshot_json(snapshot)))
        return

    console().print(f"[bold]{snapshot.entity.name or snapshot.entity.id}[/bold]")
    if snapshot.facts:
        print_facts_table(list(snapshot.facts))
    else:
        console().print("[dim](no active facts)[/dim]")


# ---------- state changes ----------


@app.command()
def merge(
    entity_ids: Annotated[list[str], typer.Argument(help="Entity ids to merge.")],
    keep: Annotated[
        str, typer.Option("--keep", help="'oldest' or an explicit entity id.")
    ] = "oldest",
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
    home: HomeOpt = None,
    actor: ActorOpt = None,
) -> None:
    """Fold multiple entities into one (with confirmation)."""
    ctx = _load(home, actor)
    if len(entity_ids) < 2:
        console().print("[red]error:[/red] merge requires at least two entity ids")
        raise typer.Exit(code=1)

    if not yes:
        prompt = (
            f"About to merge {len(entity_ids)} entities (keep={keep}). "
            "This is reversible via `vm undo` until the default 30-day window elapses."
        )
        console().print(prompt)
        if not typer.confirm("Continue?", default=False):
            raise typer.Exit(code=1)

    try:
        winner = ops.merge(ctx.store, list(entity_ids), keep=keep, clock=ctx.clock, actor=ctx.actor)
    except VemError as exc:
        console().print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console().print(f"[green]merged[/green] → [bold]{winner.name or winner.id}[/bold]")


@app.command()
def forget(
    entity: Annotated[str, typer.Argument(help="Entity name or id.")],
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the confirmation prompt. Forget is NOT reversible by undo.",
        ),
    ] = False,
    home: HomeOpt = None,
    actor: ActorOpt = None,
) -> None:
    """Hard-delete everything tied to an entity (not reversible)."""
    ctx = _load(home, actor)
    target = _resolve_entity_or_exit(ctx, entity)

    if not yes:
        console().print(
            f"[red]DANGER:[/red] forget deletes all observations, bindings, facts, "
            f"events, and relationships for [bold]{target.name or target.id}[/bold]. "
            "This is NOT reversible by `vm undo`."
        )
        if not typer.confirm("Are you sure?", default=False):
            raise typer.Exit(code=1)

    try:
        counts = ops.forget(ctx.store, entity_id=target.id, clock=ctx.clock, actor=ctx.actor)
    except VemError as exc:
        console().print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console().print(f"[green]forgotten[/green] {target.id} — counts: {counts}")


@app.command()
def restrict(
    entity: Annotated[str, typer.Argument(help="Entity name or id.")],
    home: HomeOpt = None,
    actor: ActorOpt = None,
) -> None:
    """Stop using the entity for inference without deleting (GDPR Art. 18)."""
    ctx = _load(home, actor)
    target = _resolve_entity_or_exit(ctx, entity)
    try:
        updated = ops.restrict(ctx.store, entity_id=target.id, clock=ctx.clock, actor=ctx.actor)
    except EntityUnavailableError as exc:
        console().print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console().print(f"[yellow]restricted[/yellow] [bold]{updated.name or updated.id}[/bold]")


@app.command()
def unrestrict(
    entity: Annotated[str, typer.Argument(help="Entity name or id.")],
    home: HomeOpt = None,
    actor: ActorOpt = None,
) -> None:
    """Reverse of ``restrict`` — return the entity to ACTIVE."""
    ctx = _load(home, actor)
    target = _resolve_entity_or_exit(ctx, entity)
    try:
        updated = ops.unrestrict(ctx.store, entity_id=target.id, clock=ctx.clock, actor=ctx.actor)
    except EntityUnavailableError as exc:
        console().print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console().print(f"[green]active[/green] [bold]{updated.name or updated.id}[/bold]")


# ---------- audit ----------


@app.command()
def undo(
    event_id: Annotated[
        int | None,
        typer.Option("--event-id", help="Specific event id to undo. Defaults to the latest."),
    ] = None,
    home: HomeOpt = None,
    actor: ActorOpt = None,
) -> None:
    """Reverse the most recent reversible event by this actor (or a specific one)."""
    ctx = _load(home, actor)
    try:
        new_event = ops.undo(ctx.store, event_id=event_id, clock=ctx.clock, actor=ctx.actor)
    except VemError as exc:
        console().print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    payload_id = new_event.payload.get("undone_event_id")
    console().print(f"[green]undone[/green] event {payload_id} (new event {new_event.id})")


@app.command("list")
def list_entities(
    kind: Annotated[str, typer.Option("--kind", help="instance | type | all")] = "all",
    status: Annotated[
        str, typer.Option("--status", help="active | forgotten | restricted | all")
    ] = "active",
    output_format: FormatOpt = "table",
    home: HomeOpt = None,
    actor: ActorOpt = None,
) -> None:
    """Show all entities (filtered)."""
    ctx = _load(home, actor)

    entities = _iterate_entities(ctx)
    if kind != "all":
        try:
            kind_enum = Kind(kind)
        except ValueError:
            console().print(f"[red]error:[/red] unknown kind {kind!r}")
            raise typer.Exit(code=1) from None
        entities = [e for e in entities if e.kind is kind_enum]
    if status != "all":
        try:
            status_enum = Status(status)
        except ValueError:
            console().print(f"[red]error:[/red] unknown status {status!r}")
            raise typer.Exit(code=1) from None
        entities = [e for e in entities if e.status is status_enum]

    if output_format.lower() == "json":
        typer.echo(dump_json(list_json(entities)))
        return

    if not entities:
        console().print("[dim](no entities match)[/dim]")
        return
    print_entities_table(entities)


def _iterate_entities(ctx: CliContext) -> list[Any]:
    """Enumerate entities.

    Store implementations don't yet expose a public ``list_entities`` method
    on the Protocol (see DONE.md — Protocol gap). The FakeStore stores them
    in ``_entities`` and LanceDBStore exposes the raw table via ``_table``;
    both are treated as read-only here.
    """
    if hasattr(ctx.store, "list_entities"):
        return list(ctx.store.list_entities())

    # Fallback: FakeStore has ``_entities``; LanceDBStore has ``_table("entities")``.
    cache = getattr(ctx.store, "_entities", None)
    if isinstance(cache, dict):
        return list(cache.values())

    table_accessor = getattr(ctx.store, "_table", None)
    if callable(table_accessor):
        from vemem.storage.lancedb_store import _row_to_entity
        from vemem.storage.schemas import ENTITIES_TABLE

        table = table_accessor(ENTITIES_TABLE)
        rows = table.to_arrow().to_pylist()
        return [_row_to_entity(r) for r in rows]

    return []


@app.command()
def inspect(
    entity: Annotated[str, typer.Argument(help="Entity name or id.")],
    output_format: FormatOpt = "table",
    home: HomeOpt = None,
    actor: ActorOpt = None,
) -> None:
    """Show detailed info for one entity: facts, bindings, recent events."""
    ctx = _load(home, actor)
    target = _resolve_entity_or_exit(ctx, entity)

    facts = list(ctx.store.facts_for_entity(target.id, active_only=True))
    bindings = ctx.store.bindings_for_entity(target.id, include_negative=False)
    current_bindings = [b for b in bindings if b.valid_to is None]
    events = list(ctx.store.events_affecting_entity(target.id))[-10:]

    if output_format.lower() == "json":
        typer.echo(
            dump_json(
                inspect_json(
                    target,
                    facts,
                    binding_count=len(current_bindings),
                    recent_events=events,
                )
            )
        )
        return

    print_entity_detail(target, facts, binding_count=len(current_bindings), recent_events=events)


@app.command()
def export(
    entity: Annotated[str, typer.Argument(help="Entity name or id.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write JSON to PATH instead of stdout."),
    ] = None,
    include_embeddings: Annotated[
        bool,
        typer.Option("--include-embeddings", help="Include raw embedding vectors."),
    ] = False,
    home: HomeOpt = None,
    actor: ActorOpt = None,
) -> None:
    """GDPR Art. 20 data-portability dump for one entity."""
    ctx = _load(home, actor)
    target = _resolve_entity_or_exit(ctx, entity)

    dump = ops.export(ctx.store, entity_id=target.id, include_embeddings=include_embeddings)
    rendered = dump_json(dump)
    if output is None:
        typer.echo(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")


# ---------- surfaces ----------


@app.command("export-tools")
def export_tools_cmd(
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write JSON to PATH instead of stdout."),
    ] = None,
) -> None:
    """Emit the OpenAI function-calling JSON schemas for every op."""
    from vemem.tools import export as tools_export

    payload = tools_export.all_tools()
    rendered = dump_json(payload)
    if output is None:
        typer.echo(rendered)
    else:
        tools_export.write_tools_json(output)


@app.command("serve-mcp")
def serve_mcp_cmd() -> None:
    """Launch the MCP server (equivalent to ``python -m vemem.mcp_server``)."""
    from vemem.mcp_server.server import run as run_mcp

    run_mcp()


# ---------- deferred to v0.1 ----------


@app.command()
def migrate() -> None:
    """Run schema migrations (stub in v0)."""
    console().print("[yellow]migrate[/yellow] is not implemented in v0 — coming in v0.1")


@app.command()
def repair() -> None:
    """Recover from a crashed mid-op write (stub in v0)."""
    console().print("[yellow]repair[/yellow] is not implemented in v0 — coming in v0.1")


# ---------- entry point ----------


def main() -> int:
    """Console-script entry point (``vm``)."""
    try:
        app()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0
        return code
    return 0
