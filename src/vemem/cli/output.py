"""Rich-based output helpers for the ``vm`` CLI.

Human-readable commands use Rich tables and colored status cells; scriptable
``--format json`` output uses the same dict shapes as the MCP serialization
helpers so downstream tools can parse either surface consistently.

Kept tiny on purpose — these are the *only* module that knows about Rich, so
the command layer stays focused on op-invocation + error handling.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.table import Table

from vemem.core.enums import Status
from vemem.core.types import Candidate, Entity, EventLog, Fact
from vemem.mcp_server.serialization import (
    candidate_to_dict,
    entity_to_dict,
    event_log_to_dict,
    fact_to_dict,
    recall_snapshot_to_dict,
)

# A single Console instance; Typer CliRunner captures stdout via ``sys.stdout``
# so we force stdout rather than relying on Rich's auto-detection (which picks
# a no-op Console in non-interactive contexts, hiding output from tests).
_console = Console(force_terminal=False, soft_wrap=True)


def console() -> Console:
    """Return the module-level console. Exposed for tests + advanced use."""
    return _console


# ---------- print helpers ----------


def print_json(payload: Any) -> None:
    """Write a JSON payload to stdout with stable formatting."""
    _console.print_json(data=payload)


def print_entities_table(entities: list[Entity], *, title: str = "Entities") -> None:
    """Render a list of entities as a Rich table."""
    table = Table(title=title, show_lines=False)
    table.add_column("ID", overflow="fold", style="cyan", no_wrap=False)
    table.add_column("Name", style="bold")
    table.add_column("Kind")
    table.add_column("Modality")
    table.add_column("Status")
    table.add_column("Last Seen", overflow="fold")

    for e in entities:
        status_style = {
            Status.ACTIVE: "green",
            Status.RESTRICTED: "yellow",
            Status.MERGED_INTO: "magenta",
            Status.FORGOTTEN: "red",
        }.get(e.status, "white")
        table.add_row(
            e.id,
            e.name or "<anon>",
            e.kind.value,
            e.modality.value,
            f"[{status_style}]{e.status.value}[/{status_style}]",
            e.last_seen.isoformat(timespec="seconds"),
        )

    _console.print(table)


def print_candidates_table(candidates: list[Candidate], *, title: str = "Candidates") -> None:
    """Render ranked identify candidates as a Rich table."""
    table = Table(title=title, show_lines=False)
    table.add_column("Entity ID", overflow="fold", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Kind")
    table.add_column("Confidence", justify="right")
    table.add_column("Method")
    table.add_column("Matched Obs", overflow="fold")

    for c in candidates:
        table.add_row(
            c.entity.id,
            c.entity.name or "<anon>",
            c.entity.kind.value,
            f"{c.confidence:.3f}",
            c.method.value,
            ", ".join(c.matched_observation_ids),
        )

    _console.print(table)


def print_facts_table(facts: list[Fact], *, title: str = "Facts") -> None:
    """Render the active facts for an entity."""
    table = Table(title=title, show_lines=False)
    table.add_column("Fact ID", overflow="fold", style="cyan")
    table.add_column("Content", style="bold")
    table.add_column("Source")
    table.add_column("Valid From")
    table.add_column("Actor")

    for f in facts:
        table.add_row(
            f.id,
            f.content,
            f.source.value,
            f.valid_from.isoformat(timespec="seconds"),
            f.actor,
        )

    _console.print(table)


def print_events_table(events: list[EventLog], *, title: str = "Recent events") -> None:
    """Render a short event-log slice (for ``inspect``)."""
    table = Table(title=title, show_lines=False)
    table.add_column("Event ID", justify="right")
    table.add_column("Op", style="bold")
    table.add_column("Actor")
    table.add_column("At")
    table.add_column("Reversible")

    for ev in events:
        reversible = "yes" if ev.reversible_until else "no"
        table.add_row(
            str(ev.id),
            ev.op_type,
            ev.actor,
            ev.at.isoformat(timespec="seconds"),
            reversible,
        )

    _console.print(table)


def print_entity_detail(
    entity: Entity,
    facts: list[Fact],
    binding_count: int,
    recent_events: list[EventLog],
) -> None:
    """Render the detailed view used by ``vm inspect``."""
    _console.rule(f"[bold]{entity.name or entity.id}[/bold]")
    _console.print(
        f"[cyan]id[/cyan]         {entity.id}\n"
        f"[cyan]kind[/cyan]       {entity.kind.value}\n"
        f"[cyan]modality[/cyan]   {entity.modality.value}\n"
        f"[cyan]status[/cyan]     {entity.status.value}\n"
        f"[cyan]aliases[/cyan]    {', '.join(entity.aliases) or '-'}\n"
        f"[cyan]created[/cyan]    {entity.created_at.isoformat(timespec='seconds')}\n"
        f"[cyan]last seen[/cyan]  {entity.last_seen.isoformat(timespec='seconds')}\n"
        f"[cyan]bindings[/cyan]   {binding_count} current positive",
    )

    if facts:
        print_facts_table(facts)
    else:
        _console.print("[dim](no facts)[/dim]")

    if recent_events:
        print_events_table(recent_events)


# ---------- JSON shapes (shared with MCP serialization) ----------


def entity_json(entity: Entity) -> dict[str, Any]:
    return entity_to_dict(entity)


def fact_json(fact: Fact) -> dict[str, Any]:
    return fact_to_dict(fact)


def event_log_json(ev: EventLog) -> dict[str, Any]:
    return event_log_to_dict(ev)


def candidate_json(c: Candidate) -> dict[str, Any]:
    return candidate_to_dict(c)


def list_json(entities: list[Entity]) -> dict[str, Any]:
    return {"entities": [entity_to_dict(e) for e in entities]}


def inspect_json(
    entity: Entity,
    facts: list[Fact],
    binding_count: int,
    recent_events: list[EventLog],
) -> dict[str, Any]:
    return {
        "entity": entity_to_dict(entity),
        "facts": [fact_to_dict(f) for f in facts],
        "binding_count": binding_count,
        "recent_events": [event_log_to_dict(e) for e in recent_events],
    }


def recall_snapshot_json(snapshot: Any) -> dict[str, Any]:
    return recall_snapshot_to_dict(snapshot)


def dump_json(payload: Any) -> str:
    """Return a canonically-formatted JSON string (2-space indent)."""
    return json.dumps(payload, indent=2, default=str)
