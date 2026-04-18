# Contributing to vemem

Thanks for looking. vemem is a personal OSS project that may eventually accept contributors; for now, the main use for this file is as a reference for AI agents working on the codebase.

## Dev setup

```bash
git clone https://github.com/linville-charlie/vemem
cd vemem
uv sync      # installs dev deps, creates .venv
uv run pytest                      # runs the full suite
uv run vm --help                   # tries the CLI
uv run python -m vemem.mcp_server  # spawns the MCP server over stdio
```

Python 3.13 is the local dev default (see `.python-version`) because LanceDB 0.19 segfaults on 3.14. CI runs 3.12 and 3.13; both must pass.

## Verify gate

Every commit should pass the local equivalent of CI:

```bash
uv run ruff check .           # lint
uv run ruff format --check .  # format
uv run mypy src tests         # types (strict on src/, lenient on tests/)
uv run pytest --tb=short -q   # tests
```

The existing `CLAUDE.md` loops (`verify.sh`, auto-checkpoint, regression) are the suggested automation if you use Claude Code. Without them, run the four commands above before pushing.

## TDD gate

Per CLAUDE.md Loop 2: **tests first, confirm red, then implement**. This is enforced by convention, not tooling — but every wave of v0 was built this way and the design-pressure was real. Do not skip it for "obvious" changes; obvious changes break production tests.

## Where things live

| | |
|---|---|
| Spec | `docs/spec/identity-semantics.md` — the load-bearing semantic contract |
| Implementation plan | `docs/plan/v0-implementation.md` — wave-by-wave buildout |
| Contributor notes | `CLAUDE.md` — tooling, layout, agent workflows |
| Examples | `docs/examples/{bridge.py, mcp_usage.md, openai_tools.md, cli_tour.md, claude_desktop_config.json}` |
| Compliance checklist | `COMPLIANCE.md` |
| Security policy | `SECURITY.md` |

Source layout:

```
src/vemem/
├── core/          domain types, Protocols, errors, the thirteen ops
├── storage/       LanceDBStore + PyArrow schemas
├── encoders/      InsightFace + CLIP + detectors + crop helper
├── mcp_server/    FastMCP stdio server with 14 tools
├── tools/         OpenAI function-calling JSON schemas
├── cli/           Typer-based `vm` command
├── pipeline.py    shared observe_image recipe
├── facade.py      Vemem class
└── __init__.py    public re-exports
```

Tests mirror the source tree under `tests/`.

## Conventions

- `@dataclass(frozen=True, slots=True)` for domain types.
- IDs are UUID v7 (see `vemem.core.ids.new_id`).
- Errors inherit from `vemem.core.errors.VemError` and end in `Error` (PEP 8 N818).
- Store-level operations go in the `Store` Protocol (update both `FakeStore` and `LanceDBStore`).
- Top-level re-exports in `vemem/__init__.py` — keep `__all__` sorted.
- Commit messages use lowercase prefixes: `feat(core):`, `feat(storage):`, `refactor(ops):`, `docs:`, `chore:`. Include `Co-Authored-By:` for agent-written work.
- Line length 100. Ruff format is authoritative.

## Editing the agent skill

The canonical skill lives at [`skills/vemem/`](./skills/vemem/). The openclaw plugin ships a **mirror** at `integrations/openclaw/plugin/skills/vemem/` so that `cp -r plugin ~/.openclaw/extensions/…` brings the skill with it (a symlink would point outside the copied tree and break).

**Workflow when you edit the skill:**

1. Edit `skills/vemem/SKILL.md` or any file under `skills/vemem/references/`.
2. Run the sync script:
   ```bash
   scripts/sync-bundled-skill.sh
   ```
3. Commit both trees in the same commit.

Verify the mirror matches before pushing:

```bash
diff -rq skills/vemem integrations/openclaw/plugin/skills/vemem
# should print nothing
```

A CI check enforcing bit-for-bit equality between the two trees is on the backlog. Until then, the sync script is your responsibility.

## Running the integration suite

```bash
VEMEM_RUN_INTEGRATION=1 uv run pytest
```

Downloads InsightFace (`buffalo_l`, ~200MB) and open_clip (`ViT-B-32`, ~350MB) on first run. Cached under `~/.insightface/` and `~/.cache/`. Expect ~1 minute on a clean machine.

## Filing issues

Bug reports: include `uv run python -c "import vemem; print(vemem.__version__)"` and a minimal reproduction.

Design proposals: open an issue titled `proposal: <summary>` with a sketch of the API and how it interacts with the existing spec sections.

Feature requests that would require spec changes: read `docs/spec/identity-semantics.md` first; explain which section your proposal amends.

## Release cadence

No schedule. v0 is published when Wave 4 is cut; v0.1 addresses the backlog of Protocol gaps flagged in this codebase (search `# TODO` or grep DONE.md archive).
