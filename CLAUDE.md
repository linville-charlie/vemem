# Claude Code — Autonomous Operation Manual

You operate with maximum self-agency. Fix problems independently. Only escalate
to the human for **design decisions** — architecture, product direction, tradeoffs
between approaches. Everything else you resolve yourself.

When a task has independent parts, **decompose and parallelize** rather than
working sequentially. Time is the resource to optimize.

---

## Loop 1 — Pre-Task Planning (human gate)

Before writing a single line of code, produce a plan. Do not skip this.

```
PLAN
════
Task:           <one sentence>
Decomposable:   YES / NO
  If YES →      <list the independent subtasks and which can run in parallel>
Files affected: <list>
Approach:       <how you'll implement it>
Test strategy:  <what tests you'll write first>
Design questions: <anything requiring human input — if none, write "none">
```

**Decomposition rule:** if the task touches two or more modules with no shared
write dependencies, it is decomposable. Spawn sub-agents (Loop 8) instead of
working sequentially.

- If `Design questions` is non-empty → stop and wait for human input
- If `Design questions` is "none" → proceed autonomously

---

## Loop 2 — TDD Cycle (every change, every agent)

All agents — orchestrator and sub-agents — follow this order:

1. **Read first** — scan relevant existing code and tests
2. **Write the test** — specific, minimal, named for the behavior
3. **Confirm red** — suite must fail before you implement
4. **Write implementation** — only enough to make the test pass
5. **Verify hook fires** — `verify.sh` runs lint + typecheck (parallel) then tests
6. **Refactor** — clean up, verify again

Verify runs lint and typecheck **in parallel** automatically. Tests run after
both pass. You do not need to manually trigger this.

---

## Loop 3 — Error Recovery (max 5 attempts, then escalate)

```
Attempt 1: Read full error output. Fix most likely root cause.
Attempt 2: Add debug output to narrow the failure location.
Attempt 3: Check git log — did a recent change break an assumption?
Attempt 4: Spawn a sub-agent with fresh context and a different hypothesis
           (see Loop 8). Race it against your own attempt 5.
Attempt 5: Minimal reproduction — isolate to smallest failing case.

After 5 attempts: STOP. Escalate to human (see template below).
```

On attempt 4, use this pattern:
```bash
bash ~/.claude/hooks/worktree-agent.sh fix/attempt-alt \
  "Alternative approach to: <describe the error and constraint>"
```
Let both approaches run. Take whichever goes green first, discard the other.

---

## Loop 4 — Git Checkpointing (automatic)

`verify.sh` → `git-checkpoint.sh` on every green pass. Automatic, no action needed.

- Checkpoint commits: `checkpoint(HH:MM:SS): N file(s) — filename`
- Never push checkpoint commits — squash before pushing
- To squash: `git rebase -i HEAD~<N>` → mark checkpoints as `squash`
- Sub-agents commit normally (not as checkpoints) — their commits merge cleanly

---

## Loop 5 — Regression Guard (runs on Stop)

`regression.sh` runs the pytest suite before you can finish.

If regressions are found:
1. `git log --oneline -10` — find the checkpoint that broke it
2. `git diff <hash>` — isolate the change
3. Fix it. Do not delete tests to make the suite pass.

---

## Loop 6 — Dependency Audit (runs after installs)

`dependency-audit.sh` fires automatically after any install.

- **HIGH/CRITICAL** → stop, find a safe version or flag to human
- **LOW/MODERATE** → note in summary, continue

---

## Loop 7 — Docs & Dead Code Sync (parallel sub-agent)

When implementation is green, **do not do docs/cleanup inline**.
Instead, spawn a dedicated sub-agent for it while you handle commit squashing:

```bash
bash ~/.claude/hooks/worktree-agent.sh docs/sync-$(date +%s) \
  "Sync docs and remove dead code for the changes made in: <brief description>.
   Update docstrings, README sections, remove unused imports and debug prints.
   Do not change logic — only documentation and cleanup."
```

You (orchestrator) squash checkpoint commits and write the final commit message
in parallel. Merge the docs branch back when both are done.

**Dead code checklist (for the docs sub-agent):**
- Remove unused imports (ruff flags these, also scan manually)
- Remove scaffolding, TODO stubs, debug prints from development
- Remove unreachable code after refactors
- Docstrings explain *why*, not *what* — remove redundant ones

---

## Loop 8 — Orchestrator: Parallel Task Decomposition

This is the highest-leverage loop. Use it for any task that is decomposable.

### When to parallelize

Parallelize when subtasks have **no shared write dependencies** — they touch
different files, modules, or layers. Examples:

| Task | Decomposition |
|------|--------------|
| Add user auth | Agent A: DB models + migrations · Agent B: API endpoints · Agent C: frontend forms |
| Refactor two modules | Agent A: module 1 · Agent B: module 2 |
| Multi-feature sprint | One agent per feature |
| Error recovery attempt 4+ | Agent A: current approach · Agent B: alternative hypothesis |
| Docs + implementation | Agent A: implementation · Agent B: docs sync (Loop 7) |

Do NOT parallelize when agents would write to the same files — that causes
merge conflicts. Decompose at module or layer boundaries.

### Orchestrator workflow

```
1. PLAN: identify independent subtasks (Loop 1 Decomposable field)

2. SPAWN: create a worktree per sub-agent
   bash ~/.claude/hooks/worktree-agent.sh <branch> "<task>"

3. DELEGATE: give each sub-agent a clear, self-contained task in its TASK.md
   - What to build
   - Which files to touch (and which to leave alone)
   - Which tests to write
   - What "done" looks like

4. MONITOR: check sub-agent progress
   git -C /tmp/claude-agents/<branch> log --oneline -5

5. MERGE: when a sub-agent writes DONE.md, merge it back
   bash ~/.claude/hooks/worktree-merge.sh <branch>
   
6. VERIFY: run full regression suite after all merges
   bash ~/.claude/hooks/regression.sh

7. CLEAN UP: squash all checkpoints, write one clean commit per feature
```

### Sub-agent responsibilities

Sub-agents are **not aware of each other**. The orchestrator owns:
- Deciding what's safe to parallelize
- Merging branches in the right order (least likely to conflict first)
- Resolving any merge conflicts
- Running the final regression pass

Sub-agents own:
- Their TDD loop (Loop 2)
- Their own verify/checkpoint cycle (Loops 3–4)
- Writing a clear DONE.md when finished

### Parallel spawn example

```bash
# Orchestrator spawns three agents simultaneously:
bash ~/.claude/hooks/worktree-agent.sh feat/auth-models \
  "Create User and Session SQLAlchemy models with migrations. Write pytest tests."

bash ~/.claude/hooks/worktree-agent.sh feat/auth-api \
  "Implement /auth/login and /auth/logout FastAPI endpoints using the User model.
   Assume models exist at app/models/user.py. Write integration tests."

bash ~/.claude/hooks/worktree-agent.sh feat/auth-frontend \
  "Build LoginForm and LogoutButton React components. Write jest tests.
   API base URL is in src/config.ts."

# Orchestrator works on its own subtask (e.g., wiring middleware) while agents run.

# Merge in dependency order (models first, then API, then frontend):
bash ~/.claude/hooks/worktree-merge.sh feat/auth-models
bash ~/.claude/hooks/worktree-merge.sh feat/auth-api
bash ~/.claude/hooks/worktree-merge.sh feat/auth-frontend

# Final regression pass across everything
bash ~/.claude/hooks/regression.sh
```

---

## Verification Commands (reference)

```bash
# Python (uv-managed — everything runs inside the project venv automatically)
uv sync                              # install / update from pyproject.toml + uv.lock
uv run ruff check .                  # lint
uv run ruff format .                 # format
uv run mypy src tests                # type check
uv run pytest --tb=short -q          # quick run
uv run pytest --tb=short -v          # verbose run
uv run python -m vemem.mcp_server    # launch MCP server (stdio)

# Git
git log --oneline -10                # recent checkpoints
git diff HEAD~1                      # last checkpoint diff
git reset HEAD~1                     # undo last checkpoint
git worktree list                    # active sub-agent worktrees

# Agent management
bash ~/.claude/hooks/worktree-agent.sh <branch> "<task>"   # spawn
bash ~/.claude/hooks/worktree-merge.sh <branch>            # merge
git -C /tmp/claude-agents/<branch> log --oneline -5        # check progress
```

---

## Hard Rules

| Rule | Detail |
|------|--------|
| Parallelize decomposable tasks | Sequential work on independent modules is wasted time |
| Never share write targets between parallel agents | Decompose at module/layer boundaries |
| Merge in dependency order | core (schema + ops) → storage + encoders → adapters (MCP / tool schemas / CLI) |
| Run regression after all merges | A clean sub-agent doesn't mean a clean merge |
| Never declare done with failing checks | All lint, types, and tests must be green |
| Never suppress errors silently | No `# noqa`, `// eslint-disable`, `@ts-ignore` without approval |
| Never delete tests to pass | Fix the code |
| Never push checkpoint commits | Squash first |
| Never spiral past 5 attempts | Escalate with a clean report |
| Never make design decisions alone | Raise in the Plan stage |

---

## Escalation Template

```
ESCALATION
══════════
Blocker:     <one sentence>
Attempted:   <numbered list of all attempts including any sub-agent tries>
Error:       <exact error output>
Hypothesis:  <best guess at root cause>
Options:     A) <approach> — tradeoff: <...>
             B) <approach> — tradeoff: <...>
Need:        <specific decision or input required>
```

---

## Project-Specific Notes

Project:       Visual Entity Memory — OSS Python library providing persistent visual identity
               as a first-class primitive. Bridges vision models and text LLMs by resolving
               image observations to named entities and accumulating knowledge (facts, events,
               relationships) per entity. Self-hosted, open source. Face-first in v0 with an
               encoder-agnostic architecture so CLIP / DINOv3 / SigLIP can slot in later.

Stack:         Python 3.12+ · LanceDB (default storage) · InsightFace (v0 face encoder)
               MCP server (Model Context Protocol) · OpenAI-compatible tool schemas

Package mgr:   uv — lockfile `uv.lock`, deps in `pyproject.toml`.
               Install: `uv sync`. Run anything via `uv run <cmd>` (auto-venv).

Tests:         pytest. Unit tests per module; integration tests hit real LanceDB on disk
               in tempdirs. Do NOT mock the store — the store IS the integration surface
               (mock/prod divergence is exactly the class of bug we're trying to prevent).

Lint:          ruff (see `pyproject.toml [tool.ruff]`). Format via `ruff format`.
Types:         mypy strict on `src/`, lenient on `tests/`.

Layout (target):
  src/vemem/
    ├── core/        — Entity schema, five ops (observe/identify/label/remember/recall),
    │                  correction semantics (merge/split/relabel/forget/undo)
    ├── storage/     — Storage Protocol + LanceDB backend
    ├── encoders/    — Encoder Protocol + InsightFace impl (+ CLIP stub in v0)
    ├── mcp_server/  — MCP server exposing the five ops over stdio (HTTP later)
    ├── tools/       — OpenAI-compatible tool schemas (JSON)
    └── cli/         — `vm label`, `vm inspect`, `vm list` (merge/export in v0.1)

Parallel safe: core/ must land first (defines Protocols + schema). After that,
               storage/ · encoders/ · mcp_server/ · tools/ · cli/ are independent layers
               and compose well via the Loop 8 worktree pattern.

Deploy:        N/A — library is installed via pip / `uv pip install vemem`.
               MCP server runs locally on the user's machine (stdio default, optional HTTP).
               GitHub: https://github.com/linville-charlie/vemem

Monitoring:    N/A for v0 (personal + OSS project).

Reference docs (read before planning non-trivial changes):
- Product vision:         /Users/clinville/Downloads/visual-memory-vision.md
- V0 critique + scope:    /Users/clinville/.claude/plans/bubbly-tinkering-teacup.md
- T1 identity semantics:  docs/spec/identity-semantics.md  ← load-bearing; read before touching core/, storage/, or encoders/

Open design questions (resolve before the code that depends on them):
- T2 cross-encoder gallery compatibility (live-swap vs rebuild-on-swap — 1-day spike needed)
- Auto-clustering policy (suggest-on-confidence vs commit-on-label) — deferred to v0.1, constraints in §4.7 of identity-semantics.md
- Voting-vs-threshold semantics for auto-suggest (§8 of identity-semantics.md)

---

## Custom Agents — When to Auto-Trigger

These agents live in `.claude/commands/`. Invoke manually with `/agent-name`
or follow the auto-trigger rules below. The orchestrator is responsible for
knowing when to spawn them — don't wait to be asked.

| Agent | Auto-trigger when | Manual use |
|---|---|---|
| `/test-writer` | Before any implementation (TDD gate, Loop 2 step 2) | `/test-writer <file or feature>` |
| `/debugger` | Error recovery reaches attempt 3 (Loop 3) | `/debugger <error>` |
| `/refactor` | After implementation is green, before docs (post Loop 2) | `/refactor <target>` |
| `/docs` | In parallel with commit squashing (Loop 7) | `/docs <target>` |
| `/security-review` | Before any push or PR creation | `/security-review` |
| `/pr-writer` | After security-review returns CLEAN or MERGE WITH FOLLOW-UP | `/pr-writer` |

### Full autonomous task flow with agents

```
You receive a task
      │
      ▼
[Loop 1] Plan → decomposable? → spawn worktree sub-agents (Loop 8)
      │
      ▼
[/test-writer] writes failing tests first          ← TDD gate
      │
      ▼
[Loop 2] Orchestrator implements until green
      │
      │ stuck at attempt 3?
      ├──────────────────────► [/debugger] diagnoses root cause
      │                              │
      │◄─────────────────────────────┘ returns fix or escalates
      ▼
[/refactor] cleans up implementation (optional worktree)
      │
      ├── parallel ──► [/docs] syncs docstrings + README (Loop 7 worktree)
      │
      ▼
[Loop 5] regression.sh — full suite green
      │
      ▼
[/security-review] — must return CLEAN or MERGE WITH FOLLOW-UP
      │ BLOCK MERGE? ──► fix findings, re-run security-review
      ▼
[/pr-writer] squashes commits, writes PR description
      │
      ▼
Human approves push (git push is in ask list)
```

### Agent coordination rules

- **test-writer and orchestrator never touch the same files at the same time**
  test-writer owns test files; orchestrator owns implementation files during TDD phase
- **docs runs in a worktree** — never inline, always parallel to commit squashing
- **debugger takes over error recovery** — orchestrator hands off fully, waits for result
- **security-review is a hard gate** — no push until it returns a verdict
- **pr-writer runs last** — it needs the final committed state to write accurate copy
