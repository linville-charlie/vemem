# V0 Implementation Plan

Status: **draft** — 2026-04-16
Reference spec: [`docs/spec/identity-semantics.md`](../spec/identity-semantics.md)
Reference critique: `~/.claude/plans/bubbly-tinkering-teacup.md` (local, not in repo)

## Loop 1 Plan Block

```
PLAN
════
Task:           Implement vemem v0 — working self-hosted visual entity memory
                library with LanceDB storage, InsightFace encoder, MCP server,
                OpenAI tool schemas, CLI, and a reference bridge example.
Decomposable:   YES
Files affected: src/vemem/** · tests/** · docs/examples/**
Approach:       TDD per module. Contract tests shared between FakeStore and
                LanceDBStore. Integration tests wire the real stack end-to-end.
Test strategy:  Contract tests for Protocols (parametrized over impls).
                Semantic tests for ops (spec §4 scenarios).
                Real-image tests for encoders using public-domain fixtures.
                JSON-RPC snapshot tests for MCP server.
                End-to-end tests across the full stack.
Design questions: none — all load-bearing decisions live in the spec.
```

## Dependency tree

```
                         ┌──────────────────┐
                         │  Wave 1: core    │   (sequential, lands first)
                         │  types/protocols │
                         │  errors/enums/ids│
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼────────────────────┐
              │                   │                    │
     Wave 2A: core/ops    Wave 2B: storage     Wave 2C: encoders
     (FakeStore + tests)  (LanceDB backend)    (InsightFace + CLIP stub)
              │                   │                    │
              └───────────────────┼────────────────────┘
                                  │
           ┌──────────┬───────────┼───────────┬──────────┐
           │          │           │           │          │
      Wave 3D:   Wave 3E:    Wave 3F:    Wave 3G:
      MCP        OpenAI      CLI         bridge
      server     schemas                 example
           │          │           │           │          │
           └──────────┴───────────┼───────────┴──────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │ Wave 4: integration glue │
                     │ + e2e tests              │
                     │ + COMPLIANCE/SECURITY    │
                     └──────────────────────────┘
```

---

## Wave 1 — Core foundations

**Sequential. Orchestrator, ~1 day. Must land before any Wave 2 agent starts.**

Everyone imports from here; we cannot afford churn once Wave 2 is underway.

**Files:**
- `src/vemem/core/enums.py` — `Polarity`, `Modality`, `Kind`, `OpType`, `Status`, `Source`, `Method`
- `src/vemem/core/ids.py` — UUID v7 generator
- `src/vemem/core/types.py` — frozen dataclasses for `Observation`, `Embedding`, `Entity`, `Binding`, `Fact`, `Event`, `Relationship`, `EventLog`, `Candidate`
- `src/vemem/core/protocols.py` — `Store`, `Encoder`, `Detector`, `Clock`
- `src/vemem/core/errors.py` — `ModalityMismatch`, `KindMismatch`, `EntityUnavailable`, `OperationNotReversible`, `NoCompatibleEncoder`, `SchemaVersionError`
- `tests/core/test_types.py` — frozen/hashable invariants, default values, bi-temporal field rules
- `tests/core/test_protocols.py` — Protocol satisfaction smoke tests

**Design defaults** (baked in here, cite spec line when in doubt):
- Types are `@dataclass(frozen=True, slots=True)`. Serialization handled by separate adapters (pydantic at MCP/tool boundaries).
- IDs are UUID v7 (time-ordered — better for append-only).
- `Clock` is a Protocol injected into ops so tests can freeze time.

**Done when:** `uv run ruff check`, `uv run mypy src tests`, `uv run pytest` green; types importable; each type carries a docstring that references its spec section.

---

## Wave 2 — Operations, storage, encoders

**Parallel. Three sub-agents, ~2–3 days each. Spawn simultaneously once Wave 1 is merged.**

### 2A — Core ops (branch: `feat/core-ops`)

Owns the semantics of the five core ops plus corrections.

**Files:**
- `src/vemem/core/ops.py` — `observe`, `identify`, `label`, `relabel`, `merge`, `split`, `forget`, `restrict`, `unrestrict`, `remember`, `recall`, `undo`, `export`
- `src/vemem/core/eventlog.py` — helpers for writing EventLog rows with payloads
- `tests/support/fake_store.py` — in-memory `Store` impl (also used by Wave 3 tests; lives under `tests/` so it's not published)
- `tests/core/test_ops_semantics.py` — scenarios from spec §4:
  - label → identify roundtrip
  - relabel emits positive + negative binding
  - merge migrates bindings and facts with provenance
  - split emits cross-wise negatives; facts stay on original
  - forget purges observations (multi-bound obs survive)
  - undo of merge restores pre-merge state
  - undo of forget rejects with `OperationNotReversible`
  - ModalityMismatch rejects cross-modality ops

**Must NOT touch:** `src/vemem/storage/**`, `src/vemem/encoders/**`.

**Done when:** all spec §4 scenarios pass against `FakeStore`; ops code is pure logic (no I/O, no deps on concrete backend).

### 2B — LanceDB storage (branch: `feat/storage-lancedb`)

Owns the concrete `Store` Protocol implementation.

**Files:**
- `src/vemem/storage/lancedb_store.py` — `LanceDBStore` implementing `Store`
- `src/vemem/storage/schemas.py` — PyArrow schemas per table (observations, embeddings, entities, bindings, facts, events, relationships, event_log)
- `src/vemem/storage/migrations.py` — schema version tracking (v0 has one version; scaffold supports future migrations)
- `tests/storage/test_store_contract.py` — the shared contract test suite parametrized over `FakeStore` and `LanceDBStore`
- `tests/storage/test_lancedb_specific.py` — LanceDB-only: versioning + `optimize(Prune{older_than:0})` verifies forget actually removes vectors from version history; merge_insert for bindings; transactional ordering

**Deps added:** `lancedb>=0.19`, `pyarrow>=18`, `numpy>=2`.

**Must NOT touch:** `src/vemem/core/**` (only read from it), `src/vemem/encoders/**`.

**Done when:** contract tests pass for both stores; forget+prune verified removing embeddings from LanceDB version history; `repair()` CLI stub handles a simulated mid-op crash.

### 2C — Encoders (branch: `feat/encoders`)

Owns the concrete `Encoder` and `Detector` implementations.

**Files:**
- `src/vemem/encoders/insightface_encoder.py` — `InsightFaceEncoder` (buffalo_l model, 512-d ArcFace)
- `src/vemem/encoders/insightface_detector.py` — face detector producing bboxes
- `src/vemem/encoders/clip_encoder.py` — `CLIPEncoder` via `open_clip_torch` (experimental, 768-d)
- `tests/encoders/test_insightface.py` — fixture-based similarity sanity tests (same person high, different person low)
- `tests/encoders/test_clip.py` — smoke tests (produces vectors of expected dim, L2-normalized)
- `tests/fixtures/` — 4–6 public-domain face images (Wikimedia Commons with clear license)

**Deps added:** `insightface>=0.7`, `onnxruntime>=1.20` (CPU), `open_clip_torch>=2.30`, `pillow>=11`.

**Must NOT touch:** `src/vemem/core/**`, `src/vemem/storage/**`.

**Done when:** `Encoder` Protocol satisfied by both; InsightFace embeddings are L2-normalized; CLIP runs end-to-end; both fail gracefully with a clear error if model weights aren't installed.

---

## Wave 3 — Surfaces

**Parallel. Four sub-agents, ~1–2 days each. Spawn after Wave 2 merged.**

### 3D — MCP server (branch: `feat/mcp-server`)

**Files:**
- `src/vemem/mcp_server/server.py` — MCP server via the `mcp` SDK
- `src/vemem/mcp_server/tools.py` — tool handlers wrapping core ops
- `src/vemem/mcp_server/__main__.py` — `python -m vemem.mcp_server`
- `tests/mcp_server/test_stdio.py` — JSON-RPC roundtrips for every tool
- `docs/examples/claude_desktop_config.json` — ready-to-paste config

**Deps added:** `mcp>=0.9`.

**Done when:** `uv run python -m vemem.mcp_server` starts cleanly on stdio; snapshot tests cover each tool's input/output shape.

### 3E — OpenAI tool schemas (branch: `feat/openai-tools`)

**Files:**
- `src/vemem/tools/schemas.py` — generate OpenAI function-calling JSON schemas from op signatures
- `src/vemem/tools/export.py` — `vm export-tools > tools.json`
- `tests/tools/test_schemas.py` — snapshot tests against `tests/tools/snapshots/`

**Done when:** the nine ops serialize to valid OpenAI function schemas; snapshot file committed and protected.

### 3F — CLI (branch: `feat/cli`)

**Files:**
- `src/vemem/cli/app.py` — Typer app exposing `vm label`, `vm inspect`, `vm list`, `vm forget`, `vm undo`, `vm migrate`, `vm repair`, `vm export-tools`
- `src/vemem/cli/__main__.py` — `python -m vemem.cli`
- `pyproject.toml [project.scripts]` — `vm = "vemem.cli.app:main"`
- `tests/cli/test_commands.py` — Typer `CliRunner` tests per command

**Deps added:** `typer>=0.12`, `rich>=13` (for pretty inspect output).

**Done when:** `vm --help` prints; each command has a happy-path integration test against a real tempdir store.

### 3G — Reference bridge example (branch: `docs/bridge-example`)

**Files:**
- `docs/examples/bridge.py` — ~100 LoC reference integration showing the observe→identify→remember→recall pattern with callable placeholders for VLM and LLM
- `docs/examples/README.md` — how to wire real openclaw / Ollama / Claude MCP
- `tests/examples/test_bridge.py` — exercises the example with `FakeVLM`/`FakeLLM` stubs

**Must NOT touch:** `src/vemem/**` (docs + tests only).

**Done when:** the example runs standalone with fakes and prints expected output; docs/README shows a copy-paste openclaw snippet.

---

## Wave 4 — Integration + release hygiene

**Sequential. Orchestrator, ~1 day.**

**Files:**
- `src/vemem/__init__.py` — top-level re-exports (`Store`, `Entity`, `Observation`, `Candidate`, etc.)
- `src/vemem/facade.py` — convenience `vemem.Store()` picking defaults
- `src/vemem/config.py` — config loading from env + `~/.vemem/config.toml`
- `tests/integration/test_end_to_end.py` — full stack: observe → identify → label → remember → recall → merge → undo → forget
- `COMPLIANCE.md` — deployer responsibilities; mapping of spec §7 rules to operator action items
- `SECURITY.md` — responsible disclosure process
- `CONTRIBUTING.md` — dev setup, TDD loop, how to spawn sub-agents
- `README.md` quickstart — updated and verified against integration test

**Done when:** a fresh clone + `uv sync && uv run pytest` passes end-to-end; README quickstart is copy-paste correct; all three release-hygiene docs exist and are honest.

---

## Parallelism safety (Loop 8 rule)

Strict file ownership — no agent writes to another's tree:

| Wave | Agent | Writes | Reads |
|---|---|---|---|
| 1 | orchestrator | `core/` | — |
| 2A | `feat/core-ops` | `core/ops.py`, `core/eventlog.py`, `tests/support/fake_store.py`, `tests/core/test_ops_*` | `core/*` |
| 2B | `feat/storage-lancedb` | `storage/`, `tests/storage/` | `core/*` |
| 2C | `feat/encoders` | `encoders/`, `tests/encoders/`, `tests/fixtures/` | `core/*` |
| 3D | `feat/mcp-server` | `mcp_server/`, `tests/mcp_server/`, `docs/examples/claude_desktop_config.json` | `core/`, `storage/`, `encoders/` |
| 3E | `feat/openai-tools` | `tools/`, `tests/tools/` | `core/` |
| 3F | `feat/cli` | `cli/`, `tests/cli/`, `pyproject.toml [project.scripts]` | `core/`, `storage/`, `encoders/` |
| 3G | `docs/bridge-example` | `docs/examples/`, `tests/examples/` | `core/`, `storage/`, `encoders/` |
| 4 | orchestrator | `__init__.py`, `facade.py`, `config.py`, `tests/integration/`, `COMPLIANCE.md`, `SECURITY.md`, `CONTRIBUTING.md`, `README.md` | everything |

**Merge order:** Wave 1 → 2A → 2B → 2C → 3D → 3E → 3F → 3G → Wave 4. Within parallel waves, merge 2A first (ops drive the Store contract, so any protocol nit lands there). Run `regression.sh` after every merge.

**Shared file hazards:**
- `pyproject.toml` — 2B, 2C, 3D, 3E, 3F all add dependencies. Orchestrator reviews dep additions at merge time; conflicts here are trivial (3-way merge of independent lines in `dependencies = [...]`).
- `src/vemem/__init__.py` — only Wave 4 touches this. Sub-agents don't modify top-level re-exports.

---

## Design defaults (non-blocking; already decided)

| Decision | Value | Rationale |
|---|---|---|
| Types | frozen slots dataclasses | mutable domain objects + bi-temporal model is a foot-gun; Store owns mutation |
| IDs | UUID v7 | time-ordered, better for append-only workloads |
| Default storage path | `~/.vemem/` (env override: `VEMEM_HOME`) | personal-project ergonomics |
| Embedding normalization | L2 at encoder output | one place; matches cosine search |
| CLI framework | `typer` | modern, type-hint-native |
| MCP SDK | official `mcp` Python SDK | no other viable option |
| CLIP backend | `open_clip_torch` | widest CLIP coverage |
| Test isolation | real LanceDB in tempdirs, never mocked | CLAUDE.md directive; "the store IS the integration surface" |
| Python target | 3.12 baseline; CI runs 3.12 + 3.13 | already configured |
| Compliance profile default | `"none"` | personal project default; deployers opt into `"gdpr"`/`"bipa"` |

---

## V0 acceptance criteria

All must hold before declaring v0 done:

- [ ] `uv sync && uv run pytest` green on a fresh clone
- [ ] CI matrix (3.12, 3.13) green on a clean push
- [ ] **Roundtrip test**: `observe → label → identify` returns the labeled entity
- [ ] **Knowledge test**: `remember` → `recall` returns the fact, with bi-temporal `valid_from`
- [ ] **Correction test**: `merge` two entities → `undo` → both entities restored with facts intact
- [ ] **Privacy test**: `forget` + `prune` removes the vector from LanceDB version history (tested by `checkout(pre_forget_version)` returning no row)
- [ ] **MCP test**: `uv run python -m vemem.mcp_server` responds to `list_tools` with all nine ops
- [ ] **CLI test**: `vm label --help` prints; `vm inspect <entity>` shows facts + observation count
- [ ] **Bridge test**: `docs/examples/bridge.py` runs end-to-end with fakes and prints expected output
- [ ] **README quickstart**: copy-pasting runs without modification (asserted by integration test)

---

## Explicit non-goals for v0 (deferred to v0.1+)

- Auto-clustering of unlabeled observations (only suggestion surface, no commit)
- VisMemBench
- Framework adapters (LangChain / LlamaIndex / CrewAI)
- Second production encoder beyond the experimental CLIP stub
- Second storage backend (pgvector, Chroma, Qdrant)
- Graph DB relationships beyond flat Relationship table
- Co-occurrence binding groups (M3-Agent style cross-modal)
- HTTP MCP transport (stdio only)
- Type-level auto-recognition (only instance-level)
- Voting-based auto-suggest weighting
- Cryptographic erasure / Renewable Biometric References (prune-based erasure only)
- Multi-writer concurrency beyond optimistic stamps

---

## Estimated size

| Wave | Work | Wall-clock if parallelized |
|---|---|---|
| 1 | ~500 LoC, types + protocols + tests | 1 day (orchestrator) |
| 2A ops | ~1200 LoC | 2–3 days |
| 2B storage | ~900 LoC + LanceDB nuance | 2–3 days |
| 2C encoders | ~300 LoC + fixtures | 1–2 days |
| 3D MCP | ~400 LoC | 1–2 days |
| 3E tool schemas | ~150 LoC + snapshots | 1 day |
| 3F CLI | ~500 LoC | 1–2 days |
| 3G bridge | ~200 LoC + docs | 1 day |
| 4 integration | ~400 LoC + release docs | 1 day |

With three parallel agents in Wave 2 and four in Wave 3: **~7–10 days wall-clock**, **~15–20 agent-days of effort total**.

Solo sequential (no Loop 8): ~3 weeks.

---

## Risks & contingencies

1. **LanceDB API drift.** We rely on `optimize(Prune{older_than:0})` for real erasure. Mitigation: pin `lancedb>=0.19,<0.20` and contract-test prune behavior.
2. **InsightFace install friction.** onnxruntime has platform quirks on Apple Silicon. Mitigation: CPU-only onnxruntime in deps, GPU documented as opt-in.
3. **MCP Python SDK is young.** Breaking changes possible. Mitigation: pin a known-good version; integration tests via direct JSON-RPC if SDK fails.
4. **Cross-encoder false confidence.** Identify must fail gracefully when gallery has no compatible encoder — tested explicitly.
5. **Bi-temporal bugs are subtle.** Mitigation: property-based tests (Hypothesis) on binding supersede invariants.
6. **Fixture image licensing.** Mitigation: use only Wikimedia Commons CC0 or user's own photos; document provenance in `tests/fixtures/README.md`.

---

## Execution — Loop 8 spawn commands

When Wave 1 is green and pushed:

```bash
# Wave 2: three agents in parallel
bash ~/.claude/hooks/worktree-agent.sh feat/core-ops \
  "Implement Wave 2A per docs/plan/v0-implementation.md. TDD loop.
   Do not touch storage/ or encoders/."

bash ~/.claude/hooks/worktree-agent.sh feat/storage-lancedb \
  "Implement Wave 2B per docs/plan/v0-implementation.md. TDD loop.
   Do not touch core/ (only import) or encoders/."

bash ~/.claude/hooks/worktree-agent.sh feat/encoders \
  "Implement Wave 2C per docs/plan/v0-implementation.md. TDD loop.
   Do not touch core/ (only import) or storage/."

# Merge order: 2A, 2B, 2C. Run regression after each.
bash ~/.claude/hooks/worktree-merge.sh feat/core-ops
bash ~/.claude/hooks/worktree-merge.sh feat/storage-lancedb
bash ~/.claude/hooks/worktree-merge.sh feat/encoders
bash ~/.claude/hooks/regression.sh

# Wave 3: four agents in parallel
bash ~/.claude/hooks/worktree-agent.sh feat/mcp-server "Implement Wave 3D…"
bash ~/.claude/hooks/worktree-agent.sh feat/openai-tools "Implement Wave 3E…"
bash ~/.claude/hooks/worktree-agent.sh feat/cli "Implement Wave 3F…"
bash ~/.claude/hooks/worktree-agent.sh docs/bridge-example "Implement Wave 3G…"

# Merge + regression, then Wave 4 (orchestrator).
```

## Open questions (non-blocking; park for execution)

1. **LanceDB index refresh cadence.** Newly-inserted embeddings may not appear in ANN results until the index rebuilds. Wave 2B should expose a `refresh()` method and decide on automatic-vs-manual policy. Revisit after 2B is drafted.
2. **Property-based test scope.** Hypothesis is overkill for v0 but high-value for binding supersede invariants. Decide in Wave 2A whether to add.
3. **Pydantic at MCP boundary** — use pydantic v2 for request/response validation, or the MCP SDK's own Schema types. Wave 3D decides.
4. **`vm repair` recovery semantics.** The sidecar `InFlightOps` idea from spec §6 — does v0 actually ship it, or do we rely on LanceDB's atomic commits? Lean: start without InFlightOps, add when the first crash recovery bug happens.
