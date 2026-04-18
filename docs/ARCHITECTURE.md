# vemem architecture

This is a middle-layer doc: longer than the README, shorter than the full spec. Read it if you're contributing, writing an integration, or evaluating vemem for production.

The load-bearing semantics live in [`spec/identity-semantics.md`](./spec/identity-semantics.md). This doc explains the *shape* of the code — which layer owns which concern, what the stable seams are, and how the four user-facing surfaces compose.

---

## One diagram

```
                                   ┌──────────────────────────────┐
                                   │       YOUR APPLICATION       │
                                   └───┬────────┬────────┬────────┘
                        ┌──────────────┤        │        ├──────────────┐
                        │              │        │        │              │
                  ┌─────▼────┐  ┌──────▼──┐  ┌──▼──────┐  ┌────▼─────┐
                  │ Python   │  │   MCP   │  │ OpenAI  │  │   CLI    │
                  │ (Vemem   │  │ server  │  │ tool    │  │  (vm)    │
                  │  facade) │  │  stdio  │  │ schemas │  │  typer   │
                  └─────┬────┘  └────┬────┘  └────┬────┘  └────┬─────┘
                        │            │            │            │
                        └────────────┴────────────┴────────────┘
                                         │
                                         │  13 ops
                                         ▼
                             ┌───────────────────────────┐
                             │   vemem.core.ops          │
                             │   observe / identify /    │
                             │   label / relabel /       │
                             │   merge / split /         │
                             │   forget / restrict /     │
                             │   remember / recall /     │
                             │   undo / export           │
                             └────┬───────────────┬──────┘
                                  │               │
                  ┌───────────────▼──┐   ┌────────▼──────────┐
                  │ Store Protocol   │   │ Encoder Protocol  │
                  │ Clock Protocol   │   │ Detector Protocol │
                  └────┬─────────────┘   └────┬──────────────┘
                       │                      │
             ┌─────────▼─────────┐   ┌────────▼──────────────┐
             │   LanceDBStore    │   │  InsightFaceEncoder   │
             │   (production)    │   │  InsightFaceDetector  │
             │                   │   │  CLIPEncoder (v0.1)   │
             │   FakeStore       │   │                       │
             │   (tests only)    │   │  Stubs (tests only)   │
             └───────────────────┘   └───────────────────────┘

              storage substrate       encoder substrate
              (vectors + metadata)    (image → vector)
```

## Layers, from top to bottom

### 1. Surfaces — the four ways you invoke vemem

Every surface is a thin wrapper over `vemem.core.ops`. They exist because different hosts (human at a terminal, Claude over MCP, OpenAI function-calling loop, a Python script) want different conventions.

| Surface | Code | Best for |
|---|---|---|
| Python library | `vemem.Vemem` facade or direct ops calls | Your own app wants full control |
| MCP server | `vemem.mcp_server` (FastMCP, stdio) | Claude Desktop, Cursor, any MCP client |
| OpenAI tool schemas | `vemem.tools` + `vm export-tools` | Non-MCP function-calling LLMs |
| CLI | `vemem.cli` (Typer, `vm …`) | Manual labeling, debugging, scripting |

Key invariant: **all four produce the same EventLog entries and leave the store in the same state.** If you label an entity via the CLI and identify it via MCP, it's the same entity.

### 1a. Host integrations — sitting *under* a framework instead of being called by it

Beyond the four surfaces above, vemem ships first-party integrations that slot into specific agent hosts as the automatic image-understanding layer — the agent doesn't call a vemem tool; vemem is just how the host describes images (mem0-style).

| Host | How it's wired | Source of truth |
|---|---|---|
| [openclaw](https://openclaw.dev) | TypeScript plugin at `integrations/openclaw/plugin/` registers `describeImage` via `registerMediaUnderstandingProvider`; spawns `vemem-openclaw-sidecar` (Python, `127.0.0.1:18790`) for the work | `integrations/openclaw/README.md` |

The openclaw integration offers three install paths (skill-only, plugin-only, plugin+bundled-skill). The plugin ships a **mirror** of `skills/vemem/` at `integrations/openclaw/plugin/skills/vemem/` and declares `"skills": ["./skills"]` in its manifest, so enabling the plugin auto-loads the agent skill alongside it. The mirror is kept in sync via `scripts/sync-bundled-skill.sh`. See CONTRIBUTING.md for the edit workflow.

Additional host integrations are welcome under `integrations/`. The HTTP sidecar shape (`POST /describe`, `POST /health` on loopback) is stable within 0.1.x and can be reused by any host that can run a child process.

### 2. Core ops — the semantic layer

`vemem.core.ops` holds 13 pure functions. Each takes a `Store`, a `Clock`, an `actor` string, and whatever op-specific inputs it needs. No global state. No hidden singletons. No I/O except through the Store.

The semantics are specified in `docs/spec/identity-semantics.md` §4. This is the load-bearing part of the project: the bi-temporal binding model, the supersede rules, the correction algebra, the undo mechanics. Changing ops semantics is a spec change, not a code change.

### 3. Protocols — the pluggability seam

`vemem.core.protocols` defines four `typing.Protocol`s: `Store`, `Encoder`, `Detector`, `Clock`. Everything upstream (ops, surfaces) talks to these interfaces; everything downstream (LanceDB, InsightFace, CLIP) satisfies them.

This is the seam that lets you swap storage backends (hypothetical pgvector/Chroma impls) or encoders (CLIP for objects, DINOv3 for scenes) without touching the ops layer. It's also what makes the test suite fast: `FakeStore` + stub encoder gives you the full library behavior with no network, no model weights, no filesystem.

### 4. Backends — where bytes actually live

- **`vemem.storage.lancedb_store.LanceDBStore`** — production. Embeddings by encoder id (one table per encoder), entities, bindings, facts, events, relationships, event log. Versioned by LanceDB; `forget + prune` produces actual GDPR-grade erasure (verified by `tests/storage/test_lancedb_specific.py::test_forget_physically_removes_vectors_from_version_history`).
- **`vemem.encoders.insightface_encoder.InsightFaceEncoder`** — 512-d ArcFace. Default. `id = "insightface/arcface@0.7.3"`.
- **`vemem.encoders.clip_encoder.CLIPEncoder`** — experimental scaffold via `open_clip_torch`. Probes dim at init. Not tuned for v0.
- **`vemem.encoders.insightface_detector.InsightFaceDetector`** — reuses `buffalo_l`'s SCRFD head.

Test-only:
- **`tests/support/fake_store.py`** — in-memory `Store`. Same Protocol contract, pure Python, < 500 LoC.

## Data flow: "Charlie walks into the kitchen"

End-to-end trace through the stack.

```
Camera frame (bytes)
   │
   ▼
┌─────────────────────────┐
│ Vemem.observe(img)      │
│                         │
│ ├─ sha256(img)          │ ← content hash for idempotent observation id
│ ├─ detector.detect(img) │ ← bboxes
│ └─ for each bbox:       │
│     encoder.embed(img)  │ ← 512-d ArcFace vector, L2-normalized
│     obs = Observation(  │
│       id = content_hash,│
│       bbox, modality,   │
│       detector_id, ...  │
│     )                   │
│     store.put_*         │ ← writes to LanceDB tables
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Vemem.identify(img)     │ ← read path
│                         │
│ ├─ encoder.embed(img)   │
│ ├─ store.search_        │ ← ANN over the right encoder's table
│ │   embeddings(...)     │
│ └─ resolve matched      │
│     observations to     │ ← follows current positive bindings
│     current entities    │
│                         │
│ returns list[Candidate] │
│  each with:             │
│   entity (with name)    │
│   confidence            │
│   facts (from recall)   │ ← Wave 4.1 added this
└──────────┬──────────────┘
           │
           ▼
Your agent gets:
  [
    Candidate(entity=Charlie, confidence=0.94,
              facts=[Fact("runs marathons"), Fact("works at Acme")])
  ]
```

The key insight: the identity resolution path (`observe` + `identify`) is **deterministic embedding math**. No LLM. No fact extraction. That's by design — it runs cheap on every frame.

Text only enters via `remember` (caller provides the string) and `recall` (caller reads it back). The caller LLM decides what to do with those strings.

## Repository layout

```
vemem/
├── src/vemem/
│   ├── core/
│   │   ├── enums.py          Polarity, Modality, Kind, Status, Source, Method, OpType
│   │   ├── ids.py            UUID v7 generator (no third-party deps)
│   │   ├── types.py          Observation, Embedding, Entity, Binding, Fact,
│   │   │                     Event, Relationship, EventLog, Candidate
│   │   ├── protocols.py      Store, Encoder, Detector, Clock
│   │   ├── errors.py         VemError hierarchy (ModalityMismatchError, etc.)
│   │   └── ops.py            The 13 core operations
│   ├── storage/
│   │   ├── lancedb_store.py  Production Store implementation
│   │   ├── schemas.py        PyArrow schemas per table
│   │   └── migrations.py     Schema version tracking
│   ├── encoders/
│   │   ├── insightface_encoder.py
│   │   ├── insightface_detector.py
│   │   ├── clip_encoder.py   Experimental
│   │   └── crop.py           PIL-backed helper
│   ├── mcp_server/
│   │   ├── server.py         FastMCP app
│   │   ├── tools.py          One handler per tool
│   │   ├── serialization.py  dataclass → dict
│   │   └── config.py         env-var plumbing + graceful degradation
│   ├── tools/
│   │   ├── schemas.py        14 OpenAI function-calling schemas
│   │   └── export.py         python -m vemem.tools.export
│   ├── cli/
│   │   ├── app.py            Typer app, 17 commands
│   │   ├── context.py        CliContext builder
│   │   └── output.py         Rich-based pretty printers
│   ├── pipeline.py           Shared observe_image recipe
│   ├── facade.py             Vemem class
│   └── __init__.py           Public re-exports
├── tests/
│   ├── core/                 Ops semantics + type invariants
│   ├── storage/              Shared contract tests + LanceDB specifics
│   ├── encoders/             Unit + gated integration
│   ├── mcp_server/           Handler unit + stdio subprocess integration
│   ├── tools/                Snapshot tests
│   ├── cli/                  Typer CliRunner tests
│   ├── examples/             tests for docs/examples/bridge.py
│   ├── e2e/                  Full stack end-to-end through Vemem facade
│   ├── support/fake_store.py Shared test fixture
│   └── fixtures/             Public-domain face images
├── docs/
│   ├── ARCHITECTURE.md       This file
│   ├── spec/
│   │   └── identity-semantics.md   Load-bearing design spec
│   ├── plan/
│   │   └── v0-implementation.md    Wave-by-wave build plan
│   └── examples/
│       ├── README.md
│       ├── bridge.py
│       ├── cli_tour.md
│       ├── mcp_usage.md
│       ├── openai_tools.md
│       └── claude_desktop_config.json
├── README.md                 Project landing page
├── COMPLIANCE.md             GDPR/BIPA/CCPA deployer checklist
├── SECURITY.md               Responsible disclosure
├── CONTRIBUTING.md           Dev setup + conventions
├── CLAUDE.md                 Agent-oriented notes
├── CHANGELOG.md              Release notes
├── pyproject.toml            Deps, scripts, tooling config
└── uv.lock                   Pinned dependency tree
```

## Key invariants

These are the non-negotiables. Code reviews should reject anything that breaks them:

1. **Observations are immutable.** Their id is a content hash over `(source_hash, bbox, detector_id)`. You never update an Observation row; you write a new one if anything changes.
2. **Bindings are append-only with bi-temporal `valid_from` / `valid_to`.** `close_binding(id, at)` sets `valid_to`; no other mutation is allowed. Corrections create new bindings that supersede old ones.
3. **Encoder version is part of identity-of-evidence (spec §3.1a).** Two encoders with the same model but different versions produce disjoint galleries. `identify(encoder_id=A)` on a gallery built with encoder_id=B returns nothing — not an approximate match, not a silent fallback.
4. **`forget(entity_id)` is irreversible by `undo`.** It's also required to call `prune_versions(now)` so biometric data is physically gone from LanceDB's version history. This is the GDPR Art. 17 contract.
5. **EventLog rows never contain biometric content** — only IDs and counts. This lets the audit trail outlive the data without becoming a regulated artifact itself.
6. **The deterministic hot path is pure.** `observe`, `identify`, `store.*`, `encoder.*`, `detector.*` never call an LLM. Everything LLM-mediated (fact extraction, conflict resolution) is the caller's responsibility.
7. **Core has no third-party deps.** `vemem.core` imports only stdlib. Third-party deps (lancedb, insightface, mcp, typer, …) are confined to storage / encoders / surfaces.

## Dependency direction

Strictly top-down. If you find yourself wanting to import from a lower-right module into an upper-left one, you're probably putting a method on the wrong layer.

```
surfaces (cli, mcp_server, tools, facade)
     │
     ▼
core.ops  ─────► pipeline ─────► encoders
     │                              │
     ▼                              ▼
core.protocols ◄─── storage.*  ◄──── (implements)
     │
     ▼
core.types / core.enums / core.ids / core.errors
```

`pipeline.py` depends on both core (types, protocols) and encoders (crop_image). That's the one cross-cut — it exists specifically to bridge them for the `observe_image` recipe that CLI, MCP, and the bridge example all share.

## Testing strategy

Three tiers:

1. **Unit tests** (always run, offline). Every module. Mocks the minimum necessary; no network, no model weights. ~1s for the full suite.
2. **Contract tests** (always run, offline). `tests/storage/test_store_contract.py` runs the same 29 tests against both `FakeStore` and `LanceDBStore`. Any third `Store` implementation drops in and inherits 58 test invocations.
3. **Integration tests** (gated by `VEMEM_RUN_INTEGRATION=1`). Real InsightFace, real open_clip, real MCP stdio subprocess. Downloads ~550MB of weights on first run.
4. **End-to-end tests** (`tests/e2e/`). Full stack from `Vemem()` through real LanceDBStore with stub encoder/detector. Exercises spec §4 acceptance scenarios offline.

Run defaults: `uv run pytest` → unit + contract + e2e, 263+ tests, ~3 seconds.
Full run: `VEMEM_RUN_INTEGRATION=1 uv run pytest` → ~1 minute, 280+ tests.

## Where the complexity lives

Roughly ordered by how much design attention each layer absorbs:

1. **Semantics of the ops** (spec §4, `core/ops.py`). This is most of the hard thinking — bi-temporal bindings, correction algebra, undo as inverse-op dispatch, ModalityMismatch vs KindMismatch, what happens to facts on merge vs split.
2. **Protocol design**. The surface of `Store` is the thing that gets reused. If it's wrong, every backend has to work around it. 28 methods at v0; grows cautiously.
3. **Privacy surface** (`forget + prune`, spec §7, `COMPLIANCE.md`). The verified "prune actually removes" test is load-bearing for every claim the project makes about GDPR/BIPA.
4. **Encoder integration** (Wave 2C). Getting InsightFace + open_clip to load on diverse hosts without flaming out on missing weights is the edgy part.
5. Everything else (surfaces, serialization, CLI, docs). Mechanical once the layers above are right.

## Reading order if you're new

1. `README.md` — what + why, in 2 minutes
2. `docs/examples/bridge.py` — the usage pattern, in ~150 lines of Python
3. `docs/spec/identity-semantics.md` §1–§4 — the semantics
4. `src/vemem/core/ops.py` — the thirteen ops
5. `src/vemem/core/protocols.py` — the pluggability seam
6. `COMPLIANCE.md` if you're deploying
7. `CONTRIBUTING.md` if you're sending a PR

For AI agents: start with `CLAUDE.md`.
