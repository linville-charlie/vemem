# Changelog

All notable changes to vemem.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions before 1.0 may break API without notice.

## [Unreleased]

### Added

**First-party integrations**
- `vemem.integrations.openclaw` — automatic image-understanding provider for [openclaw](https://openclaw.dev). Ships a long-lived HTTP sidecar (`vemem-openclaw-sidecar` console script) plus a drop-in TypeScript plugin under `integrations/openclaw/plugin/`. Registers vemem as openclaw's media-understanding provider so every image attachment is transparently face-recognized + fact-recalled before the thinking LLM sees the conversation — no agent-side tool calls required.
- `integrations/` top-level directory for future host integrations. Each entry is supported: tests, changelog, and issues welcome.

**MCP server**
- `observe_image` and `identify_image` accept an optional `image_path` alongside `image_base64`, sidestepping tool-argument size caps that silently truncate multi-MB images. Exactly one of the two inputs must be provided.

**Bridges**
- `bridges/openclaw_bridge.py` — ollama-powered CLI demo (VLM + thinking LLM via vemem). Illustrates the observe → identify → recall → reason flow end-to-end with a hard invariant that image bytes never reach the thinking LLM.
- `bridges/` now clearly scoped as demo scripts; production integrations live under `integrations/`.

### Changed

- Root README adds an Integrations section listing supported hosts.
- pyproject.toml declares the `vemem-openclaw-sidecar` console script.

## [0.1.0] — 2026-04-17 — initial v0

First pre-alpha release. All surfaces from the v0 implementation plan are live; the API isn't frozen yet.

### Added

**Core model** (spec §3–§4)
- Frozen domain types: `Observation`, `Embedding`, `Entity`, `Binding`, `Fact`, `Event`, `Relationship`, `EventLog`, `Candidate`
- `Store`, `Encoder`, `Detector`, `Clock` Protocols — the pluggability seam
- UUID v7 IDs (no third-party deps)
- `VemError` hierarchy (`ModalityMismatchError`, `KindMismatchError`, `EntityUnavailableError`, `OperationNotReversibleError`, `NoCompatibleEncoderError`, `SchemaVersionError`)

**Operations** — thirteen, all bi-temporal, all with full undo
- `observe`, `identify`, `label`, `relabel`, `merge`, `split`, `forget`, `restrict`, `unrestrict`, `remember`, `recall`, `undo`, `export`
- `RecallSnapshot` with facts + events + relationships (Wave 4.1)
- `Candidate.facts` auto-populated by `identify` (Wave 4.1)
- `undo` dispatches per op; scans store.list_events for the most recent reversible event by actor

**Storage**
- `LanceDBStore` — production Store implementation
  - One embeddings table per `encoder_id`
  - PyArrow schemas per table
  - Schema version tracking for future migrations
  - **GDPR-verified erasure**: `forget` + `prune_versions(now)` physically removes embeddings from LanceDB's version history (`test_forget_physically_removes_vectors_from_version_history`)
- `FakeStore` (test-only, in-memory) — pure-Python reference Store implementation used by contract tests

**Encoders**
- `InsightFaceEncoder` — 512-d ArcFace via `buffalo_l`, L2-normalized output, auto-download on first use
- `InsightFaceDetector` — SCRFD face detection, returns `(x, y, w, h)` bboxes
- `CLIPEncoder` — experimental scaffold via `open_clip_torch`, dim probed at init
- `crop_image` helper, PIL-backed

**Four surfaces**
- Python library via `from vemem import Vemem`
- MCP server (`python -m vemem.mcp_server`) with 14 tools over stdio
- OpenAI function-calling JSON schemas (`vm export-tools`, snapshot-tested)
- CLI (`vm`, Typer + Rich) with 17 commands

**Vemem facade**
- Context-manager compatible (`with Vemem() as vem:`)
- Accepts injected Store/Encoder/Detector/Clock, or loads LanceDB + InsightFace defaults
- Graceful degradation when InsightFace weights are missing
- Sentinel-based encoder/detector defaults distinguish "omit" from "disable"

**Documentation**
- `docs/spec/identity-semantics.md` — load-bearing design spec
- `docs/plan/v0-implementation.md` — wave-by-wave build plan
- `docs/ARCHITECTURE.md` — layer-by-layer overview
- `docs/examples/` — bridge.py + MCP/CLI/OpenAI walkthroughs + Claude Desktop config
- `COMPLIANCE.md`, `SECURITY.md`, `CONTRIBUTING.md`

**Tests** — 263 passing unit + contract + e2e; 15 gated integration tests opt in via `VEMEM_RUN_INTEGRATION=1`. Core / storage / encoders / mcp / tools / cli / examples / e2e tiers.

### Known limitations

- Python 3.14 not supported by lancedb 0.19.x (segfault at `connect()`); local dev pinned to 3.13 via `.python-version`. CI matrix is 3.12 + 3.13.
- No ANN index built on write — v0 uses LanceDB brute force (fine for < ~10k vectors; noticeable at larger scale).
- `close_binding` is delete + re-add, not atomic. A crash mid-op would leave the binding missing; `repair()` stub will cover this in v0.1 via the EventLog-last-write invariant.
- `CLIPEncoder` is a scaffold, not tuned for object re-identification. Thresholds are not calibrated.
- First use of `InsightFaceEncoder` downloads ~200MB of weights; first use of `CLIPEncoder` downloads ~350MB.

### Deferred to v0.1 (tracked as gaps in Wave 3/4 DONE-doc archaeology)

- `compliance_profile={"gdpr", "bipa", "ccpa", "none"}` config enum
- Cryptographic erasure / Renewable Biometric References
- Auto-clustering of unlabeled observations (scaffold exists; commit policy undecided)
- VisMemBench benchmark
- Property-based tests for bi-temporal invariants (Hypothesis)
- Second storage backend (pgvector, Chroma)
- HTTP MCP transport
- `find_entity_by_name_or_alias` promoted to Protocol
- GitHub Actions Node 20 deprecation cleanup

## Versioning

Pre-1.0 semver: minor bumps (0.X.0) may break the API; patch bumps (0.0.X) won't. We'll tag 1.0 when we're ready to commit to the public surface — at earliest, after VisMemBench v0 and a second production encoder prove the Protocol surface holds.
