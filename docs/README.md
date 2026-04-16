# vemem documentation

```
docs/
├── README.md                      ← you are here
├── ARCHITECTURE.md                how the layers fit
├── spec/
│   └── identity-semantics.md      load-bearing design spec
├── plan/
│   └── v0-implementation.md       wave-by-wave build record
└── examples/
    ├── README.md                  tour of the four usage surfaces
    ├── bridge.py                  runnable VLM↔vemem↔LLM reference
    ├── cli_tour.md                vm command walkthrough
    ├── mcp_usage.md               MCP tool reference + JSON-RPC trace
    ├── openai_tools.md            function-calling schemas guide
    └── claude_desktop_config.json copy-paste MCP config
```

## Where to start

| You are… | Read |
|---|---|
| Evaluating vemem | [`../README.md`](../README.md) → [`examples/bridge.py`](examples/bridge.py) |
| Building an integration | [`ARCHITECTURE.md`](ARCHITECTURE.md) → [`examples/`](examples/) matching your host |
| Writing a PR | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) → [`ARCHITECTURE.md`](ARCHITECTURE.md) → [`spec/identity-semantics.md`](spec/identity-semantics.md) |
| Deploying to users | [`../COMPLIANCE.md`](../COMPLIANCE.md) → [`../SECURITY.md`](../SECURITY.md) |
| An AI agent working on vemem | [`../CLAUDE.md`](../CLAUDE.md) → [`plan/v0-implementation.md`](plan/v0-implementation.md) |

## Top-level files (repo root)

- [`../README.md`](../README.md) — landing page, quick start, "does it replace mem0?"
- [`../CHANGELOG.md`](../CHANGELOG.md) — release notes
- [`../COMPLIANCE.md`](../COMPLIANCE.md) — GDPR/BIPA/CCPA checklist for deployers
- [`../SECURITY.md`](../SECURITY.md) — responsible disclosure
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — dev setup, verify gate, conventions
- [`../CLAUDE.md`](../CLAUDE.md) — conventions for AI agents working on this repo
- [`../LICENSE`](../LICENSE) — MIT

## The spec is load-bearing

`spec/identity-semantics.md` is not a nice-to-have — it's the design document every operation in `src/vemem/core/ops.py` implements. If the code and the spec disagree, one of them is a bug. The spec wins in design discussions; the code wins in "what does the library actually do" questions, but drifted code is always a defect.

Read it before:
- Adding a new op
- Changing binding, fact, or entity semantics
- Proposing a `Store` Protocol change
- Reviewing a PR that touches `core/ops.py`

## Surface cheat sheet

Four user-facing surfaces, one semantic core:

| Surface | Code | When |
|---|---|---|
| Python library | `from vemem import Vemem` | Your own app |
| MCP server | `python -m vemem.mcp_server` | Claude Desktop, Cursor |
| OpenAI tool schemas | `vm export-tools > tools.json` | OpenAI/Anthropic/Gemini/Ollama |
| CLI | `vm label`, `vm inspect`, … | Manual work, scripts |

See [`examples/README.md`](examples/README.md) for a side-by-side.

## Not (yet) in the docs

These exist in the code but don't have their own doc pages because they're rarely used by library consumers:

- Schema migrations — see `src/vemem/storage/migrations.py`; v0 has one schema version
- Snapshot tests for the OpenAI schemas — see `tests/tools/snapshots/tools.json`
- Property-based tests for bi-temporal invariants — not yet written (v0.1 candidate)

A full API reference (sphinx / pdoc) is also deferred; docstrings in the source are authoritative for now.
