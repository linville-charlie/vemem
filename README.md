# vemem

**Persistent visual entity memory for AI agents.** Self-hosted, open source, model-agnostic.

vemem is the memory layer for what AI agents *see*, not just what they're told. It resolves image observations (faces, objects, places) to named entities and accumulates knowledge against those entities across sessions — the piece that's missing from every text-based memory system.

## Status

Pre-alpha. v0 implements the full op surface from [`docs/spec/identity-semantics.md`](./docs/spec/identity-semantics.md) with 260+ tests, but the API is not yet frozen.

| | |
|---|---|
| Docs index | [`docs/README.md`](./docs/README.md) |
| Architecture | [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) |
| Spec | [`docs/spec/identity-semantics.md`](./docs/spec/identity-semantics.md) |
| Plan | [`docs/plan/v0-implementation.md`](./docs/plan/v0-implementation.md) |
| Examples | [`docs/examples/`](./docs/examples/) — MCP config, OpenAI tools, CLI tour, bridge.py |
| Changelog | [`CHANGELOG.md`](./CHANGELOG.md) |
| Compliance | [`COMPLIANCE.md`](./COMPLIANCE.md) |
| Security | [`SECURITY.md`](./SECURITY.md) |
| Contributing | [`CONTRIBUTING.md`](./CONTRIBUTING.md) |

## What it does

Given an image, `identify()` returns named entities a text LLM can reason about — no training, no prompt stuffing, no per-session re-introduction. Corrections (`relabel`, `merge`, `split`), privacy-safe deletion (`forget` with LanceDB version prune), and bi-temporal audit are first-class.

Works as:
- **a Python library** you import (`from vemem import Vemem`)
- **an MCP server** any MCP-capable LLM client can call (Claude Desktop, Cursor, custom agents)
- **OpenAI function-calling tool schemas** for any non-MCP function-calling LLM (OpenAI, Anthropic, Gemini, Ollama)
- **a CLI** (`vm label`, `vm inspect`, `vm forget`, …) for manual work and debugging

## Installation

```bash
git clone https://github.com/linville-charlie/vemem
cd vemem
uv sync
```

On first use, `uv run python -c "from vemem import Vemem; Vemem()"` will download InsightFace's `buffalo_l` weights (~200MB) to `~/.insightface/`. Offline-only deployments should pre-install the weights.

Python 3.12 and 3.13 supported. 3.14 is blocked by a lancedb 0.19 segfault; we pin to 3.13 locally via `.python-version`.

## Quick start

```python
from vemem import Vemem

with Vemem() as vem:                                # LanceDB at ~/.vemem, InsightFace
    observations = vem.observe(image_bytes)         # detect + embed
    candidates = vem.identify(image_bytes)          # [] on first image
    if not candidates:
        entity = vem.label([o.id for o in observations], name="Charlie")
        vem.remember(entity.id, "training for Boston Marathon")

    # later — different session, different image of Charlie
    matches = vem.identify(another_image)
    # → [Candidate(entity=Charlie, confidence=0.94,
    #              facts=[Fact("training for Boston Marathon")])]
```

Test it without the real encoder (no model download, works offline):

```python
from vemem import Vemem
from tests.support.fake_store import FakeStore  # test-only, not shipped

class StubDetector:
    id = "stub/detector@1"
    def detect(self, img): return [(0, 0, 100, 100)]

class StubEncoder:
    id = "stub/encoder@1"; dim = 4
    def embed(self, img): return (float(img[0]), 1.0, 0.0, 0.0)

vem = Vemem(store=FakeStore(), encoder=StubEncoder(), detector=StubDetector())
```

## Run as an MCP server

```bash
uv run python -m vemem.mcp_server
```

Then paste [`docs/examples/claude_desktop_config.json`](./docs/examples/claude_desktop_config.json) into your Claude Desktop config. Remote Claude can now call `identify_image`, `label`, `recall`, `forget`, … against your local store without your biometric data ever leaving the laptop.

## Run as a CLI

```bash
uv run vm --help
uv run vm observe photo.jpg
uv run vm label obs_... --name Charlie
uv run vm remember <entity_id> --fact "runs marathons"
uv run vm recall <entity_id>
```

## Compliance

vemem stores biometric identifiers. **If you deploy it, you are the data controller under GDPR / BIPA / CCPA.**

- `forget(entity_id)` → hard delete + LanceDB `optimize(cleanup_older_than=0)` (verified by test — spec §4.5, §7)
- `restrict(entity_id)` — GDPR Art. 18 (stop processing without deleting)
- `export(entity_id)` — GDPR Art. 20 (portability)
- EventLog never stores raw biometric content

See [`COMPLIANCE.md`](./COMPLIANCE.md) for the full deployer checklist.

## Does it replace mem0 / supermemory?

No — **they compose.** vemem owns visual identity (image → entity id); mem0 / supermemory own text-conversational memory. Typical pipeline:

```
image → VLM → scene text ─┐
      → vemem → entity_id ├──► LLM
mem0.search(user_id=entity_id) → facts about that entity ─┘
```

The `entity_id` vemem returns becomes the `user_id` mem0 keys on. Most serious multimodal agents will want both.

## License

MIT — see [LICENSE](./LICENSE).
