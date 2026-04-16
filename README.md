# vemem

**Persistent visual entity memory for AI agents.** Self-hosted, open source, model-agnostic.

vemem is the memory layer for what AI agents *see*, not just what they're told. It resolves image observations (faces, objects, places) to named entities and accumulates knowledge against those entities across sessions — the piece that's missing from every text-based memory system.

## Status

Early development — pre-alpha, face-first, `v0` in progress.

- Specification: [`docs/spec/identity-semantics.md`](./docs/spec/identity-semantics.md)
- Architecture + contributor notes: [`CLAUDE.md`](./CLAUDE.md)

## What it does

Given an image, `identify()` returns named entities a text LLM can reason about — no training, no prompt stuffing, no per-session re-introduction. Corrections (`relabel`, `merge`, `split`), privacy-safe deletion (`forget` with LanceDB prune), and bi-temporal audit are first-class.

Works as:
- a Python library you import
- an MCP server any LLM client can call
- OpenAI-compatible tool schemas for function-calling pipelines

## Installation

```bash
uv pip install vemem          # PyPI (once published)
# or
git clone https://github.com/linville-charlie/vemem
cd vemem && uv sync
```

## Quick start

```python
from vemem import Store

store = Store()                                    # LanceDB on disk, ~/.vemem by default
obs = store.observe(image, modality="face")        # detect + embed
candidates = store.identify(image)                 # ANN lookup — [] if first time

# bind an observation to a named entity
charlie = store.label(obs, entity_name="Charlie")
store.remember(charlie.id, fact="runs marathons")

# later, different session, different image
matches = store.identify(other_image)
# → [Candidate(entity="Charlie", confidence=0.94, facts=["runs marathons"])]
```

## Compliance

vemem stores biometric identifiers. If you deploy it, **you are the data controller** under GDPR / BIPA / CCPA.

- `forget(entity_id)` performs hard delete + LanceDB version prune (actual erasure, not soft-delete)
- `restrict(entity_id)` handles GDPR Art. 18 (stop processing without erasing)
- `export(entity_id)` handles GDPR Art. 20 (portability)
- Face modality ships with `consent_required=True` by default

See `docs/spec/identity-semantics.md` §7 for the non-negotiable compliance rules.

## License

MIT — see [LICENSE](./LICENSE).
