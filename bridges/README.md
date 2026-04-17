# bridges/

Standalone demonstration scripts that show how to wire vemem into any
ollama-compatible agent setup. **These are demos, not first-party
integrations.** For a supported host integration, see
[`integrations/`](../integrations/).

| File | What it is |
|---|---|
| [`openclaw_bridge.py`](./openclaw_bridge.py) | Self-contained CLI that wires [ollama](https://ollama.com) cloud models (any VLM + any thinking model) through vemem. `label`, `observe`, `ask` subcommands. Good for reading end-to-end: image → InsightFace → vemem identify + recall → thinking LLM. |

## First-party integrations

If you're running a supported host, use the real integration instead:

- **openclaw** → [`integrations/openclaw`](../integrations/openclaw/) — automatic image-description provider, no agent prompting required.

The integrations/ tree is covered by tests and changelog entries; this
bridges/ tree is just reference code.

## openclaw_bridge.py

Demonstrates the three-actor flow end-to-end:

```
image ─► InsightFace (detect + embed)          ─┐
       ─► qwen3.5:cloud (VLM, optional) ─ scene ─┤
                                                  ▼
                                          text-only context
                                                  │
                                                  ▼
                                  glm-5.1:cloud (thinking LLM)
```

Two guarantees the bridge enforces:

1. **The thinking LLM never sees images.** A `_assert_text_only()` guard runs before every `client.chat()` call; the function raises `RuntimeError` if any `images` field or image content-part slipped into a message. Unit-tested against both failure modes.
2. **vemem is the identity source of truth.** VLM scene descriptions are appended to the text context only when explicitly requested via `--use-vlm`; the default path emits *only* what vemem knows (recognized entity + recalled facts, or "unrecognized").

### Quick start

```bash
# Install the optional runtime dep
uv pip install ollama

# One-time: label a face + attach a fact
uv run python bridges/openclaw_bridge.py label path/to/charlie.jpg Charlie \
    --fact "training for Boston Marathon"

# Later: ask the thinking LLM about any photo (vemem only, safest)
uv run python bridges/openclaw_bridge.py ask path/to/new_photo.jpg "How's training going?"

# Or include a VLM scene description in the context
uv run python bridges/openclaw_bridge.py ask photo.jpg "What's happening?" --use-vlm
```

Env vars: `VEMEM_HOME` (LanceDB path), `OLLAMA_HOST` (default `http://localhost:11434`), `LLM_MODEL` (default `glm-5.1:cloud`), `VLM_MODEL` (default `qwen3.5:cloud`).

Swap the ollama cloud models for OpenAI, Anthropic, or anything function-calling-capable — the contracts are `bytes → str` (VLM) and `(user_msg, context_text) → str` (LLM).
