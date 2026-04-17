# bridges/

Reference integrations that wire vemem into real agent frameworks. Everything here is **orthogonal to the core library** — you can delete this folder and vemem still works. It's here so integrators have a concrete starting point instead of a blank file.

| File | What it is | When to reach for it |
|---|---|---|
| [`openclaw_bridge.py`](./openclaw_bridge.py) | Self-contained CLI that wires [ollama](https://ollama.com) cloud models (any VLM + any thinking model) through vemem. `label`, `observe`, `ask` subcommands. | You want a minimal, understandable Python reference for "image → vemem identity → text LLM answer." Great for local testing and reading. |
| [`vemem_http.py`](./vemem_http.py) | Long-lived HTTP sidecar that exposes a single `/describe` endpoint over the vemem observe + identify + recall pipeline. Pre-loads InsightFace + LanceDB once on startup. | You're writing a plugin in another language (TypeScript, Go, Rust) and want to call vemem over HTTP instead of binding to the Python library. This is the pattern [`docs/examples/openclaw-plugin`](../docs/examples/openclaw-plugin/) uses. |

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

## vemem_http.py

Framework-agnostic HTTP microservice. Start it once; many clients can hit it.

```
POST /describe   { "path": "/abs/file.jpg" }   → { "text": "vemem: ..." }
POST /health     {}                             → { "ok": true }
```

Text returned by `/describe` is ready to inject directly into a prompt:

```
vemem: 1 face(s) detected.
Recognized: Charlie (conf 0.94). Known facts: [training for Boston Marathon]
```

Why path-based, not base64? Many agent-host tool layers cap argument size. A 1–2MB image becomes 1.3–2.7MB of base64, which truncates silently in several clients. A filesystem path moves the bytes out-of-band and sidesteps the class entirely. (The MCP server in `src/vemem/mcp_server/` now accepts `image_path` too, for the same reason.)

### Run it

```bash
uv run python bridges/vemem_http.py                 # starts server
uv run python bridges/vemem_http.py --test           # self-test on first image in CWD
```

Env vars: `VEMEM_HOME`, `VEMEM_HTTP_HOST` (default `127.0.0.1`), `VEMEM_HTTP_PORT` (default `18790`).

### Design notes

- **Observation on read.** `/describe` calls `observe_image` before `identify`, so every image the system ever sees is persisted. This lets you retroactively label unknown faces with their existing observation id.
- **Cache refresh.** If another process (CLI, MCP server) labels a new face while the sidecar is running, the sidecar refreshes its encoder-tables cache before identify so newly-registered encoders are visible.
- **Graceful port collision.** If port 18790 is already bound (common when a plugin host loads the bridge twice), the second instance detects `EADDRINUSE` and exits 0 — the running server wins.

## Using both together

The `openclaw-plugin` example in [`docs/examples/openclaw-plugin`](../docs/examples/openclaw-plugin/) shows a TypeScript plugin that:

1. Spawns `vemem_http.py` on plugin load,
2. Registers vemem as the host's image-description provider,
3. Writes each incoming image attachment to a temp file,
4. POSTs the path to `/describe`,
5. Returns the text — which the host injects as the image's `Description` before the LLM ever sees the message.

The thinking LLM receives only text. Identity + facts flow from vemem. Corrections (label / merge / forget / undo) stay on the vemem MCP server, reached through the host's tool-call pathway.
