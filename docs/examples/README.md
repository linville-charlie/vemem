# vemem examples

Hands-on tours of how to use vemem. Each file is self-contained and
copy-paste runnable (or copy-paste editable, for the config) once you have
the repo installed via `uv sync`.

| File | What it shows |
|---|---|
| [`bridge.py`](./bridge.py) | End-to-end Python integration with **stub** VLM/LLM — shows the shape without real models. |
| [`real_bridge.md`](./real_bridge.md) | **Copy-paste recipes** for Ollama, OpenAI, and Claude with real models. Also covers corrections, the 30-day undo window, source-image linking, and MCP actor attribution. |
| [`mcp_usage.md`](./mcp_usage.md) | Using the vemem MCP server from any MCP-capable client, with a JSON-RPC walkthrough of every tool. |
| [`claude_desktop_config.json`](./claude_desktop_config.json) | Ready-to-paste Claude Desktop configuration for the vemem MCP server. |
| [`cli_tour.md`](./cli_tour.md) | Guided walkthrough of the `vm` console script — observe, label, remember, identify, undo, forget. |
| [`openai_tools.md`](./openai_tools.md) | Using the OpenAI-compatible tool schemas (`vm export-tools`) for function-calling agents (OpenAI, Groq, together.ai, Ollama in tools mode). |

## Quick start

```bash
uv sync
uv run python docs/examples/bridge.py      # canned two-session scenario
uv run python -m vemem.mcp_server          # stdio MCP server
vm --help                                  # CLI
```

## The bridge pattern (`bridge.py`)

`bridge.py` is the shortest honest answer to **"how do I make my LLM
remember who it saw?"** It is ~150 lines and has three moving parts:

1. a VLM that turns image bytes into a scene description,
2. vemem (`observe → label → identify → recall`) that resolves identity
   and attaches knowledge against stable entity ids,
3. a text LLM that reasons over the assembled context.

The fake VLM and fake LLM in `bridge.py` are deterministic Python functions
so the example runs with no network, no API keys, and no model weights.
Replace them with real calls to plug into production.

### Swap in a real VLM

Replace the `describe_scene` placeholder with a call to any vision-language
model. The contract is `bytes -> str`.

**Ollama + Qwen2-VL (local, no API key):**

```python
import ollama

def describe_scene(image_bytes: bytes) -> str:
    result = ollama.chat(
        model="qwen2-vl:7b",
        messages=[{
            "role": "user",
            "content": "Describe the scene in one sentence.",
            "images": [image_bytes],
        }],
    )
    return result["message"]["content"].strip()
```

**OpenAI vision:**

```python
import base64
from openai import OpenAI

client = OpenAI()

def describe_scene(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the scene in one sentence."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    )
    return (result.choices[0].message.content or "").strip()
```

**Anthropic (Claude) vision:**

```python
import base64
from anthropic import Anthropic

client = Anthropic()

def describe_scene(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    msg = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64,
                }},
                {"type": "text", "text": "Describe the scene in one sentence."},
            ],
        }],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
```

### Swap in a real LLM

Replace the `chat` placeholder. The contract is `(user_msg: str, context: str) -> str`
and the `context` is the vemem-assembled prompt-ready string
(`"People visible: Charlie (conf 0.94). Known facts: [...]"`).

**Ollama:**

```python
import ollama

def chat(user_msg: str, context: str) -> str:
    result = ollama.chat(
        model="llama3.1",
        messages=[
            {"role": "system", "content": f"Visual context:\n{context}"},
            {"role": "user",   "content": user_msg},
        ],
    )
    return result["message"]["content"]
```

**OpenAI:**

```python
from openai import OpenAI

client = OpenAI()

def chat(user_msg: str, context: str) -> str:
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Visual context:\n{context}"},
            {"role": "user",   "content": user_msg},
        ],
    )
    return result.choices[0].message.content or ""
```

**Claude:**

```python
from anthropic import Anthropic

client = Anthropic()

def chat(user_msg: str, context: str) -> str:
    msg = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=512,
        system=f"Visual context:\n{context}",
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
```

### Swap in real vemem backends

The example uses `FakeStore` (in-memory, under `tests/`) plus stub
detector/encoder so it can run in CI. For production:

```python
from vemem.encoders.insightface_detector import InsightFaceDetector
from vemem.encoders.insightface_encoder import InsightFaceEncoder
from vemem.storage.lancedb_store import LanceDBStore

store = LanceDBStore(path="~/.vemem")
detector = InsightFaceDetector()
encoder = InsightFaceEncoder()
bridge = Bridge(store=store, clock=SystemClock(), encoder=encoder, detector=detector)
```

All ops accept the `Store` / `Encoder` / `Detector` protocols directly, so
the bridge doesn't change when you swap backends — only the constructor
arguments do.

## When to use which surface

| Surface | Use when |
|---|---|
| `bridge.py` (library) | You're writing Python and you want full control over the VLM→LLM loop. Your agent framework already does scheduling; you want vemem calls inline. |
| MCP server | You're using Claude Desktop / Cursor / any MCP client. It auto-discovers the tools and the LLM drives the flow via function calls. No bridge code needed. |
| OpenAI tool schemas | You're using an OpenAI-compatible LLM (Groq, together.ai, Ollama-in-tools-mode) and want the model to call `observe_image` / `identify_image` / `remember` as tools. |
| CLI (`vm`) | You're a human doing manual labeling, inspecting the store, or scripting a one-off migration. |

The four surfaces are views over the same core ops (`src/vemem/core/ops.py`).
Pick whichever fits your host; you can mix them (e.g. MCP for conversational
labeling + CLI for one-off `vm forget`).

## Further reading

- Spec (load-bearing): [`../spec/identity-semantics.md`](../spec/identity-semantics.md)
- Architecture overview: [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
- V0 implementation plan: [`../plan/v0-implementation.md`](../plan/v0-implementation.md)
- Compliance checklist (if you deploy): [`../../COMPLIANCE.md`](../../COMPLIANCE.md)
- Security disclosure: [`../../SECURITY.md`](../../SECURITY.md)
