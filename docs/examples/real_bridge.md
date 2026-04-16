# Real-world bridge: VLM + vemem + LLM

`bridge.py` shows the **shape** with stubs. This doc shows the **wiring** with real models. Copy the one closest to your stack and adjust.

---

## Ollama (fully local, no API keys)

```bash
pip install ollama   # or: brew install ollama && ollama pull qwen2-vl:7b && ollama pull llama3.1
```

```python
import ollama
from vemem import Vemem, Source

vem = Vemem()  # LanceDB at ~/.vemem, InsightFace face encoder

# ---- write path: camera frame comes in ----

image_bytes = open("photo.jpg", "rb").read()

observations = vem.observe(image_bytes)
candidates = vem.identify(image_bytes, k=3, min_confidence=0.4)

if candidates:
    # Known person — attach a VLM scene note as a fact
    entity = candidates[0].entity
    scene = ollama.chat(
        model="qwen2-vl:7b",
        messages=[{
            "role": "user",
            "content": "Describe what this person is doing in one sentence.",
            "images": [image_bytes],
        }],
    )["message"]["content"]
    vem.remember(entity.id, scene, source=Source.VLM)
else:
    # Unknown — just store the observation. Label later via CLI or assistant.
    print(f"New face detected: {[o.id for o in observations]}")
    print("Label with: vem.label([obs_id], name='Charlie')")

# ---- read path: assistant needs context ----

candidates = vem.identify(image_bytes, k=3)
context_parts = []
for c in candidates:
    snapshot = vem.recall(c.entity.id)
    facts = "; ".join(f.content for f in snapshot.facts)
    context_parts.append(f"{c.entity.name} (conf {c.confidence:.2f}): {facts}")
context = "People visible:\n" + "\n".join(context_parts) if context_parts else "No known faces."

# ---- chat with context ----

reply = ollama.chat(
    model="llama3.1",
    messages=[
        {"role": "system", "content": f"Visual context from the camera:\n{context}"},
        {"role": "user", "content": "What's happening? Who's here?"},
    ],
)
print(reply["message"]["content"])
```

## OpenAI (GPT-4o vision + GPT-4o chat)

```python
import base64
from openai import OpenAI
from vemem import Vemem, Source

client = OpenAI()
vem = Vemem()

image_bytes = open("photo.jpg", "rb").read()
b64 = base64.b64encode(image_bytes).decode("ascii")

# observe + identify
observations = vem.observe(image_bytes)
candidates = vem.identify(image_bytes, k=3)

# VLM scene note via GPT-4o vision
if candidates:
    scene = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what this person is doing."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    ).choices[0].message.content or ""
    vem.remember(candidates[0].entity.id, scene, source=Source.VLM)

# recall + chat
context = ""
for c in vem.identify(image_bytes, k=3):
    snap = vem.recall(c.entity.id)
    context += f"- {c.entity.name}: {', '.join(f.content for f in snap.facts)}\n"

reply = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": f"Visual context:\n{context}"},
        {"role": "user", "content": "Who's here and what are they doing?"},
    ],
).choices[0].message.content
print(reply)
```

## Claude (Anthropic vision + chat)

```python
import base64
from anthropic import Anthropic
from vemem import Vemem, Source

client = Anthropic()
vem = Vemem()

image_bytes = open("photo.jpg", "rb").read()
b64 = base64.b64encode(image_bytes).decode("ascii")

observations = vem.observe(image_bytes)
candidates = vem.identify(image_bytes, k=3)

if candidates:
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": "Describe what this person is doing."},
            ],
        }],
    )
    scene = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    vem.remember(candidates[0].entity.id, scene, source=Source.VLM)

context = ""
for c in vem.identify(image_bytes, k=3):
    snap = vem.recall(c.entity.id)
    context += f"- {c.entity.name}: {', '.join(f.content for f in snap.facts)}\n"

reply = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    system=f"Visual context:\n{context}",
    messages=[{"role": "user", "content": "Who's here?"}],
)
print("".join(b.text for b in reply.content if getattr(b, "type", "") == "text"))
```

---

## Common patterns

### Correction flow (when identify is wrong)

```python
# identify() says this is Charlie at 0.71 confidence
candidates = vem.identify(image_bytes)

# User corrects: "That's actually Dana"
if user_says_thats_dana:
    obs_id = candidates[0].matched_observation_ids[0]
    vem.relabel(obs_id, "Dana")
    # A negative binding is emitted against Charlie —
    # the clusterer can never re-assign this face to Charlie.
```

### The 30-day undo window

Every reversible operation (label, relabel, merge, split, remember, restrict) sets `reversible_until = now + 30 days` in the EventLog. After that window, `undo()` refuses to reverse the event.

```python
# Undo the most recent reversible op
vem.undo()

# Undo a specific event by id (if you know it)
vem.undo(event_id=42)

# After 30 days, this raises OperationNotReversibleError.
# `forget()` is NEVER undoable regardless of the window.
```

If you need a longer window, change `DEFAULT_UNDO_WINDOW` in `vemem.core.ops` — it's a `timedelta(days=30)` constant. Making it shorter (e.g. 7 days) reduces EventLog scan cost; making it longer increases the rollback surface.

### Tying facts back to source images

vemem stores `source_uri` on each Observation but does **not** store or fetch image bytes (spec §3.1b). To tie a recalled fact back to its source image:

```python
snapshot = vem.recall(entity_id)

# Each fact was added by the caller (you), who also called observe().
# The entity's observations carry source_uri — that's the link.
bindings = vem.store.bindings_for_entity(entity_id)
for b in bindings:
    if b.valid_to is None:  # current positive binding
        obs = vem.store.get_observation(b.observation_id)
        print(f"  Seen in: {obs.source_uri} at {obs.detected_at}")
```

Facts don't carry which image they came from (they're statements about the entity, not about a specific image). If you need that link, store the source_uri in the fact content:

```python
vem.remember(entity.id, f"[from {source_uri}] wearing a blue shirt")
```

Or use an Event (which has `occurred_at`) instead of a Fact for time-anchored observations.

### Actor attribution for MCP

The MCP server defaults to `actor="mcp:unknown"` because MCP clients (including Claude Desktop) do not currently forward a user principal. Workarounds:

1. **Per-deployment override**: set `VEMEM_MCP_ACTOR=mcp:alice` in the environment before starting the server.
2. **Per-tool override**: every MCP tool accepts an optional `actor` parameter. Savvy clients can pass it; most won't.
3. **Accept the limitation**: for a personal single-user setup, `mcp:unknown` is fine — there's only one user. Multi-user MCP deployments need a proxy that injects the principal.

The CLI defaults to `cli:$USER`, which is more useful out of the box.
