# Using the OpenAI-Compatible Tool Schemas

The `vemem` MCP server is one way to call the library from an LLM. The other
way — for any function-calling LLM that doesn't speak MCP — is to paste the
library's OpenAI-compatible tool schemas into your prompt and wire each tool
call back to the library yourself.

The two surfaces describe the **same** operations. The MCP server is the
"live" stateful surface; these JSON schemas are the "stateless
paste-into-your-prompt" surface. If they ever drift, that is a bug.

## Getting the schemas

Three equivalent ways to get the list of tool schemas:

```bash
# CLI — writes canonical JSON to stdout
uv run python -m vemem.tools.export > tools.json

# Or with an explicit output path
uv run python -m vemem.tools.export --output tools.json

# Python
python -c "import json; from vemem.tools import all_tools; print(json.dumps(all_tools(), indent=2))"
```

Inside Python:

```python
from vemem.tools import all_tools, schema_for, TOOL_NAMES

tools = all_tools()           # list[dict] — all 14 tools
label = schema_for("label")   # dict — one tool by name
print(TOOL_NAMES)             # ("observe_image", "identify_image", ...)
```

Every schema follows the OpenAI function-calling format:

```json
{
  "type": "function",
  "function": {
    "name": "label",
    "description": "Commit a user-authoritative positive binding...",
    "parameters": {
      "type": "object",
      "properties": { "...": "..." },
      "required": ["observation_ids", "entity_name_or_id"]
    }
  }
}
```

## The 14 tools

| Tool | Purpose |
|---|---|
| `observe_image` | Detect faces, persist one observation per detection |
| `identify_image` | Read-only recognition against the gallery |
| `identify_by_name` | Resolve entity by name/id and return its recall snapshot |
| `label` | Bind observations to an entity (creates it if new) |
| `relabel` | Move one observation to a different entity (+ negative binding) |
| `merge` | Fold loser entities into a winner |
| `split` | Break one entity into N with cross-wise negative bindings |
| `forget` | GDPR Art. 17 hard delete (not undoable) |
| `restrict` / `unrestrict` | GDPR Art. 18 pause without deleting |
| `remember` / `recall` | Bi-temporal facts attached to an entity |
| `undo` | Reverse a reversible op within its window |
| `export` | GDPR Art. 20 portability dump for one entity |

Image inputs (`observe_image`, `identify_image`) take **base64-encoded bytes**
in the `image_base64` parameter. The schema tags this field with
`format: "byte"` and `contentEncoding: "base64"` per OpenAPI 3.0 convention.

## Using with OpenAI (Chat Completions)

```python
import base64, json
from pathlib import Path
from openai import OpenAI
from vemem.tools import all_tools

client = OpenAI()
tools = all_tools()

image_b64 = base64.b64encode(Path("team_photo.jpg").read_bytes()).decode()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    tools=tools,
    messages=[
        {"role": "system", "content": "You help the user tag and recall people."},
        {"role": "user", "content": "Find the people in this image."},
    ],
    # OpenAI can't stream image bytes to tool calls — pass the handle via a
    # previous turn or your own scratchpad. Here we just show the plumbing:
    tool_choice="auto",
)

for call in response.choices[0].message.tool_calls or []:
    name = call.function.name
    args = json.loads(call.function.arguments)
    # Dispatch: call the corresponding vemem op with `args` and return the
    # result to the next turn as a tool message.
    ...
```

## Using with Anthropic (tool use)

Anthropic's tool-use format is compatible but uses `input_schema` instead of
`parameters`, and is not wrapped in a `type: function` envelope. Translate
once when you load:

```python
from anthropic import Anthropic
from vemem.tools import all_tools

anthropic_tools = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in all_tools()
]

client = Anthropic()
resp = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    tools=anthropic_tools,
    messages=[{"role": "user", "content": "Recall everything we know about Charlie."}],
)
# Iterate resp.content for tool_use blocks; dispatch the same way as OpenAI.
```

## Using with Ollama (function calling)

Ollama's chat API accepts the same `tools` shape as OpenAI for models that
support function calling (Llama 3.1, Mistral Nemo, Qwen, etc.):

```python
import ollama
from vemem.tools import all_tools

resp = ollama.chat(
    model="llama3.1",
    tools=all_tools(),  # same shape as OpenAI's
    messages=[{"role": "user", "content": "Who's in this photo?"}],
)
for call in resp["message"].get("tool_calls", []):
    name = call["function"]["name"]
    args = call["function"]["arguments"]  # already a dict on Ollama
    ...
```

## Dispatching tool calls to the library

The schemas describe the shapes. Dispatch can be as thin as:

```python
from vemem.core import ops
from vemem.core.enums import Source
from vemem.core.protocols import Clock, Store
# ...your store and clock setup...

def dispatch(name: str, args: dict, *, store: Store, clock: Clock, actor: str):
    if name == "label":
        return ops.label(store, args["observation_ids"], args["entity_name_or_id"],
                         clock=clock, actor=actor)
    if name == "remember":
        return ops.remember(store, entity_id=args["entity_id"],
                            content=args["content"],
                            source=Source(args.get("source", "user")),
                            clock=clock, actor=actor)
    # ... one branch per tool ...
```

Or, if you want the exact same implementation the MCP server uses, reach into
the MCP tool handlers — they already accept kwargs matching these schemas:

```python
from vemem.mcp_server.tools import ServerContext, label_tool, remember_tool

ctx = ServerContext(store=..., clock=..., encoder=..., detector=...)
result_dict = label_tool(ctx,
                         observation_ids=args["observation_ids"],
                         entity_name_or_id=args["entity_name_or_id"],
                         actor=actor)
```

## Keeping the schemas current

The committed `tests/tools/snapshots/tools.json` is the canonical export
shape, guarded by a snapshot test. When you change a tool's parameters:

```bash
uv run python -m vemem.tools.export > tests/tools/snapshots/tools.json
uv run pytest tests/tools/
```

The snapshot test will pass; review the diff before committing. Any drift
from this file in the wild means an older caller is looking at an older
schema — no silent breakage.
