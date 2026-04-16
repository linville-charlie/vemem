# Using the vemem MCP server

The vemem MCP server exposes the core identity + knowledge ops as
[Model Context Protocol](https://modelcontextprotocol.io) tools, so any
MCP-capable client — Claude Desktop, Cursor, custom agents — can read and
write a local vemem store.

## Running it

```bash
uv sync                                # install deps, including `mcp`
uv run python -m vemem.mcp_server      # stdio transport (default)
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `VEMEM_HOME` | `~/.vemem` | Where the LanceDB dataset lives |
| `VEMEM_ENCODER` | `insightface` | Which encoder to load (only `insightface` is wired in v0) |
| `VEMEM_MCP_TEST_MODE` | unset | Set to `1` to swap real backends for in-memory stubs (test use only) |

On first start the server downloads ~200MB of InsightFace weights into
`~/.insightface/models/buffalo_l/`. If the download fails, the server still
starts — the non-image tools (`label`, `recall`, `merge`, `undo`, …) keep
working, and the image-dependent tools return a clear error pointing to the
weight-download command.

## Wire it to Claude Desktop

See `docs/examples/claude_desktop_config.json` for a copy-paste starting
point. Edit `mcpServers.vemem.args` to point `--directory` at your local
clone of this repo, and `env.VEMEM_HOME` at wherever you want the data
stored.

On macOS, the config file lives at
`~/Library/Application Support/Claude/claude_desktop_config.json`.

## Tool surface

All ops from `docs/spec/identity-semantics.md` §4 are exposed.

### Core identity

| Tool | What it does |
|---|---|
| `observe_image` | Detect + persist observations from a base64-encoded image. Returns observation ids + bboxes. Idempotent. |
| `identify_image` | Ranked `Candidate` entities per detected face. No state mutation. |
| `identify_by_name` | Resolve an entity by name or id; returns a recall snapshot. |

### State changes

| Tool | What it does |
|---|---|
| `label` | Commit positive binding: "these observations are this entity." Creates the entity if new. |
| `relabel` | Move one observation to a different entity (plus a negative binding against the old one). |
| `merge` | Fold losers into a winner — facts migrate with provenance. Rejects modality/kind mismatches. |
| `split` | Break one entity into N; emits cross-wise negatives so the clusterer can't re-merge. |
| `forget` | Hard-delete everything tied to an entity + prune LanceDB history (GDPR Art. 17). NOT reversible. |
| `restrict` / `unrestrict` | GDPR Art. 18: stop using an entity for inference without deleting it. |

### Knowledge

| Tool | What it does |
|---|---|
| `remember` | Attach a free-text fact to an entity. Bi-temporal — stays valid until retracted. |
| `recall` | Return an entity plus its active facts. |

### Audit

| Tool | What it does |
|---|---|
| `undo` | Reverse a prior reversible op (default: most recent by actor). |
| `export` | GDPR Art. 20 data portability dump for one entity. |

## Actor attribution

Every write op accepts an optional `actor` argument (default `"mcp:unknown"`).
The event log records `actor` on every row so you can trace who did what —
especially useful when multiple MCP clients and a local CLI all share a
store. Clients that identify their principal can override per-call.

## End-to-end example (JSON-RPC)

A minimal round-trip, with base64-encoded image bytes elided:

```json
// → observe_image
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{
  "name":"observe_image","arguments":{"image_base64":"..."}}}

// ← {"observations":[{"id":"obs_..." ,"bbox":[x,y,w,h],...}]}

// → label
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
  "name":"label","arguments":{
    "observation_ids":["obs_..."],"entity_name_or_id":"Charlie",
    "actor":"user:alice"}}}

// ← {"id":"ent_...","name":"Charlie","status":"active",...}

// → remember
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
  "name":"remember","arguments":{
    "entity_id":"ent_...","content":"runs marathons","actor":"user:alice"}}}

// → identify_image (later, on a new photo)
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{
  "name":"identify_image","arguments":{"image_base64":"..."}}}

// ← {"detections":[{"bbox":[...],
//     "candidates":[{"entity":{"name":"Charlie",...},"confidence":0.94,...}]}]}
```
