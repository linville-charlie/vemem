# openclaw-plugin — seamless vemem integration for [openclaw](https://openclaw.dev)

This is a working reference for wiring vemem into an agent framework as an **automatic image-understanding provider** instead of a manually-called MCP tool. Drop it into `~/.openclaw/extensions/vemem-bridge/`, restart, and every image attachment is processed through vemem before the thinking LLM sees the conversation.

The problem it solves — and the design — are the same problem mem0/supermemory solve for conversational memory: agents don't reliably reach for a tool they *could* call. They do reliably respond to text that's already in the prompt. So instead of asking the agent to "remember to use vemem," this plugin makes vemem the input layer, transparently.

## What this plugin does

```
Discord image attachment
        │
        ▼
┌────────────────────────────────────────────────┐
│ openclaw media-understanding pipeline          │
│   • provider lookup in registry                │
│   • finds `vemem` (this plugin registered it)  │
│   • calls describeImage(buffer, fileName, …)   │
└────────────┬───────────────────────────────────┘
             ▼
┌────────────────────────────────────────────────┐
│ vemem-bridge plugin (index.ts)                 │
│   • writes buffer to /tmp/openclaw-vemem-*     │
│   • POSTs { path } → http://127.0.0.1:18790    │
└────────────┬───────────────────────────────────┘
             ▼
┌────────────────────────────────────────────────┐
│ vemem_http.py sidecar                          │
│   observe → identify → recall                  │
│   returns: "vemem: 1 face. Recognized: Charlie │
│            (conf 0.94). Known facts: [...]"    │
└────────────┬───────────────────────────────────┘
             ▼
Message reaches thinking LLM with a pre-built
"Description: ..." block. LLM never sees bytes.
```

Corrections (`label`, `merge`, `forget`, `undo`) stay on the vemem MCP server — those are cases where you *want* the agent to reason ("this is Alice, not Bob"). This plugin owns only the passive observe-and-describe path.

## Install

1. Clone vemem, install its deps: `git clone https://github.com/linville-charlie/vemem && cd vemem && uv sync`.
2. Copy this directory into your openclaw extensions folder:
   ```bash
   cp -r docs/examples/openclaw-plugin ~/.openclaw/extensions/vemem-bridge
   ```
3. Register the plugin in `~/.openclaw/openclaw.json`:
   ```json
   {
     "tools": {
       "media": {
         "image": {
           "enabled": true,
           "models": [
             { "provider": "vemem", "model": "vemem", "type": "provider" }
           ]
         }
       }
     },
     "plugins": {
       "allow": ["vemem-bridge"],
       "entries": {
         "vemem-bridge": {
           "enabled": true,
           "config": {
             "vememDir": "/absolute/path/to/vemem",
             "vememHome": "/absolute/path/to/lancedb-dir",
             "sidecarPort": 18790
           }
         }
       }
     }
   }
   ```
4. (Optional, recommended) Also register the vemem MCP server so the agent can call `label` / `merge` / `forget` when the user gives a correction:
   ```json
   "mcp": {
     "servers": {
       "vemem": {
         "command": "uv",
         "args": [
           "--directory", "/absolute/path/to/vemem",
           "run", "python", "-m", "vemem.mcp_server"
         ],
         "env": { "VEMEM_HOME": "/absolute/path/to/lancedb-dir" }
       }
     }
   }
   ```
5. Restart the gateway: `systemctl --user restart openclaw-gateway`.

On startup the plugin spawns `bridges/vemem_http.py` once and waits for `/health` to succeed (InsightFace cold-start is ~3s on CPU). If another instance of the sidecar is already running on that port, the new plugin instance reuses it and the duplicate sidecar exits cleanly.

## Verification

```bash
# sidecar reachable
curl -s -X POST http://127.0.0.1:18790/health -d '{}'

# label a face
uv run python bridges/openclaw_bridge.py label photo.jpg "Barack Obama" \
    --fact "44th President of the United States"

# describe via sidecar (what openclaw actually calls)
curl -s -X POST http://127.0.0.1:18790/describe \
    -H "content-type: application/json" \
    -d '{"path": "/absolute/path/to/photo.jpg"}'
# → {"text":"vemem: 1 face(s) detected.\nRecognized: Barack Obama (conf 1.00). Known facts: [...]"}
```

## Files

| File | Purpose |
|---|---|
| `index.ts` | The plugin. Spawns the sidecar, registers vemem as a media-understanding provider, proxies image bytes over HTTP. |
| `package.json` | openclaw extension descriptor (`main: "index.ts"`). |
| `openclaw.plugin.json` | Plugin manifest + JSON Schema for the config you set in `openclaw.json`. |

## Caveats

- **CPU-only InsightFace** on a 4-core/8GB machine adds ~400–800 ms per image describe after warmup. Acceptable for chat; not real-time.
- **Group photos are under-served** today — see the upstream issue on per-bbox cropping. The plugin returns "N faces, all unrecognized" for group shots; solo portraits work well (observed confidences 0.76–1.00 across 4 varied Obama photos with one training shot).
- **Port 18790** is a fixed default. If you need to change it, set both `config.sidecarPort` in the plugin entry AND ensure `VEMEM_HTTP_PORT` agrees if you launch the sidecar manually for debugging.

## Why this is the right seam

openclaw's plugin API exposes `api.registerMediaUnderstandingProvider(...)`. When you register one, the host's image pipeline calls YOUR `describeImage(buffer, fileName, …)` instead of the default. The returned text becomes the `Description:` block the LLM sees — identical to what Gemini/OpenAI vision models would have produced. From the agent's perspective nothing has changed except that the description is now grounded in a persistent identity store. No SOUL.md nudging, no tool-call dance, no base64 piping through exec. The agent doesn't need to learn to use vemem; vemem is just how this host describes images.
