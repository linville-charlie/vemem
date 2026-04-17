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

## Prerequisites

Before you install the plugin:

| Requirement | Why | How to check |
|---|---|---|
| openclaw running | This is a plugin for it | `systemctl --user status openclaw-gateway` (Linux) / `launchctl list \| grep openclaw` (macOS) — or however you normally run it |
| Python **3.12** or **3.13** | vemem's `requires-python`; 3.14 is blocked by a LanceDB segfault | `python3 --version` |
| `uv` on `PATH` | the plugin spawns `uv --directory ... run python bridges/vemem_http.py` | `which uv` — install via [astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) if missing |
| A C/C++ toolchain | `insightface` builds a small native extension during `uv sync` | Linux: `sudo apt install build-essential python3-dev` · macOS: `xcode-select --install` · Windows: install the [MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| ~300 MB free disk | InsightFace's `buffalo_l` weights (~200 MB) auto-download on first describe; LanceDB store grows with your gallery | — |

If you want Ella (or your agent) to also be able to *correct* identities (label new faces, merge duplicates, forget), you'll additionally need the vemem MCP server — included in the same repo, no extra install.

## Install

1. **Clone and install vemem.** Pick a path you can refer to later as `$VEMEM_DIR`:
   ```bash
   git clone https://github.com/linville-charlie/vemem "$HOME/code/vemem"
   cd "$HOME/code/vemem"
   uv sync                              # installs deps + builds insightface (takes ~60s)
   uv run pytest -q                     # optional smoke test (should print "266 passed")
   ```
   On first real use `insightface` downloads its detector/encoder weights into `~/.insightface/` (~200 MB). Network required for that one-time step.

2. **Pick a data directory** (`$VEMEM_HOME`). This is where LanceDB will store your gallery. Any writable absolute path works — e.g. `$HOME/.vemem` or a per-agent subdirectory like `~/.openclaw/memory/vemem`. Keep it separate from `$VEMEM_DIR` (the code).

3. **Drop the plugin into your openclaw extensions folder.** The destination is the directory you copy to; the plugin id is read from `openclaw.plugin.json`:
   ```bash
   cp -r "$VEMEM_DIR/docs/examples/openclaw-plugin" \
         "$HOME/.openclaw/extensions/vemem-bridge"
   ```

4. **Register the plugin and pick vemem as the image-understanding provider.** Merge the blocks below into your existing `~/.openclaw/openclaw.json` (don't replace the whole file — add under the matching top-level keys).
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
             "vememDir":  "/absolute/path/to/$VEMEM_DIR",
             "vememHome": "/absolute/path/to/$VEMEM_HOME",
             "sidecarPort": 18790
           }
         }
       }
     }
   }
   ```
   `vememDir` and `vememHome` are **required** — the plugin refuses to start if they are missing, rather than silently pointing at the author's home directory. `sidecarPort` defaults to 18790; change only if you have a conflict.

5. **(Recommended) Also wire up the vemem MCP server** so the agent can call `label` / `merge` / `forget` / `undo` when the user gives a correction — this is where agent reasoning actually helps. Add under `"mcp.servers"` in the same `openclaw.json`:
   ```json
   "mcp": {
     "servers": {
       "vemem": {
         "command": "uv",
         "args": [
           "--directory", "/absolute/path/to/$VEMEM_DIR",
           "run", "python", "-m", "vemem.mcp_server"
         ],
         "env": { "VEMEM_HOME": "/absolute/path/to/$VEMEM_HOME" }
       }
     }
   }
   ```

6. **Restart the gateway.** Use whichever supervisor is running it:
   - Linux (systemd user unit): `systemctl --user restart openclaw-gateway`
   - Launched manually: stop the old process, start it again
   - macOS (launchd) / Windows: use the openclaw launcher you use today

   Watch the gateway log — on Linux the default is `/tmp/openclaw/openclaw-YYYY-MM-DD.log`. You want to see (within ~5 seconds):
   ```
   [vemem-bridge] spawning sidecar: uv --directory ... run python bridges/vemem_http.py
   [vemem-bridge] registered media-understanding provider "vemem" (capabilities: image)
   [vemem-bridge] sidecar: [vemem-http] INFO warm: detector=insightface/buffalo_l@0.7.3 encoder=...
   [vemem-bridge] sidecar: [vemem-http] INFO listening on http://127.0.0.1:18790
   [vemem-bridge] sidecar ready at http://127.0.0.1:18790
   ```
   If instead you see `missing required config.vememDir` or `sidecar failed to come up within 60s`, the plugin stopped cleanly — re-check step 4 or your `$VEMEM_DIR` path.

## Verification

After step 6, wait ~5 seconds for the sidecar to finish warming InsightFace, then:

```bash
# 1. sidecar is listening
curl -s -X POST http://127.0.0.1:18790/health -d '{}'
# → {"ok": true}

# 2. describe on a face you haven't labeled yet (use any portrait photo)
curl -s -X POST http://127.0.0.1:18790/describe \
    -H "content-type: application/json" \
    -d '{"path": "/absolute/path/to/photo.jpg"}'
# → {"text": "vemem: 1 face(s) detected.\nUnrecognized faces: 1. ..."}

# 3. label it using the included CLI
cd "$VEMEM_DIR"
VEMEM_HOME=/absolute/path/to/$VEMEM_HOME \
  uv run python bridges/openclaw_bridge.py label /absolute/path/to/photo.jpg "Alice" \
  --fact "the user"

# 4. re-describe — should now be recognized
curl -s -X POST http://127.0.0.1:18790/describe \
    -H "content-type: application/json" \
    -d '{"path": "/absolute/path/to/photo.jpg"}'
# → {"text": "vemem: 1 face(s) detected.\nRecognized: Alice (conf 1.00). Known facts: [the user]"}
```

If step 4 reports `"Unrecognized"` after labeling, the sidecar is pointed at a different `VEMEM_HOME` than the CLI — re-check paths.

**Send an image on your normal chat surface** (Discord, Telegram, whatever you have wired up). The agent's response should reference what vemem reported (`"that's Alice — the user"`) rather than describing the scene in free-form. If the agent still describes pixel-level details ("dark blue shirt, outdoor lighting..."), something else is still handling media-understanding — grep the gateway log for `gemini` / `openai` / `anthropic` describe calls and confirm `tools.media.image.models` is set as above.

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
