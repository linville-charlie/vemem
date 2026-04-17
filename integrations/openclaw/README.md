# vemem ⇄ [openclaw](https://openclaw.dev) — first-party integration

vemem is supported as an automatic image-understanding provider for openclaw. After installing, every image attachment your agent receives is transparently described through vemem — face detection + persistent identity + recalled facts — before the thinking LLM sees the conversation. No agent-side tool calls, no prompting, no SOUL.md nudges. The agent just sees text like `"vemem: 1 face(s) detected. Recognized: Charlie (conf 0.94). Known facts: [training for Boston marathon]"` in place of whatever the default vision model would have produced.

The same seam mem0/supermemory use for conversational memory, applied to visual identity.

> **Platform status:** end-to-end verified on **Ubuntu 24.04** (kernel 6.8, Python 3.13, 4 CPU cores, CPU-only InsightFace). **macOS and Windows install steps are written from the docs of the underlying tools but have not been run by the authors.** If you hit a platform-specific issue, please open an issue at https://github.com/linville-charlie/vemem/issues — we'll fold the fix back in.

## Architecture

```
Image attachment arrives at openclaw
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
│ plugin/index.ts                                │
│   • writes buffer to /tmp/openclaw-vemem-*     │
│   • POSTs { path } → http://127.0.0.1:18790    │
└────────────┬───────────────────────────────────┘
             ▼
┌────────────────────────────────────────────────┐
│ vemem-openclaw-sidecar (Python)                │
│   (vemem.integrations.openclaw.sidecar)        │
│   observe → identify → recall                  │
│   returns "vemem: Charlie (conf 0.94)…"        │
└────────────┬───────────────────────────────────┘
             ▼
Message reaches thinking LLM with a pre-built
"Description: ..." block. LLM never sees bytes.
```

Corrections (`label`, `merge`, `forget`, `undo`) are exposed separately via the vemem MCP server. Those are cases where you *want* the agent to reason ("this is Alice, not Bob"). This plugin owns only the passive observe-and-describe path.

## Contents of this directory

| Path | What it is |
|---|---|
| `README.md` | This file — the canonical install + usage guide. |
| `plugin/index.ts` | The openclaw plugin. Registers vemem as a media-understanding provider and manages the sidecar. |
| `plugin/package.json` | openclaw extension descriptor (`main: "index.ts"`). |
| `plugin/openclaw.plugin.json` | Plugin manifest + JSON Schema config surface. |

The Python sidecar ships in the vemem Python package at `src/vemem/integrations/openclaw/sidecar.py`, exposed via the `vemem-openclaw-sidecar` console script. The plugin launches it automatically.

## Prerequisites

| Requirement | Why | How to check |
|---|---|---|
| openclaw running | This is a plugin for it | Linux: `systemctl --user status openclaw-gateway` · macOS: `launchctl list \| grep openclaw` · or however you normally run it |
| Python **3.12** or **3.13** | vemem's `requires-python`; 3.14 is blocked by a LanceDB segfault | `python3 --version` |
| `vemem` installed | Ships the sidecar console script and all Python deps | `vemem-openclaw-sidecar --help` should exit 0 after install (see next section) |
| A C/C++ toolchain | `insightface` builds a small native extension at install time | Linux: `sudo apt install build-essential python3-dev` · macOS: `xcode-select --install` · Windows: [MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| ~300 MB free disk | InsightFace's `buffalo_l` weights (~200 MB) auto-download on first describe; LanceDB store grows with your gallery | — |

## Install — two modes

### Mode A: tool install (recommended — what the README verifies)

Install vemem into an isolated environment whose console scripts land on your user `PATH`:

```bash
uv tool install git+https://github.com/linville-charlie/vemem
# or: pipx install git+https://github.com/linville-charlie/vemem
# or (once vemem is on PyPI): uv tool install vemem / pipx install vemem
```

After this, three scripts are available:

| Script | What it is |
|---|---|
| `vm` | The vemem CLI (`vm observe`, `vm label`, `vm remember`, …) |
| `vemem-openclaw-sidecar` | The HTTP sidecar this plugin spawns |
| `vemem-mcp-server` | The MCP server exposing corrections (`label`, `merge`, `forget`, `undo`) |

Verify with `command -v vemem-openclaw-sidecar` — it should print something under `~/.local/bin/` (or your tool-install bin path).

### Mode B: dev (against a repo checkout)

Useful while vemem is pre-alpha and you want to track `main`:

```bash
git clone https://github.com/linville-charlie/vemem "$HOME/code/vemem"
cd "$HOME/code/vemem"
uv sync
```

Remember the absolute path — you'll pass it as `vememDir` below.

## Wire it up

1. **Drop the plugin into your openclaw extensions folder.** Only the `plugin/` subdirectory is needed:
   ```bash
   cp -r "$(python -c 'import vemem, pathlib; print(pathlib.Path(vemem.__file__).parent.parent.parent)')/integrations/openclaw/plugin" \
         "$HOME/.openclaw/extensions/vemem-bridge"
   ```
   Or if you're in Mode B with a local checkout:
   ```bash
   cp -r "$HOME/code/vemem/integrations/openclaw/plugin" \
         "$HOME/.openclaw/extensions/vemem-bridge"
   ```

2. **Pick vemem as the image-understanding provider.** Merge into your existing `~/.openclaw/openclaw.json` (don't replace the whole file — add under matching keys):
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
           "config": {}
         }
       }
     }
   }
   ```
   Mode A (tool-installed vemem): **that's it** — empty config works. The plugin runs `vemem-openclaw-sidecar` off `PATH` and writes LanceDB data to `~/.vemem`.

   Mode B (repo checkout) or custom data directory:
   ```json
   "vemem-bridge": {
     "enabled": true,
     "config": {
       "vememDir":  "/home/you/code/vemem",
       "vememHome": "/home/you/.openclaw/memory/vemem"
     }
   }
   ```
   `vememDir` switches the plugin to running `uv --directory <path> run vemem-openclaw-sidecar`. `vememHome` overrides the LanceDB location. Both are optional — omit either one to use defaults.

3. **(Recommended) Also register the vemem MCP server** so the agent can call `label` / `merge` / `forget` / `undo` when the user gives a correction. This is where agent reasoning actually helps. Add under `"mcp.servers"`:
   ```json
   "mcp": {
     "servers": {
       "vemem": {
         "command": "vemem-mcp-server",
         "args": [],
         "env": { "VEMEM_HOME": "/home/you/.openclaw/memory/vemem" }
       }
     }
   }
   ```
   (Mode B: replace with `"command": "uv", "args": ["--directory", "/home/you/code/vemem", "run", "vemem-mcp-server"]`.)

4. **Restart the gateway.** Use whichever supervisor runs it:
   - Linux (systemd user unit): `systemctl --user restart openclaw-gateway`
   - macOS (launchd) / Windows: use the openclaw launcher you use today

   Watch the log — on Linux the default is `/tmp/openclaw/openclaw-YYYY-MM-DD.log`. Within ~5 seconds you should see:
   ```
   [vemem-bridge] spawning sidecar: vemem-openclaw-sidecar
   [vemem-bridge] registered media-understanding provider "vemem" (capabilities: image)
   [vemem-bridge] sidecar: [vemem-http] INFO warm: detector=insightface/buffalo_l@0.7.3 ...
   [vemem-bridge] sidecar: [vemem-http] INFO listening on http://127.0.0.1:18790
   [vemem-bridge] sidecar ready at http://127.0.0.1:18790
   ```
   If instead you see `failed to spawn sidecar` the `vemem-openclaw-sidecar` command isn't on `PATH` — re-run the Mode A install (`uv tool install git+https://github.com/linville-charlie/vemem` or `pipx install …`), or set `config.vememDir` to fall back to Mode B.

## Verification

After step 4, wait ~5 seconds for the sidecar to finish warming InsightFace, then:

```bash
# 1. sidecar is listening
curl -s -X POST http://127.0.0.1:18790/health -d '{}'
# → {"ok": true}

# 2. describe an unlabeled face
curl -s -X POST http://127.0.0.1:18790/describe \
    -H "content-type: application/json" \
    -d '{"path": "/absolute/path/to/photo.jpg"}'
# → {"text": "vemem: 1 face(s) detected.\nUnrecognized faces: 1. ..."}

# 3. label it using the vemem CLI (pip-install put `vm` on PATH too).
#    label binds obs → entity; remember attaches a fact.
VEMEM_HOME=/home/you/.openclaw/memory/vemem \
  vm observe /absolute/path/to/photo.jpg               # prints observation id
VEMEM_HOME=/home/you/.openclaw/memory/vemem \
  vm label obs_... --name "Alice"                      # prints entity id
VEMEM_HOME=/home/you/.openclaw/memory/vemem \
  vm remember ent_... --fact "the user"

# 4. re-describe — should now be recognized
curl -s -X POST http://127.0.0.1:18790/describe \
    -H "content-type: application/json" \
    -d '{"path": "/absolute/path/to/photo.jpg"}'
# → {"text": "vemem: 1 face(s) detected.\nRecognized: Alice (conf 1.00). Known facts: [the user]"}
```

If step 4 still reports `"Unrecognized"`, the sidecar is pointed at a different `VEMEM_HOME` than the CLI — re-check paths (both default to `~/.vemem` when unset).

**Send an image on your normal chat surface** (Discord, Telegram, Slack — wherever you have openclaw wired). The agent's response should reference what vemem reported ("that's Alice — the user") rather than free-form describing the pixels. If the agent still does generic scene description, something else is handling media-understanding — grep the gateway log for `gemini` / `openai` / `anthropic` describe calls and confirm `tools.media.image.models` is set as above.

## Caveats

- **CPU-only InsightFace** on a 4-core/8GB machine adds 400–800 ms per describe after warmup. Acceptable for chat; not real-time.
- **Group photos are under-served** today — see [issue #2](https://github.com/linville-charlie/vemem/issues/2). Solo portraits work well (observed confidences 0.76–1.00 across 4 varied Obama photos trained on one reference shot).
- **Port 18790** is a fixed default. Change via `config.sidecarPort` if you need; both the plugin and sidecar will pick it up from env.
- **Plugin `allow` list.** openclaw logs a provenance warning for any plugin not in `plugins.allow`. Adding `"vemem-bridge"` silences it and pins trust.

## Why this seam

openclaw's plugin API exposes `api.registerMediaUnderstandingProvider(...)`. When you register one, the host's image pipeline calls YOUR `describeImage(buffer, fileName, …)` instead of the default. The returned text becomes the `Description:` block the LLM sees — identical to what Gemini/OpenAI vision models would have produced. From the agent's perspective nothing has changed except that the description is now grounded in a persistent identity store. No base64 piping through exec, no tool-call dance, no prompt engineering. The agent doesn't need to learn to use vemem; vemem is just how this host describes images.
