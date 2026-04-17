/**
 * vemem-bridge — openclaw plugin
 *
 * Replaces the default image-description provider with vemem. On register:
 *   1. spawn a long-lived Python sidecar (loads InsightFace + LanceDB once),
 *   2. wait for /health to come up,
 *   3. register `vemem` as a media-understanding provider.
 *
 * Per incoming image attachment, openclaw calls describeImage(buffer, ...).
 * We write the buffer to a temp file and POST the path to the sidecar, which
 * runs detect → identify → recall and returns ready-to-inject text like:
 *
 *   vemem: 1 face(s) detected.
 *   Recognized: Charlie (conf 0.94). Known facts: [training for Boston]
 *
 * The thinking LLM never receives image bytes — only this text.
 *
 * Corrections (label / merge / forget / undo) stay on the vemem MCP server.
 * This plugin is strictly the auto-describe path.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { spawn, type ChildProcess } from "node:child_process";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir, homedir } from "node:os";
import { join } from "node:path";

type PluginConfig = {
  vememDir?: string;
  vememHome?: string;
  sidecarHost?: string;
  sidecarPort?: number;
  warmupTimeoutSeconds?: number;
  requestTimeoutSeconds?: number;
};

const DEFAULTS = {
  vememDir: "/home/ella/.openclaw/workspace/vemem",
  vememHome: "/home/ella/.openclaw/memory/vemem",
  sidecarHost: "127.0.0.1",
  sidecarPort: 18790,
  warmupTimeoutSeconds: 60,
  requestTimeoutSeconds: 30,
};

function pickConfig(raw: unknown): Required<PluginConfig> {
  const cfg = (raw ?? {}) as PluginConfig;
  return {
    vememDir: cfg.vememDir || DEFAULTS.vememDir,
    vememHome: cfg.vememHome || DEFAULTS.vememHome,
    sidecarHost: cfg.sidecarHost || DEFAULTS.sidecarHost,
    sidecarPort: cfg.sidecarPort || DEFAULTS.sidecarPort,
    warmupTimeoutSeconds: cfg.warmupTimeoutSeconds || DEFAULTS.warmupTimeoutSeconds,
    requestTimeoutSeconds: cfg.requestTimeoutSeconds || DEFAULTS.requestTimeoutSeconds,
  };
}

async function sleep(ms: number): Promise<void> {
  await new Promise((r) => setTimeout(r, ms));
}

async function waitForHealth(
  url: string,
  timeoutSeconds: number,
  log: (msg: string) => void,
): Promise<boolean> {
  const deadline = Date.now() + timeoutSeconds * 1000;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url, { method: "POST", body: "{}" });
      if (res.ok) {
        const body = (await res.json()) as { ok?: boolean };
        if (body.ok) return true;
      }
    } catch {
      // sidecar not up yet
    }
    await sleep(500);
  }
  log(`sidecar failed to come up within ${timeoutSeconds}s`);
  return false;
}

const plugin = {
  id: "vemem-bridge",
  register(api: OpenClawPluginApi) {
    const cfg = pickConfig(api.pluginConfig);
    const log = (msg: string) => api.logger.info(`[vemem-bridge] ${msg}`);
    const warn = (msg: string) => api.logger.warn(`[vemem-bridge] ${msg}`);
    const error = (msg: string) => api.logger.error(`[vemem-bridge] ${msg}`);

    const baseUrl = `http://${cfg.sidecarHost}:${cfg.sidecarPort}`;
    let sidecar: ChildProcess | null = null;

    // The plugin can be loaded by multiple subsystems (gateway + plugins runtime)
    // in the same process — don't race to bind the port. Probe /health first and
    // reuse an already-running sidecar.
    const maybeSpawn = async () => {
      try {
        const probe = await fetch(`${baseUrl}/health`, { method: "POST", body: "{}" });
        if (probe.ok) {
          log(`sidecar already running at ${baseUrl}, reusing`);
          return;
        }
      } catch {
        // not up; spawn one below
      }
      log(`spawning sidecar: uv --directory ${cfg.vememDir} run python bridges/vemem_http.py`);
      sidecar = spawn(
        "uv",
        ["--directory", cfg.vememDir, "run", "python", "bridges/vemem_http.py"],
        {
          env: {
            ...process.env,
            HOME: process.env.HOME || homedir(),
            VEMEM_HOME: cfg.vememHome,
            VEMEM_HTTP_HOST: cfg.sidecarHost,
            VEMEM_HTTP_PORT: String(cfg.sidecarPort),
          },
          stdio: ["ignore", "pipe", "pipe"],
        },
      );
      sidecar.stderr?.on("data", (b: Buffer) => {
        for (const line of b.toString("utf8").split("\n")) {
          if (line.trim()) log(`sidecar: ${line.trim()}`);
        }
      });
      sidecar.on("exit", (code, signal) => {
        if (code !== 0) warn(`sidecar exited code=${code} signal=${signal}`);
        sidecar = null;
      });
    };
    void maybeSpawn();

    // Block providers from firing until /health returns OK — otherwise the
    // first image request races the InsightFace warmup and times out.
    let healthy = false;
    void waitForHealth(`${baseUrl}/health`, cfg.warmupTimeoutSeconds, warn).then((ok) => {
      healthy = ok;
      if (ok) log(`sidecar ready at ${baseUrl}`);
    });

    api.registerMediaUnderstandingProvider({
      id: "vemem",
      capabilities: ["image"],
      describeImage: async (req) => {
        if (!healthy) {
          return {
            text: "vemem: sidecar not ready yet, skipping auto-describe.",
            model: "vemem",
          };
        }
        const dir = await mkdtemp(join(tmpdir(), "openclaw-vemem-"));
        const imgPath = join(dir, req.fileName || "image.jpg");
        try {
          await writeFile(imgPath, req.buffer);
          const controller = new AbortController();
          const timer = setTimeout(
            () => controller.abort(),
            cfg.requestTimeoutSeconds * 1000,
          );
          try {
            const res = await fetch(`${baseUrl}/describe`, {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify({ path: imgPath }),
              signal: controller.signal,
            });
            if (!res.ok) {
              const body = await res.text();
              error(`describe failed http=${res.status} body=${body.slice(0, 200)}`);
              return { text: `vemem: error (http ${res.status}).`, model: "vemem" };
            }
            const body = (await res.json()) as { text?: string };
            return { text: body.text || "vemem: empty response.", model: "vemem" };
          } finally {
            clearTimeout(timer);
          }
        } catch (err) {
          error(`describe threw: ${err instanceof Error ? err.message : String(err)}`);
          return {
            text: "vemem: sidecar unreachable — ignoring image this turn.",
            model: "vemem",
          };
        } finally {
          await rm(dir, { recursive: true, force: true }).catch(() => {});
        }
      },
    });

    log(`registered media-understanding provider "vemem" (capabilities: image)`);

    // Lifecycle: kill sidecar cleanly on shutdown so ports don't leak.
    api.on("before_shutdown", () => {
      if (sidecar) {
        log("stopping sidecar");
        sidecar.kill("SIGTERM");
      }
    });
  },
};

export default plugin;
