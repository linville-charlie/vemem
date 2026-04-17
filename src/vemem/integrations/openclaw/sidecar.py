"""HTTP sidecar that backs the openclaw vemem-bridge plugin.

Long-lived process. Loads InsightFace + LanceDBStore once on startup so
``/describe`` responses are fast. Accepts a filesystem path (not base64)
to avoid tool-arg truncation in the host framework. Returns a ready-to-
inject text block summarizing what vemem sees (face count + recognized
entities + recalled facts).

The HTTP shape is intentionally neutral — any host language (TypeScript,
Go, Rust, …) can POST to ``/describe`` — but the first-party companion
plugin lives at ``integrations/openclaw/plugin/`` in this repo.

Run it three ways:

- **console script** (installed with vemem):  ``vemem-openclaw-sidecar``
- **as a module**:                            ``python -m vemem.integrations.openclaw``
- **directly from source**:                   ``python sidecar.py``

Env vars:
    VEMEM_HOME       — LanceDB path (default ~/.vemem)
    VEMEM_HTTP_PORT  — listen port (default 18790)
    VEMEM_HTTP_HOST  — bind address (default 127.0.0.1)

Endpoints:
    POST /describe  { path: "/abs/file.jpg" }  → { text: "vemem: ..." }
    POST /health    {}                          → { ok: true }
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from vemem.core import ops
from vemem.core.enums import Modality
from vemem.encoders.insightface_detector import InsightFaceDetector
from vemem.encoders.insightface_encoder import InsightFaceEncoder
from vemem.pipeline import observe_image
from vemem.storage.lancedb_store import LanceDBStore


class _SystemClock:
    def now(self) -> datetime:
        return datetime.now(tz=UTC)


log = logging.getLogger("vemem-http")
logging.basicConfig(
    level=logging.INFO, stream=sys.stderr, format="[vemem-http] %(levelname)s %(message)s"
)

VEMEM_HOME = os.environ.get("VEMEM_HOME", str(Path.home() / ".vemem"))
HOST = os.environ.get("VEMEM_HTTP_HOST", "127.0.0.1")
PORT = int(os.environ.get("VEMEM_HTTP_PORT", "18790"))

log.info("warming up: loading InsightFace + LanceDBStore at %s", VEMEM_HOME)
STORE = LanceDBStore(path=VEMEM_HOME)
DETECTOR = InsightFaceDetector()
ENCODER = InsightFaceEncoder()
CLOCK = _SystemClock()
log.info("warm: detector=%s encoder=%s", DETECTOR.id, ENCODER.id)


def _refresh_encoder_cache() -> None:
    """Re-read the encoders table so identify sees rows written by other processes.

    The CLI / MCP server run in separate processes and may register encoders
    (or add embeddings) we haven't seen in-memory. Cheap — one table read.
    """
    STORE._encoder_tables = STORE._load_encoder_tables()


def describe(image_path: str) -> str:
    """Run the vemem observe+read path and render a human-readable summary.

    Uses the full ``observe_image`` pipeline so evidence of every image the
    agent sees is persisted — even unlabeled ones — for later retroactive
    labeling. Then queries for matches against the stored gallery.
    """
    if not Path(image_path).exists():
        return f"vemem: image path not found: {image_path}"
    image_bytes = Path(image_path).read_bytes()

    _refresh_encoder_cache()

    try:
        observations = observe_image(
            STORE,
            image_bytes=image_bytes,
            detector=DETECTOR,
            encoder=ENCODER,
            clock=CLOCK,
            modality=Modality.FACE,
            source_uri=f"file://{image_path}",
        )
    except Exception as exc:  # never crash the describe path on a bad write
        log.exception("observe failed: %s", exc)
        observations = []

    if not observations:
        return "vemem: no faces detected in image."

    named: list[str] = []
    unknown = 0
    for _obs in observations:
        vector = ENCODER.embed(image_bytes)
        candidates = ops.identify(
            STORE, encoder_id=ENCODER.id, vector=vector, k=3, min_confidence=0.5
        )
        if not candidates:
            unknown += 1
            continue
        top = max(candidates, key=lambda c: c.confidence)
        snap = ops.recall(STORE, entity_id=top.entity.id)
        facts = "; ".join(f.content for f in snap.facts) if snap.facts else "no facts yet"
        named.append(f"{top.entity.name} (conf {top.confidence:.2f}). Known facts: [{facts}]")

    lines = [f"vemem: {len(observations)} face(s) detected."]
    if named:
        lines.append("Recognized: " + "; ".join(named))
    if unknown:
        lines.append(
            f"Unrecognized faces: {unknown}. To teach vemem, reply with a name and it "
            "can be labeled via the vemem MCP tools."
        )
    return "\n".join(lines)


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            parsed = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def do_POST(self) -> None:
        try:
            if self.path == "/health":
                self._json(200, {"ok": True})
                return
            if self.path == "/describe":
                body = self._read_json()
                path = body.get("path")
                if not path or not isinstance(path, str):
                    self._json(400, {"error": "missing string field: path"})
                    return
                text = describe(path)
                self._json(200, {"text": text})
                return
            self._json(404, {"error": "not found"})
        except Exception as exc:  # return error text, don't kill the process
            log.exception("handler error")
            self._json(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:  # route to our logger
        log.info("%s - %s", self.address_string(), fmt % args)


def self_test() -> int:
    """--test flag: run describe() on an image given via ``VEMEM_TEST_IMAGE``.

    Falls back to the first JPEG/PNG found in ``VEMEM_TEST_DIR`` (default cwd)
    so CI and local smoke tests can both run without special setup.
    """
    explicit = os.environ.get("VEMEM_TEST_IMAGE")
    if explicit:
        print(f"--- {Path(explicit).name} ---")
        print(describe(explicit))
        return 0
    search_dir = Path(os.environ.get("VEMEM_TEST_DIR", os.getcwd()))
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img in search_dir.rglob(ext):
            print(f"--- {img.name} ---")
            print(describe(str(img)))
            return 0
    print(f"No test images found under {search_dir}. Set VEMEM_TEST_IMAGE.")
    return 1


def main() -> None:
    if "--test" in sys.argv:
        sys.exit(self_test())
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as exc:
        if exc.errno == 98:  # address in use — another instance wins, exit clean
            log.info("port %d already in use, deferring to running instance", PORT)
            sys.exit(0)
        raise
    log.info("listening on http://%s:%d", HOST, PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
