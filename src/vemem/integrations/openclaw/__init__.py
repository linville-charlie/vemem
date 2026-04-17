"""openclaw integration: persistent HTTP sidecar backing the vemem-bridge plugin.

The :mod:`~vemem.integrations.openclaw.sidecar` module runs a small HTTP
server that openclaw's vemem-bridge plugin calls to describe each incoming
image attachment. The sidecar is what makes the integration "seamless" —
it holds InsightFace + LanceDB warm across requests so the agent never
waits on a cold start.

Run it directly via the ``vemem-openclaw-sidecar`` console script (installed
with vemem), or explicitly with ``python -m vemem.integrations.openclaw``.

The TypeScript plugin that drives this sidecar lives at
``integrations/openclaw/plugin/`` in the source tree. See
``integrations/openclaw/README.md`` for install instructions.
"""

from vemem.integrations.openclaw.sidecar import describe, main

__all__ = ["describe", "main"]
