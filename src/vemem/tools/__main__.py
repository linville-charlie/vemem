"""Allow ``python -m vemem.tools`` to emit the tool JSON.

Equivalent to ``python -m vemem.tools.export`` — kept as a convenience so
callers don't have to remember the submodule path.
"""

from __future__ import annotations

from vemem.tools.export import main

if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    raise SystemExit(main())
