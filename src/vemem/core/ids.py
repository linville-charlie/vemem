"""UUID v7 generation without external dependencies.

Why v7: lexicographic order matches chronological order, which matters for our
append-only event log and binding supersede chains. Python 3.14 ships
``uuid.uuid7`` in stdlib; we target 3.12+ so we implement it per RFC 9562.

The produced UUID is compatible with ``uuid.UUID`` — callers can parse the
return value with ``uuid.UUID(new_id())`` if they need the integer/bytes form.
"""

from __future__ import annotations

import secrets
import time
import uuid


def new_id() -> str:
    """Return a time-ordered UUID v7 as a string.

    Layout (RFC 9562 §5.7):
      - 48 bits Unix epoch milliseconds (big-endian)
      - 4 bits version (0b0111 = 7)
      - 12 bits random (rand_a)
      - 2 bits variant (0b10)
      - 62 bits random (rand_b)
    """
    ts_ms = time.time_ns() // 1_000_000
    ts_bytes = ts_ms.to_bytes(6, "big")
    rand_bytes = secrets.token_bytes(10)

    b = bytearray(ts_bytes + rand_bytes)
    b[6] = (b[6] & 0x0F) | 0x70
    b[8] = (b[8] & 0x3F) | 0x80

    return str(uuid.UUID(bytes=bytes(b)))
