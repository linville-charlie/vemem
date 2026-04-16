"""Tests for UUID v7 ID generation.

UUID v7 is time-ordered — lexicographic sort matches chronological order. This
property is load-bearing for append-only workloads (EventLog ordering, binding
supersede chains).
"""

import time
import uuid as stdlib_uuid

from vemem.core.ids import new_id


def test_new_id_returns_valid_uuid_string() -> None:
    out = new_id()
    assert isinstance(out, str)
    parsed = stdlib_uuid.UUID(out)
    assert parsed.version == 7


def test_new_id_is_unique() -> None:
    ids = {new_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_new_id_is_time_ordered() -> None:
    earlier = new_id()
    time.sleep(0.002)
    later = new_id()
    assert earlier < later


def test_new_id_version_bits_are_7() -> None:
    out = new_id()
    parsed = stdlib_uuid.UUID(out)
    assert parsed.version == 7
    assert parsed.variant == stdlib_uuid.RFC_4122
