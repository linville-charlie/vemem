"""Tests that the core Protocols can be satisfied by a trivial implementation.

Runtime Protocol satisfaction is weak (Python doesn't enforce method signatures
at runtime), so the real value here is (a) the Protocol types import, (b) a
hand-written satisfying class type-checks, (c) the Clock Protocol works with a
frozen-time test double.
"""

from datetime import UTC, datetime

from vemem.core.protocols import Clock


def test_clock_protocol_satisfiable_by_fixed_clock() -> None:
    class FixedClock:
        def __init__(self, at: datetime) -> None:
            self._at = at

        def now(self) -> datetime:
            return self._at

    fixed = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)
    clock: Clock = FixedClock(fixed)
    assert clock.now() == fixed


def test_protocols_importable() -> None:
    from vemem.core.protocols import Detector, Encoder, Store  # noqa: F401
