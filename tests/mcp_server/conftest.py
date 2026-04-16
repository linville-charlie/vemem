"""Shared fixtures for MCP server tests.

Keeps the unit tests (``test_tools.py``) and the stdio integration test
(``test_server.py``) consistent — both build a ``ServerContext`` from the
same ingredients (FakeStore + StubEncoder + StubDetector + FixedClock).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from tests.mcp_server._test_backends import StubDetector, StubEncoder
from tests.support.fake_store import FakeStore
from vemem.mcp_server.tools import ServerContext


class FixedClock:
    """Monotonically-advancing test clock.

    ``now()`` returns the start time + one microsecond per call so that
    events ordered by ``(at, id)`` are distinguishable even when they happen
    within the same logical op.
    """

    def __init__(self, start: datetime | None = None) -> None:
        self._start = start or datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)
        self._tick = 0

    def now(self) -> datetime:
        self._tick += 1
        return self._start.replace(microsecond=self._tick)


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock()


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def stub_encoder() -> StubEncoder:
    return StubEncoder()


@pytest.fixture
def stub_detector() -> StubDetector:
    return StubDetector()


@pytest.fixture
def ctx(
    store: FakeStore,
    clock: FixedClock,
    stub_encoder: StubEncoder,
    stub_detector: StubDetector,
) -> ServerContext:
    """Build a fully-wired test context."""
    return ServerContext(
        store=store,
        clock=clock,
        encoder=stub_encoder,
        detector=stub_detector,
        encoder_error=None,
    )


@pytest.fixture
def ctx_without_encoder(store: FakeStore, clock: FixedClock) -> ServerContext:
    """Context simulating 'weights not installed' — encoder + detector are None."""
    return ServerContext(
        store=store,
        clock=clock,
        encoder=None,
        detector=None,
        encoder_error="insightface weights not found at ~/.insightface/models/",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip subprocess integration tests unless VEMEM_RUN_INTEGRATION=1.

    The stdio integration test spawns a real subprocess and is slower (and
    more brittle on CI) than the in-process handler tests. Default-off.
    """
    if os.environ.get("VEMEM_RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="integration tests; set VEMEM_RUN_INTEGRATION=1 to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
