"""Shared fixtures for CLI tests.

The CLI commands construct a :class:`vemem.cli.context.CliContext` on each
invocation via ``build_cli_context``. To exercise commands without real
LanceDB + InsightFace weights, the tests install a process-global test
context factory that hands the CLI a pre-built context wrapping a
``FakeStore`` + stub encoder/detector. This is the same pattern the MCP
server uses via ``VEMEM_MCP_TEST_MODE``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from tests.mcp_server._test_backends import StubDetector, StubEncoder
from tests.support.fake_store import FakeStore
from vemem.cli import context as cli_context
from vemem.cli.context import CliContext


class FixedClock:
    """Monotonically-advancing test clock (one microsecond per call)."""

    def __init__(self, start: datetime | None = None) -> None:
        self._start = start or datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)
        self._tick = 0

    def now(self) -> datetime:
        self._tick += 1
        return self._start.replace(microsecond=self._tick)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock()


@pytest.fixture
def cli_ctx(store: FakeStore, clock: FixedClock) -> CliContext:
    return CliContext(
        store=store,
        clock=clock,
        encoder=StubEncoder(),
        detector=StubDetector(),
        encoder_error=None,
        actor="cli:tester",
    )


@pytest.fixture
def cli_ctx_no_encoder(store: FakeStore, clock: FixedClock) -> CliContext:
    return CliContext(
        store=store,
        clock=clock,
        encoder=None,
        detector=None,
        encoder_error="insightface weights not found; run `uv run python -c ...`",
        actor="cli:tester",
    )


@pytest.fixture
def install_ctx(cli_ctx: CliContext) -> Iterator[CliContext]:
    """Install ``cli_ctx`` as the process-global context used by ``build_cli_context``."""
    cli_context.set_test_context(cli_ctx)
    try:
        yield cli_ctx
    finally:
        cli_context.set_test_context(None)


@pytest.fixture
def install_ctx_no_encoder(cli_ctx_no_encoder: CliContext) -> Iterator[CliContext]:
    cli_context.set_test_context(cli_ctx_no_encoder)
    try:
        yield cli_ctx_no_encoder
    finally:
        cli_context.set_test_context(None)
