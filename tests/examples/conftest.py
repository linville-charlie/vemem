"""Shared fixtures for the bridge example tests.

The bridge example (``docs/examples/bridge.py``) is a standalone script, not
part of the installed ``vemem`` package. We load it as a module via
``importlib`` so the script stays copy-paste readable for users while still
being importable by pytest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from tests.support.fake_store import FakeStore

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_PATH = REPO_ROOT / "docs" / "examples" / "bridge.py"


def _load_bridge_module() -> ModuleType:
    """Load ``docs/examples/bridge.py`` as ``bridge_example`` once per session."""
    if "bridge_example" in sys.modules:
        return sys.modules["bridge_example"]
    spec = importlib.util.spec_from_file_location("bridge_example", BRIDGE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bridge_example"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bridge_module() -> ModuleType:
    return _load_bridge_module()


class FixedClock:
    """Monotonically-advancing test clock (microsecond-tick per call)."""

    def __init__(self, start: datetime | None = None) -> None:
        self._start = start or datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)
        self._tick = 0

    def now(self) -> datetime:
        self._tick += 1
        return self._start.replace(microsecond=self._tick)


class StubDetector:
    """One bbox for any non-empty image; zero bboxes otherwise."""

    id: str = "test-stub/detector@0"

    def detect(self, image_bytes: bytes) -> list[tuple[int, int, int, int]]:
        return [(10, 10, 100, 100)] if image_bytes else []


class StubEncoder:
    """8-d deterministic hash encoder — same bytes → same vector."""

    id: str = "test-stub/encoder@0"
    dim: int = 8

    def embed(self, image_crop: bytes) -> tuple[float, ...]:
        digest = hashlib.sha256(image_crop).digest()
        raw = [(b - 128) / 128.0 for b in digest[: self.dim]]
        norm = math.sqrt(sum(x * x for x in raw))
        if norm == 0.0:
            return tuple([1.0] + [0.0] * (self.dim - 1))
        return tuple(x / norm for x in raw)


class RecordingLLM:
    """Stub LLM that records the last (user_msg, context) it was called with.

    Lets tests assert on the context that the bridge assembled and handed off,
    without depending on the stub's exact reply wording.
    """

    def __init__(self) -> None:
        self.last_user_msg: str | None = None
        self.last_context: str | None = None
        self.reply: str = "stub-llm: ok"

    def __call__(self, user_msg: str, context: str) -> str:
        self.last_user_msg = user_msg
        self.last_context = context
        return self.reply


class RecordingVLM:
    """Stub VLM returning a fixed description; records each call."""

    def __init__(self, description: str = "a person in indoor lighting") -> None:
        self.description = description
        self.calls: int = 0

    def __call__(self, image_bytes: bytes) -> str:
        self.calls += 1
        return self.description


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock()


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def detector() -> StubDetector:
    return StubDetector()


@pytest.fixture
def encoder() -> StubEncoder:
    return StubEncoder()


@pytest.fixture
def vlm() -> RecordingVLM:
    return RecordingVLM()


@pytest.fixture
def llm() -> RecordingLLM:
    return RecordingLLM()
