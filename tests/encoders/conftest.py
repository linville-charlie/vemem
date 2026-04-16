"""Shared fixtures + integration gating for the encoders test tree.

The real encoder integration tests download ~200MB of InsightFace weights on
first run. Running them on every CI invocation would be hostile. We mark slow
model-bearing tests with ``@pytest.mark.integration`` and skip them unless
``VEMEM_RUN_INTEGRATION=1`` is set in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration-marked tests unless VEMEM_RUN_INTEGRATION=1."""

    if os.environ.get("VEMEM_RUN_INTEGRATION") == "1":
        return

    skip = pytest.mark.skip(reason="integration tests; set VEMEM_RUN_INTEGRATION=1 to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
