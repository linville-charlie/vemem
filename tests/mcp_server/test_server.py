"""Integration tests: spawn the MCP server as a subprocess and talk JSON-RPC.

These tests catch wire-format regressions that unit tests would miss — for
example, a handler that accidentally returns a non-JSON-serializable object.
They also prove that ``python -m vemem.mcp_server`` launches cleanly and
responds to ``list_tools`` + a representative ``call_tool`` round-trip.

Uses the MCP SDK's ``stdio_client`` + ``ClientSession`` to avoid hand-rolling
the JSON-RPC protocol.

Gated behind the ``integration`` marker (see ``conftest.py``) because
subprocess spawn is slower than in-process unit tests. Runs offline: the
server is launched with ``VEMEM_MCP_TEST_MODE=1`` so it uses the in-memory
FakeStore + stub encoder instead of LanceDB + InsightFace weights.
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.integration


IMAGE_BYTES = b"fake-image-bytes-for-integration"
IMAGE_B64 = base64.b64encode(IMAGE_BYTES).decode("ascii")


@asynccontextmanager
async def _spawn_server() -> AsyncIterator[ClientSession]:
    """Spawn the server via ``python -m vemem.mcp_server`` in test mode.

    Yields a fully-initialized :class:`ClientSession`. Test mode swaps the
    heavy backends for the FakeStore + stub encoder so the subprocess starts
    fast and without network dependencies.
    """
    env = os.environ.copy()
    env["VEMEM_MCP_TEST_MODE"] = "1"
    # Ensure the subprocess sees the tests/ tree so the test-mode context can
    # import tests.support.fake_store. Pytest runs from the repo root, so
    # PYTHONPATH defaults already cover this; we force it explicitly in case
    # pytest is invoked from a different CWD.
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root + (os.pathsep + existing if existing else "")

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "vemem.mcp_server"],
        env=env,
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


def test_list_tools_returns_expected_surface() -> None:
    async def inner() -> list[str]:
        async with _spawn_server() as session:
            result = await session.list_tools()
            return sorted(t.name for t in result.tools)

    names = asyncio.run(inner())
    expected = {
        "observe_image",
        "identify_image",
        "identify_by_name",
        "label",
        "relabel",
        "merge",
        "split",
        "forget",
        "restrict",
        "unrestrict",
        "remember",
        "recall",
        "undo",
        "export",
    }
    assert expected.issubset(set(names))


def test_observe_label_recall_roundtrip_via_stdio() -> None:
    """End-to-end: observe → label → remember → recall via real JSON-RPC."""

    async def inner() -> dict[str, Any]:
        async with _spawn_server() as session:
            observed = await session.call_tool(
                "observe_image",
                arguments={"image_base64": IMAGE_B64},
            )
            assert observed.structuredContent is not None
            obs_id = observed.structuredContent["observations"][0]["id"]

            labeled = await session.call_tool(
                "label",
                arguments={
                    "observation_ids": [obs_id],
                    "entity_name_or_id": "Renata",
                    "actor": "test:integration",
                },
            )
            assert labeled.structuredContent is not None
            entity_id = labeled.structuredContent["id"]

            await session.call_tool(
                "remember",
                arguments={
                    "entity_id": entity_id,
                    "content": "speaks three languages",
                    "actor": "test:integration",
                },
            )

            recalled = await session.call_tool(
                "recall",
                arguments={"entity_id": entity_id},
            )
            assert recalled.structuredContent is not None
            return dict(recalled.structuredContent)

    snap = asyncio.run(inner())
    assert snap["entity"]["name"] == "Renata"
    facts = snap["facts"]
    assert len(facts) == 1
    assert facts[0]["content"] == "speaks three languages"
