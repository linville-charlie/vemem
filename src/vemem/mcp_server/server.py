"""FastMCP server wiring the core ops into MCP tools.

Instantiates the store, encoder, and detector, then registers every tool in
``tools.py`` as an MCP-callable function. Tool schemas are generated from the
function signatures by FastMCP — so updates to handler signatures flow
through without a separate schema file.

The heavy components (LanceDB, InsightFace) are loaded in ``build_context``
so they can be overridden in tests via ``VEMEM_MCP_TEST_MODE``. In that mode
we inject the in-memory ``FakeStore`` + deterministic stub encoder/detector
so the integration test roundtrips JSON-RPC without network or model weights.
"""

from __future__ import annotations

import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from vemem.core.protocols import Store
from vemem.mcp_server import tools
from vemem.mcp_server.config import (
    UTCClock,
    default_store_path,
    load_encoder_and_detector,
    test_mode_enabled,
)
from vemem.mcp_server.tools import ServerContext

SERVER_NAME = "vemem"
SERVER_INSTRUCTIONS = (
    "Visual entity memory — persistent identity for faces (v0). "
    "Use observe_image + label to teach; identify_image / identify_by_name to recognize; "
    "remember + recall for per-entity knowledge; merge / split / relabel / forget for "
    "corrections and privacy; undo for time-limited reversal."
)


def build_context() -> ServerContext:
    """Construct a :class:`ServerContext` for the current process environment.

    In test mode (env ``VEMEM_MCP_TEST_MODE=1``) we return a context wired to
    the in-memory FakeStore and a stub encoder/detector so the integration
    test can exercise the wire format without downloading weights. In normal
    mode we open a LanceDB dataset at ``$VEMEM_HOME`` and attempt to load the
    InsightFace pipeline; if weights are missing we still build a context —
    the image tools report the error on invocation.
    """

    if test_mode_enabled():
        return _build_test_context()

    from vemem.storage.lancedb_store import LanceDBStore

    store: Store = LanceDBStore(path=default_store_path())
    status = load_encoder_and_detector()
    # Log the encoder error to stderr so operators see it at startup.
    if status.error is not None:
        print(f"[vemem.mcp_server] {status.error}", file=sys.stderr)
    return ServerContext(
        store=store,
        clock=UTCClock(),
        encoder=status.encoder,
        detector=status.detector,
        encoder_error=status.error,
    )


def _build_test_context() -> ServerContext:
    """Build a fully in-memory context for the integration test.

    Uses the test-only ``FakeStore`` and a deterministic stub encoder/detector
    so the server can be spawned over stdio without needing network or model
    weights. The test imports from ``tests.support.fake_store`` via the
    installed project tree; we rely on that path being importable when the
    test suite runs ``python -m vemem.mcp_server``.
    """
    # The FakeStore and stubs live under tests/ (not shipped). Import lazily
    # so a normal production import path never pulls them in.
    from tests.mcp_server._test_backends import (
        StubDetector,
        StubEncoder,
    )
    from tests.support.fake_store import FakeStore

    store = FakeStore()
    return ServerContext(
        store=store,
        clock=UTCClock(),
        encoder=StubEncoder(),
        detector=StubDetector(),
        encoder_error=None,
    )


def create_server(ctx: ServerContext | None = None) -> FastMCP:
    """Build the FastMCP app with every tool registered.

    Separating ``create_server`` from ``build_context`` lets tests pass a
    fully-fixtured context (including a preloaded store) rather than going
    through the env-var path every time.
    """

    if ctx is None:
        ctx = build_context()

    mcp = FastMCP(name=SERVER_NAME, instructions=SERVER_INSTRUCTIONS)

    # ---- core identity ------------------------------------------------------

    @mcp.tool(
        name="observe_image",
        description=(
            "Detect and persist observations for every entity (face) in an image. "
            "Accepts a base64-encoded image, runs the detector + encoder, and returns "
            "the observation ids + bboxes that can feed into `label`. Idempotent."
        ),
    )
    def observe_image(
        image_base64: str,
        source_uri: str = "mcp://inline",
        modality: str = "face",
    ) -> dict[str, Any]:
        return tools.observe_image(
            ctx,
            image_base64=image_base64,
            source_uri=source_uri,
            modality=modality,
        )

    @mcp.tool(
        name="identify_image",
        description=(
            "Identify entities in an image without mutating state. Runs the detector "
            "+ encoder and returns ranked Candidate matches per detected face."
        ),
    )
    def identify_image(
        image_base64: str,
        k: int = 5,
        min_confidence: float = 0.5,
        prefer: str = "instance",
    ) -> dict[str, Any]:
        return tools.identify_image(
            ctx,
            image_base64=image_base64,
            k=k,
            min_confidence=min_confidence,
            prefer=prefer,
        )

    @mcp.tool(
        name="identify_by_name",
        description=(
            "Resolve an entity by name or id and return its recall snapshot "
            "(entity metadata + active facts). Convenience wrapper around `recall`."
        ),
    )
    def identify_by_name(entity_name_or_id: str) -> dict[str, Any]:
        return tools.identify_by_name(ctx, entity_name_or_id=entity_name_or_id)

    # ---- state changes ------------------------------------------------------

    @mcp.tool(
        name="label",
        description=(
            "Commit a user-authoritative positive binding: 'these observations are "
            "this entity'. Creates the entity if the name is new. Use to teach."
        ),
    )
    def label_tool(
        observation_ids: list[str],
        entity_name_or_id: str,
        actor: str = tools.DEFAULT_ACTOR,
    ) -> dict[str, Any]:
        return tools.label_tool(
            ctx,
            observation_ids=observation_ids,
            entity_name_or_id=entity_name_or_id,
            actor=actor,
        )

    @mcp.tool(
        name="relabel",
        description=(
            "Move a single observation to a different entity. Also emits a negative "
            "binding against the old entity so the auto-clusterer never re-attaches."
        ),
    )
    def relabel_tool(
        observation_id: str,
        new_entity_name_or_id: str,
        actor: str = tools.DEFAULT_ACTOR,
    ) -> dict[str, Any]:
        return tools.relabel_tool(
            ctx,
            observation_id=observation_id,
            new_entity_name_or_id=new_entity_name_or_id,
            actor=actor,
        )

    @mcp.tool(
        name="merge",
        description=(
            "'These are the same.' Folds losers into a winner entity; facts and "
            "relationships migrate with provenance. Rejects modality or kind "
            "mismatches. `keep` is 'oldest' (default) or an explicit entity id."
        ),
    )
    def merge_tool(
        entity_ids: list[str],
        keep: str = "oldest",
        actor: str = tools.DEFAULT_ACTOR,
    ) -> dict[str, Any]:
        return tools.merge_tool(
            ctx,
            entity_ids=entity_ids,
            keep=keep,
            actor=actor,
        )

    @mcp.tool(
        name="split",
        description=(
            "'This is actually N different entities.' `groups[0]` stays on the "
            "original id; each subsequent group becomes a new entity. Cross-wise "
            "negatives prevent auto-re-merge. `fact_policy` in "
            "{'keep_original', 'copy_to_all', 'manual'}."
        ),
    )
    def split_tool(
        entity_id: str,
        groups: list[list[str]],
        fact_policy: str = "keep_original",
        actor: str = tools.DEFAULT_ACTOR,
    ) -> dict[str, Any]:
        return tools.split_tool(
            ctx,
            entity_id=entity_id,
            groups=groups,
            fact_policy=fact_policy,
            actor=actor,
        )

    @mcp.tool(
        name="forget",
        description=(
            "Hard-delete everything tied to an entity and prune old LanceDB versions "
            "(GDPR Art. 17). NOT reversible by undo. Returns per-table deletion counts."
        ),
    )
    def forget_tool(
        entity_id: str,
        grace_days: int = 0,
        actor: str = tools.DEFAULT_ACTOR,
    ) -> dict[str, Any]:
        return tools.forget_tool(
            ctx,
            entity_id=entity_id,
            grace_days=grace_days,
            actor=actor,
        )

    @mcp.tool(
        name="restrict",
        description=(
            "Stop using the entity for inference without deleting it (GDPR Art. 18). "
            "Restricted entities are excluded from identify() but facts remain readable."
        ),
    )
    def restrict_tool(
        entity_id: str,
        actor: str = tools.DEFAULT_ACTOR,
    ) -> dict[str, Any]:
        return tools.restrict_tool(ctx, entity_id=entity_id, actor=actor)

    @mcp.tool(
        name="unrestrict",
        description="Reverse `restrict` — return the entity to ACTIVE status.",
    )
    def unrestrict_tool(
        entity_id: str,
        actor: str = tools.DEFAULT_ACTOR,
    ) -> dict[str, Any]:
        return tools.unrestrict_tool(ctx, entity_id=entity_id, actor=actor)

    # ---- knowledge ---------------------------------------------------------

    @mcp.tool(
        name="remember",
        description=(
            "Attach a free-text fact to an entity. Facts are bi-temporal and stay "
            "valid until retracted. `source` is 'user' | 'vlm' | 'llm' | 'import'."
        ),
    )
    def remember_tool(
        entity_id: str,
        content: str,
        source: str = "user",
        actor: str = tools.DEFAULT_ACTOR,
    ) -> dict[str, Any]:
        return tools.remember_tool(
            ctx,
            entity_id=entity_id,
            content=content,
            source=source,
            actor=actor,
        )

    @mcp.tool(
        name="recall",
        description="Return an entity plus its currently-active facts.",
    )
    def recall_tool(entity_id: str) -> dict[str, Any]:
        return tools.recall_tool(ctx, entity_id=entity_id)

    # ---- audit -------------------------------------------------------------

    @mcp.tool(
        name="undo",
        description=(
            "Reverse a prior reversible operation. With no `event_id`, undoes the "
            "most recent reversible event by `actor`. Forget is not reversible."
        ),
    )
    def undo_tool(
        event_id: int | None = None,
        actor: str = tools.DEFAULT_ACTOR,
    ) -> dict[str, Any]:
        return tools.undo_tool(ctx, event_id=event_id, actor=actor)

    @mcp.tool(
        name="export",
        description=(
            "GDPR Art. 20 data portability dump — observations, bindings, facts, "
            "events, relationships, event log for one entity. Embeddings excluded "
            "by default (biometric vectors in exports are usually worse than useless)."
        ),
    )
    def export_tool(
        entity_id: str,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        return tools.export_tool(
            ctx,
            entity_id=entity_id,
            include_embeddings=include_embeddings,
        )

    return mcp


def run() -> None:
    """Entry point — build the server and start the stdio transport.

    Blocks forever; FastMCP's ``run`` manages the event loop and stdio pipes.
    """
    server = create_server()
    server.run(transport="stdio")
