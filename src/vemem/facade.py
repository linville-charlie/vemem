"""Top-level :class:`Vemem` convenience wrapper.

Most callers want one object with every op as a method, sensible defaults
for storage and encoders, and a graceful-degradation story when face weights
aren't installed yet. That's :class:`Vemem`. For full control over backends
(custom Store, custom Encoder) pass them in explicitly, or skip this class
entirely and call the :mod:`vemem.core.ops` functions against your own Store.

Quick start::

    from vemem import Vemem

    vem = Vemem()                          # LanceDB at ~/.vemem, InsightFace
    observations = vem.observe(image_bytes)
    candidates = vem.identify(image_bytes)
    if not candidates:
        entity = vem.label([o.id for o in observations], name="Charlie")
        vem.remember(entity.id, fact="runs marathons")
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from vemem.core import ops
from vemem.core.enums import Modality, Source
from vemem.core.ops import RecallSnapshot
from vemem.core.protocols import Clock, Detector, Encoder, Store
from vemem.core.types import Candidate, Entity, EventLog, Fact, Observation

# Sentinel distinguishing "caller did not pass" (→ auto-load InsightFace) from
# "caller passed None" (→ image pipeline disabled).
_UNSET: Any = object()


class _UTCClock:
    """The one ``Clock`` impl the library needs. Not cached — constructing is free."""

    def now(self) -> datetime:
        return datetime.now(UTC)


def _default_home() -> Path:
    """Resolve ``VEMEM_HOME`` → ``~/.vemem``."""
    env = os.environ.get("VEMEM_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".vemem"


def _try_load_insightface() -> tuple[Encoder | None, Detector | None, str | None]:
    """Attempt to import + construct the face-default encoder/detector.

    Returns ``(encoder, detector, error_message)``. Encoder/detector are
    ``None`` on any failure; ``error_message`` explains what to install.
    """
    try:
        from vemem.encoders.insightface_detector import InsightFaceDetector
        from vemem.encoders.insightface_encoder import InsightFaceEncoder
    except ImportError as e:  # pragma: no cover - insightface is a declared dep
        return None, None, f"insightface import failed: {e}"

    try:
        encoder = InsightFaceEncoder()
        detector = InsightFaceDetector()
    except Exception as e:
        return (
            None,
            None,
            (
                f"InsightFace failed to initialize: {e}. "
                "Weights usually live under ~/.insightface/models/buffalo_l/ and "
                "download automatically on first construction; a network-blocked "
                "first run is the typical cause."
            ),
        )
    return encoder, detector, None


class Vemem:
    """High-level convenience wrapper around a vemem store + encoders.

    Thread-safety: same semantics as the underlying Store (v0 is single-writer
    by design; spec §6). Context-manager compatible — ``with Vemem() as vem: …``
    closes the store on exit.
    """

    def __init__(
        self,
        *,
        home: Path | str | None = None,
        store: Store | None = None,
        encoder: Encoder | None = _UNSET,
        detector: Detector | None = _UNSET,
        clock: Clock | None = None,
        actor: str = "lib:python",
    ) -> None:
        """Passing ``encoder=None`` / ``detector=None`` explicitly disables the
        image pipeline; omitting them triggers InsightFace auto-load with
        graceful fallback if weights aren't installed yet.
        """
        self._actor = actor
        self._clock: Clock = clock or _UTCClock()

        # Store — use the provided one, or open a LanceDBStore at ``home``.
        if store is not None:
            self._store = store
            self._owns_store = False
        else:
            from vemem.storage.lancedb_store import LanceDBStore

            resolved_home = Path(home).expanduser().resolve() if home else _default_home()
            resolved_home.mkdir(parents=True, exist_ok=True)
            self._store = LanceDBStore(path=str(resolved_home))
            self._owns_store = True

        # Encoder / Detector resolution:
        # - If caller passed a concrete value, use it.
        # - If caller passed None explicitly, leave disabled.
        # - If caller omitted (sentinel), try the InsightFace auto-load.
        self._encoder_load_error: str | None = None
        auto_encoder: Encoder | None = None
        auto_detector: Detector | None = None
        need_auto = encoder is _UNSET or detector is _UNSET
        if need_auto:
            auto_encoder, auto_detector, self._encoder_load_error = _try_load_insightface()

        self._encoder = auto_encoder if encoder is _UNSET else encoder
        self._detector = auto_detector if detector is _UNSET else detector

    # ---- lifecycle ---------------------------------------------------------

    @property
    def store(self) -> Store:
        """Direct access to the underlying store (for advanced callers)."""
        return self._store

    @property
    def actor(self) -> str:
        return self._actor

    def close(self) -> None:
        if self._owns_store:
            self._store.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---- image path --------------------------------------------------------

    def _require_image_pipeline(self) -> tuple[Encoder, Detector]:
        if self._encoder is None or self._detector is None:
            hint = self._encoder_load_error or "no encoder/detector configured"
            raise RuntimeError(f"vemem image pipeline unavailable — {hint}")
        return self._encoder, self._detector

    def observe(
        self,
        image_bytes: bytes,
        *,
        source_uri: str | None = None,
        modality: Modality = Modality.FACE,
    ) -> list[Observation]:
        """Detect, embed, and persist observations from an image.

        Delegates to :func:`vemem.pipeline.observe_image`. Raises
        ``RuntimeError`` with a diagnostic if the encoder isn't loaded.
        """
        from vemem.pipeline import observe_image

        encoder, detector = self._require_image_pipeline()
        return observe_image(
            self._store,
            image_bytes=image_bytes,
            detector=detector,
            encoder=encoder,
            clock=self._clock,
            modality=modality,
            source_uri=source_uri,
        )

    def identify(
        self,
        query: bytes | tuple[float, ...],
        *,
        k: int = 5,
        min_confidence: float = 0.5,
        prefer: str = "instance",
        encoder_id: str | None = None,
    ) -> list[Candidate]:
        """Return candidate entities matching ``query``.

        ``query`` may be raw image bytes (we encode) or a pre-computed vector
        (we pass straight to the Store). If you pass a vector, supply
        ``encoder_id`` so the Store can route to the right gallery.
        """
        if isinstance(query, (bytes, bytearray, memoryview)):
            encoder, _ = self._require_image_pipeline()
            vec = tuple(encoder.embed(bytes(query)))
            eid = encoder.id
        else:
            vec = tuple(query)
            if encoder_id is None:
                raise ValueError("encoder_id is required when query is a vector")
            eid = encoder_id

        return ops.identify(
            self._store,
            encoder_id=eid,
            vector=vec,
            k=k,
            min_confidence=min_confidence,
            prefer=prefer,
        )

    # ---- op passthroughs ---------------------------------------------------

    def label(self, observation_ids: list[str], name: str) -> Entity:
        return ops.label(self._store, observation_ids, name, clock=self._clock, actor=self._actor)

    def relabel(self, observation_id: str, new_name: str) -> Entity:
        return ops.relabel(
            self._store, observation_id, new_name, clock=self._clock, actor=self._actor
        )

    def merge(self, entity_ids: list[str], *, keep: str = "oldest") -> Entity:
        return ops.merge(self._store, entity_ids, keep=keep, clock=self._clock, actor=self._actor)

    def split(
        self,
        entity_id: str,
        groups: list[list[str]],
        *,
        fact_policy: str = "keep_original",
    ) -> list[Entity]:
        return ops.split(
            self._store,
            entity_id,
            groups,
            fact_policy=fact_policy,
            clock=self._clock,
            actor=self._actor,
        )

    def forget(self, entity_id: str, *, grace_days: int = 0) -> dict[str, int]:
        return ops.forget(
            self._store,
            entity_id=entity_id,
            clock=self._clock,
            actor=self._actor,
            grace_days=grace_days,
        )

    def restrict(self, entity_id: str) -> Entity:
        return ops.restrict(self._store, entity_id=entity_id, clock=self._clock, actor=self._actor)

    def unrestrict(self, entity_id: str) -> Entity:
        return ops.unrestrict(
            self._store, entity_id=entity_id, clock=self._clock, actor=self._actor
        )

    def remember(self, entity_id: str, fact: str, *, source: Source = Source.USER) -> Fact:
        return ops.remember(
            self._store,
            entity_id=entity_id,
            content=fact,
            source=source,
            clock=self._clock,
            actor=self._actor,
        )

    def recall(self, entity_id: str, *, active_only: bool = True) -> RecallSnapshot:
        return ops.recall(self._store, entity_id=entity_id, active_only=active_only)

    def undo(self, event_id: int | None = None) -> EventLog:
        return ops.undo(self._store, event_id=event_id, clock=self._clock, actor=self._actor)

    def export(self, entity_id: str, *, include_embeddings: bool = False) -> dict[str, Any]:
        return ops.export(self._store, entity_id=entity_id, include_embeddings=include_embeddings)
