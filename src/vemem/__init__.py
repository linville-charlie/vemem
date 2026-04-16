"""vemem — Persistent visual entity memory for AI agents.

Bridges vision models and text LLMs by resolving image observations to named
entities and accumulating knowledge per entity across sessions.

Quick start::

    from vemem import Vemem
    vem = Vemem()                           # LanceDB at ~/.vemem, InsightFace
    observations = vem.observe(image_bytes)
    entity = vem.label([o.id for o in observations], name="Charlie")
    vem.remember(entity.id, fact="runs marathons")

For full control over backends (custom Store, custom Encoder), construct
a :class:`Vemem` with the pieces you want, or skip this class and call the
:mod:`vemem.core.ops` functions directly.

See ``docs/spec/identity-semantics.md`` for the design and ``CLAUDE.md`` for
contributor notes.
"""

from vemem.core.enums import Kind, Method, Modality, OpType, Polarity, Source, Status
from vemem.core.errors import (
    EntityUnavailableError,
    KindMismatchError,
    ModalityMismatchError,
    NoCompatibleEncoderError,
    OperationNotReversibleError,
    SchemaVersionError,
    VemError,
)
from vemem.core.ops import RecallSnapshot
from vemem.core.types import (
    Binding,
    Candidate,
    Embedding,
    Entity,
    Event,
    EventLog,
    Fact,
    Observation,
    Relationship,
)
from vemem.facade import Vemem

__version__ = "0.0.1"

__all__ = [
    "Binding",
    "Candidate",
    "Embedding",
    "Entity",
    "EntityUnavailableError",
    "Event",
    "EventLog",
    "Fact",
    "Kind",
    "KindMismatchError",
    "Method",
    "Modality",
    "ModalityMismatchError",
    "NoCompatibleEncoderError",
    "Observation",
    "OpType",
    "OperationNotReversibleError",
    "Polarity",
    "RecallSnapshot",
    "Relationship",
    "SchemaVersionError",
    "Source",
    "Status",
    "VemError",
    "Vemem",
]
