"""Domain errors raised by the core ops.

All inherit from ``VemError`` so callers can catch the library's errors as a
group without swallowing unrelated ``Exception`` subclasses.
"""

from __future__ import annotations


class VemError(Exception):
    """Base class for every error raised by vemem core."""


class ModalityMismatchError(VemError):
    """Operation mixes entities or observations of different modalities."""


class KindMismatchError(VemError):
    """Operation mixes ``instance`` and ``type`` entities (see §4.3).

    ``merge`` across kinds is meaningless; callers likely want an ``instance_of``
    relationship instead.
    """


class EntityUnavailableError(VemError):
    """Target entity is ``forgotten`` or ``merged_into`` and cannot be used."""


class OperationNotReversibleError(VemError):
    """Requested ``undo`` on an event that is not reversible (see §4.5, §4.6)."""


class NoCompatibleEncoderError(VemError):
    """Query encoder has no bindings in the gallery (see §4.0).

    The library deliberately refuses to silently translate across encoders.
    """


class SchemaVersionError(VemError):
    """Store schema version is incompatible with this library version."""
