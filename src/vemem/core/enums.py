"""Enumerated types used across the domain model.

Kept as plain ``str`` enums so they round-trip cleanly through JSON, MCP wire
formats, LanceDB columns, and tool schemas without custom serializers.
"""

from __future__ import annotations

from enum import StrEnum


class Polarity(StrEnum):
    """Whether a binding claims identity or forbids it.

    Negative bindings are the durable "this is NOT that entity" signal. See
    ``docs/spec/identity-semantics.md`` §3.3.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"


class Modality(StrEnum):
    """What kind of signal an observation carries.

    v0 exercises ``face`` end-to-end; the others are scaffold for v0.1+.
    """

    FACE = "face"
    OBJECT = "object"
    SCENE = "scene"
    AUDIO = "audio"


class Kind(StrEnum):
    """Whether an entity is a specific instance or a class.

    See ``docs/spec/identity-semantics.md`` §5. v0 exercises ``instance``.
    """

    INSTANCE = "instance"
    TYPE = "type"


class Status(StrEnum):
    """Lifecycle state of an entity.

    ``MERGED_INTO`` and ``FORGOTTEN`` are terminal tombstones that preserve the
    audit trail without keeping biometric content (see §4.5, §4.3).
    """

    ACTIVE = "active"
    MERGED_INTO = "merged_into"
    FORGOTTEN = "forgotten"
    RESTRICTED = "restricted"


class Source(StrEnum):
    """Where a statement (fact / event / relationship) originated."""

    USER = "user"
    VLM = "vlm"
    LLM = "llm"
    IMPORT = "import"


class Method(StrEnum):
    """How a binding was produced.

    Distinguishes authoritative user decisions from auto-suggestions and
    cross-session migrations. See §3.3.
    """

    USER_LABEL = "user_label"
    USER_REJECT = "user_reject"
    AUTO_SUGGEST = "auto_suggest"
    LLM_ASSIST = "llm_assist"
    MIGRATED = "migrated"


class OpType(StrEnum):
    """Identifier for operations recorded in the EventLog (§3.7)."""

    LABEL = "label"
    RELABEL = "relabel"
    MERGE = "merge"
    SPLIT = "split"
    FORGET = "forget"
    REMEMBER = "remember"
    RETRACT_FACT = "retract_fact"
    AUTO_SUGGEST_COMMIT = "auto_suggest_commit"
    UNDO = "undo"
    RESTRICT = "restrict"
    UNRESTRICT = "unrestrict"
    EXPORT = "export"
