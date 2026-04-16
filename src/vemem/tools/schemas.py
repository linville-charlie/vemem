"""OpenAI-compatible function-calling JSON schemas for every core op.

These schemas describe the same operation surface as ``vemem.mcp_server.tools``
but in the OpenAI function-calling wire format so non-MCP function-calling
LLMs (OpenAI chat completions, Gemini function calling, Anthropic tool use,
Ollama function calling, openclaw pipelines, ...) can call vemem ops without
the MCP SDK.

Each schema is a dict of the form::

    {
        "type": "function",
        "function": {
            "name": str,
            "description": str,
            "parameters": {
                "type": "object",
                "properties": {...},
                "required": [...],
            },
        },
    }

Field shapes follow JSON Schema draft-07 with OpenAPI 3.0 conventions for
binary payloads (``format="byte"``, ``contentEncoding="base64"`` for
base64-encoded image bytes — as required by §3.1 of the identity spec).

Enum-valued parameters quote the values from ``vemem.core.enums`` at import
time so any future enum change automatically updates the public schema (and
the snapshot test catches the drift).

The MCP server is the "live" stateful surface; these schemas are the
"stateless paste-into-your-prompt" surface. Both describe the SAME ops — if
they drift, that is a bug.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from vemem.core.enums import Modality, Source

SchemaBuilder = Callable[[], dict[str, Any]]

# ---------- shared fragments ----------


def _enum_values(enum_cls: type[StrEnum]) -> list[str]:
    """Return the string values of a ``StrEnum`` in declaration order."""
    return [member.value for member in enum_cls]


_ACTOR_FIELD: dict[str, Any] = {
    "type": "string",
    "description": (
        "Who is making this call, for audit attribution. Convention: "
        '"kind:id" — e.g. "user:alice", "agent:my_assistant", "llm:gpt-4". '
        'Defaults to "mcp:unknown" when omitted.'
    ),
    "default": "mcp:unknown",
}

_IMAGE_BASE64_FIELD: dict[str, Any] = {
    "type": "string",
    "format": "byte",
    "contentEncoding": "base64",
    "description": (
        "Raw image bytes, base64-encoded (standard or URL-safe). The server "
        "decodes, hashes, runs the detector, and persists one observation per "
        "detected face."
    ),
}


def _string_array(item_description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "description": item_description,
    }


# ---------- per-op schemas ----------


def _observe_image_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "observe_image",
            "description": (
                "Detect and persist observations for every entity (face) in an image. "
                "Accepts a base64-encoded image, runs the detector + encoder, and "
                "returns the observation ids + bboxes that can feed into `label`. "
                "Idempotent: re-observing the same image returns the same ids."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_base64": _IMAGE_BASE64_FIELD,
                    "source_uri": {
                        "type": "string",
                        "description": (
                            "Caller's reference to the image (path, URL, or opaque "
                            "id). The library stores this but does not fetch it."
                        ),
                        "default": "mcp://inline",
                    },
                    "modality": {
                        "type": "string",
                        "enum": _enum_values(Modality),
                        "description": ("Observation modality. v0 exercises `face` end-to-end."),
                        "default": Modality.FACE.value,
                    },
                },
                "required": ["image_base64"],
            },
        },
    }


def _identify_image_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "identify_image",
            "description": (
                "Identify entities in an image without mutating state. Runs the "
                "detector + encoder and returns ranked Candidate matches per "
                "detected face."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_base64": _IMAGE_BASE64_FIELD,
                    "k": {
                        "type": "integer",
                        "description": "Max number of candidates to return per face.",
                        "default": 5,
                        "minimum": 1,
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": (
                            "Drop candidates below this similarity threshold. Defaults "
                            "are conservative; tune up for stricter matching."
                        ),
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "prefer": {
                        "type": "string",
                        "enum": ["instance", "type", "both"],
                        "description": (
                            "Ranking preference when both instance and type entities "
                            "match the same observation."
                        ),
                        "default": "instance",
                    },
                },
                "required": ["image_base64"],
            },
        },
    }


def _identify_by_name_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "identify_by_name",
            "description": (
                "Resolve an entity by name or id and return its recall snapshot "
                "(entity metadata + active facts). Convenience wrapper around `recall`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name_or_id": {
                        "type": "string",
                        "description": (
                            "Either an entity id (ent_*) or a human-readable name / "
                            "alias. Ids take precedence; name lookup falls back to "
                            "exact-match on name or alias."
                        ),
                    },
                },
                "required": ["entity_name_or_id"],
            },
        },
    }


def _label_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "label",
            "description": (
                "Commit a user-authoritative positive binding: 'these observations "
                "are this entity'. Creates the entity if the name is new. Use to teach."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_ids": _string_array(
                        "Observations to bind to the target entity. Typically returned "
                        "by a prior `observe_image` call."
                    ),
                    "entity_name_or_id": {
                        "type": "string",
                        "description": (
                            "Target entity — id (ent_*) to reuse, or a new name to "
                            "create an entity. Re-using a name never triggers a "
                            "merge; identity is the id, not the name."
                        ),
                    },
                    "actor": _ACTOR_FIELD,
                },
                "required": ["observation_ids", "entity_name_or_id"],
            },
        },
    }


def _relabel_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "relabel",
            "description": (
                "Move a single observation to a different entity. Also emits a "
                "negative binding against the old entity so the auto-clusterer "
                "never re-attaches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_id": {
                        "type": "string",
                        "description": "Observation to re-bind.",
                    },
                    "new_entity_name_or_id": {
                        "type": "string",
                        "description": (
                            "Target entity for the observation. Same name/id rules as `label`."
                        ),
                    },
                    "actor": _ACTOR_FIELD,
                },
                "required": ["observation_id", "new_entity_name_or_id"],
            },
        },
    }


def _merge_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "merge",
            "description": (
                "'These are the same.' Folds losers into a winner entity; facts and "
                "relationships migrate with provenance. Rejects modality or kind "
                "mismatches. `keep` is 'oldest' (default) or an explicit entity id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_ids": _string_array(
                        "Entities to merge. Requires at least two. All must share "
                        "the same modality and kind."
                    ),
                    "keep": {
                        "type": "string",
                        "description": (
                            "Winner selection: 'oldest' picks the one with the "
                            "earliest `created_at`; otherwise pass an entity id to "
                            "force that entity to win."
                        ),
                        "default": "oldest",
                    },
                    "actor": _ACTOR_FIELD,
                },
                "required": ["entity_ids"],
            },
        },
    }


def _split_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "split",
            "description": (
                "'This is actually N different entities.' `groups[0]` stays on the "
                "original id; each subsequent group becomes a new entity. Cross-wise "
                "negative bindings prevent auto-re-merge. `fact_policy` controls "
                "whether facts are copied, kept on the original, or deferred to "
                "manual reconciliation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Entity to split.",
                    },
                    "groups": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "description": (
                            "List of observation-id groups. `groups[0]` stays on the "
                            "original entity; each subsequent group becomes a new "
                            "entity with the same kind + modality."
                        ),
                    },
                    "fact_policy": {
                        "type": "string",
                        "enum": ["keep_original", "copy_to_all", "manual"],
                        "description": (
                            "How to treat facts on the original entity. "
                            "`keep_original` (default) leaves them on the original; "
                            "`copy_to_all` duplicates them to each split-off entity; "
                            "`manual` defers to the caller."
                        ),
                        "default": "keep_original",
                    },
                    "actor": _ACTOR_FIELD,
                },
                "required": ["entity_id", "groups"],
            },
        },
    }


def _forget_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "forget",
            "description": (
                "Hard-delete everything tied to an entity and prune old LanceDB "
                "versions (GDPR Art. 17). NOT reversible by undo. Returns per-table "
                "deletion counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Entity to forget.",
                    },
                    "grace_days": {
                        "type": "integer",
                        "description": (
                            "Reserved for a future soft-delete grace window. v0 "
                            "always hard-deletes regardless of this value."
                        ),
                        "default": 0,
                        "minimum": 0,
                    },
                    "actor": _ACTOR_FIELD,
                },
                "required": ["entity_id"],
            },
        },
    }


def _restrict_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "restrict",
            "description": (
                "Stop using the entity for inference without deleting it (GDPR "
                "Art. 18). Restricted entities are excluded from `identify_image` "
                "but their facts remain readable via `recall`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Entity to restrict.",
                    },
                    "actor": _ACTOR_FIELD,
                },
                "required": ["entity_id"],
            },
        },
    }


def _unrestrict_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "unrestrict",
            "description": "Reverse `restrict` — return the entity to ACTIVE status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Entity to unrestrict.",
                    },
                    "actor": _ACTOR_FIELD,
                },
                "required": ["entity_id"],
            },
        },
    }


def _remember_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Attach a free-text fact to an entity. Facts are bi-temporal and "
                "stay valid until retracted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Entity the fact is about.",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Natural-language fact. The library stores this as "
                            "opaque text; callers that need structure can embed JSON."
                        ),
                    },
                    "source": {
                        "type": "string",
                        "enum": _enum_values(Source),
                        "description": "Where this statement came from.",
                        "default": Source.USER.value,
                    },
                    "actor": _ACTOR_FIELD,
                },
                "required": ["entity_id", "content"],
            },
        },
    }


def _recall_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Return an entity plus its currently-active facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Entity to read.",
                    },
                },
                "required": ["entity_id"],
            },
        },
    }


def _undo_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "undo",
            "description": (
                "Reverse a prior reversible operation. With no `event_id`, undoes "
                "the most recent reversible event by `actor`. `forget` is not "
                "reversible by undo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "integer",
                        "description": (
                            "EventLog id of the event to reverse. Omit to undo the "
                            "most recent reversible event by the calling actor."
                        ),
                    },
                    "actor": _ACTOR_FIELD,
                },
                "required": [],
            },
        },
    }


def _export_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "export",
            "description": (
                "GDPR Art. 20 data portability dump — observations, bindings, "
                "facts, events, relationships, event log for one entity. Raw "
                "embedding vectors are excluded by default (biometric vectors in "
                "exports are usually worse than useless)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Entity to export.",
                    },
                    "include_embeddings": {
                        "type": "boolean",
                        "description": (
                            "Include raw embedding vectors in the dump. Off by "
                            "default; turn on only for round-trip migration."
                        ),
                        "default": False,
                    },
                },
                "required": ["entity_id"],
            },
        },
    }


# ---------- kind-filtered accessors ----------


# Declaration order matches the MCP server's tool registration order so the
# snapshot lines up cleanly with `list_tools` output.
_SCHEMA_BUILDERS: tuple[tuple[str, SchemaBuilder], ...] = (
    ("observe_image", _observe_image_schema),
    ("identify_image", _identify_image_schema),
    ("identify_by_name", _identify_by_name_schema),
    ("label", _label_schema),
    ("relabel", _relabel_schema),
    ("merge", _merge_schema),
    ("split", _split_schema),
    ("forget", _forget_schema),
    ("restrict", _restrict_schema),
    ("unrestrict", _unrestrict_schema),
    ("remember", _remember_schema),
    ("recall", _recall_schema),
    ("undo", _undo_schema),
    ("export", _export_schema),
)


def build_all_schemas() -> list[dict[str, Any]]:
    """Return a fresh list of every tool schema, in canonical order.

    Built fresh on each call so callers can mutate the result without poisoning
    the next caller's view.
    """
    return [builder() for _name, builder in _SCHEMA_BUILDERS]


def schema_for(name: str) -> dict[str, Any]:
    """Return the OpenAI schema for a single tool by name.

    Raises ``KeyError`` if the name is not a recognized tool — callers should
    guard with :data:`TOOL_NAMES` if they need a soft lookup.
    """
    for tool_name, builder in _SCHEMA_BUILDERS:
        if tool_name == name:
            return builder()
    raise KeyError(f"unknown tool: {name!r}")


TOOL_NAMES: tuple[str, ...] = tuple(name for name, _ in _SCHEMA_BUILDERS)
