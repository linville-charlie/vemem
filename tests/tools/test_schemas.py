"""Tests for the OpenAI-compatible tool schema export (Wave 3E).

Lock the exact set of exported tools, the OpenAI function-calling wire shape,
and the contract between the schemas and the enums they quote. Also snapshot
the canonical ``tools.json`` so drift from the MCP surface is caught at CI
time rather than when some third-party caller silently sees a schema change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vemem.core.enums import Source
from vemem.tools import all_tools, write_tools_json
from vemem.tools.export import main as export_main

EXPECTED_TOOL_NAMES = {
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

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "tools.json"
REGEN_CMD = "uv run python -m vemem.tools.export > tests/tools/snapshots/tools.json"


# ---------- shape + inventory ----------


def test_all_tools_returns_all_fourteen():
    tools = all_tools()
    assert len(tools) == 14
    names = {t["function"]["name"] for t in tools}
    assert names == EXPECTED_TOOL_NAMES


def test_every_tool_has_openai_function_shape():
    for tool in all_tools():
        assert tool["type"] == "function"
        fn = tool["function"]
        assert set(fn.keys()) >= {"name", "description", "parameters"}
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        # required must be a subset of properties
        assert set(params["required"]) <= set(params["properties"].keys())


def test_every_tool_description_nontrivial():
    for tool in all_tools():
        desc = tool["function"]["description"]
        assert isinstance(desc, str)
        assert len(desc) >= 20, f"tool {tool['function']['name']} has a trivial description"


# ---------- required-field spot checks ----------


def _tool_by_name(name: str) -> dict[str, Any]:
    by_name = {t["function"]["name"]: t for t in all_tools()}
    return by_name[name]


def test_label_requires_observation_ids_and_entity():
    schema = _tool_by_name("label")
    required = set(schema["function"]["parameters"]["required"])
    assert required == {"observation_ids", "entity_name_or_id"}


def test_merge_requires_entity_ids():
    schema = _tool_by_name("merge")
    required = set(schema["function"]["parameters"]["required"])
    assert required == {"entity_ids"}


def test_forget_requires_entity_id_only():
    schema = _tool_by_name("forget")
    required = set(schema["function"]["parameters"]["required"])
    assert required == {"entity_id"}


def test_split_requires_entity_and_groups():
    schema = _tool_by_name("split")
    required = set(schema["function"]["parameters"]["required"])
    assert required == {"entity_id", "groups"}


def test_undo_has_no_required_fields():
    schema = _tool_by_name("undo")
    required = schema["function"]["parameters"]["required"]
    assert required == []


# ---------- enum sync ----------


def test_remember_source_enum_matches_source_stringenum():
    schema = _tool_by_name("remember")
    source_prop = schema["function"]["parameters"]["properties"]["source"]
    assert set(source_prop["enum"]) == {s.value for s in Source}


def test_observe_image_modality_enum_matches_modality_stringenum():
    from vemem.core.enums import Modality

    schema = _tool_by_name("observe_image")
    modality_prop = schema["function"]["parameters"]["properties"]["modality"]
    assert set(modality_prop["enum"]) == {m.value for m in Modality}


# ---------- image/base64 field annotations ----------


def test_image_base64_fields_are_tagged_byte_format():
    for tool_name in ("observe_image", "identify_image"):
        schema = _tool_by_name(tool_name)
        image_prop = schema["function"]["parameters"]["properties"]["image_base64"]
        assert image_prop["type"] == "string"
        # OpenAPI 3.0 convention: base64-encoded
        assert image_prop.get("format") == "byte"
        assert image_prop.get("contentEncoding") == "base64"
        assert "base64" in image_prop["description"].lower()


def test_list_parameters_are_typed_arrays():
    # label: observation_ids is list[str]
    label = _tool_by_name("label")
    obs_prop = label["function"]["parameters"]["properties"]["observation_ids"]
    assert obs_prop["type"] == "array"
    assert obs_prop["items"]["type"] == "string"

    # split: groups is list[list[str]]
    split = _tool_by_name("split")
    groups_prop = split["function"]["parameters"]["properties"]["groups"]
    assert groups_prop["type"] == "array"
    assert groups_prop["items"]["type"] == "array"
    assert groups_prop["items"]["items"]["type"] == "string"


# ---------- snapshot ----------


def test_snapshot_exists():
    assert SNAPSHOT_PATH.exists(), (
        f"canonical snapshot missing: {SNAPSHOT_PATH}. Run `{REGEN_CMD}` to regenerate it."
    )


def test_tools_match_snapshot():
    if not SNAPSHOT_PATH.exists():
        pytest.fail(f"canonical snapshot missing. Run `{REGEN_CMD}` to regenerate it.")
    recorded = json.loads(SNAPSHOT_PATH.read_text())
    actual = all_tools()
    if recorded != actual:
        pytest.fail(
            "Exported tool schemas have drifted from the committed snapshot.\n"
            f"Regenerate with:\n    {REGEN_CMD}\n"
            "Then review the diff and commit. The snapshot is the contract the "
            "OpenAI-style callers see."
        )


def test_write_tools_json_roundtrips(tmp_path: Path):
    out = tmp_path / "tools.json"
    write_tools_json(out)
    loaded = json.loads(out.read_text())
    assert loaded == all_tools()


def test_export_main_writes_to_stdout(capsys: pytest.CaptureFixture[str]):
    rc = export_main([])
    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed == all_tools()


def test_export_main_writes_to_file(tmp_path: Path):
    out = tmp_path / "tools.json"
    rc = export_main(["--output", str(out)])
    assert rc == 0
    parsed = json.loads(out.read_text())
    assert parsed == all_tools()
