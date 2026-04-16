"""Smoke test: the package imports and reports its version.

Placeholder until the core ops are implemented. Exists so CI has something to
run and the project layout is validated end-to-end.
"""


def test_package_imports() -> None:
    import vemem

    assert vemem.__version__ == "0.0.1"


def test_submodules_import() -> None:
    from vemem import cli, core, encoders, mcp_server, storage, tools  # noqa: F401
