"""Schema version tracking for the LanceDB backend.

v0 has exactly one schema version (``1``). This module exists now — rather
than being added later — so the migration call site is wired in from the
start; when v0.1 introduces a schema change, the migration path is already
the place it belongs, avoiding the "add migrations later" retrofit that
usually corrupts the first few real stores.

The contract:
    - ``CURRENT_SCHEMA_VERSION`` is the version this code understands.
    - ``write_schema_version`` records it in the ``meta`` table on fresh init.
    - ``read_schema_version`` returns what's recorded on disk (or ``None`` if
      the store is new).
    - ``check_schema_compat`` raises ``SchemaVersionError`` on an unknown
      forward version — we never silently open a newer store with older code.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from vemem.core.errors import SchemaVersionError
from vemem.storage.schemas import META_TABLE

if TYPE_CHECKING:
    import lancedb


CURRENT_SCHEMA_VERSION: int = 1
"""Schema version understood by this library build."""

SCHEMA_VERSION_KEY: str = "schema_version"


def read_schema_version(db: lancedb.DBConnection) -> int | None:
    """Return the schema version recorded in the ``meta`` table, or ``None``."""

    if META_TABLE not in db.table_names():
        return None
    table = db.open_table(META_TABLE)
    rows = table.search().where(f"key = '{SCHEMA_VERSION_KEY}'").limit(1).to_arrow()
    if rows.num_rows == 0:
        return None
    payload = rows.column("value_json")[0].as_py()
    parsed = json.loads(payload)
    if not isinstance(parsed, int):
        raise SchemaVersionError(f"meta.{SCHEMA_VERSION_KEY} holds non-int value: {parsed!r}")
    return parsed


def write_schema_version(db: lancedb.DBConnection, version: int) -> None:
    """Persist ``version`` in the ``meta`` table.

    Assumes ``meta`` exists (created in ``LanceDBStore.__init__``). Overwrites
    any existing row for ``schema_version``.
    """

    table = db.open_table(META_TABLE)
    table.delete(f"key = '{SCHEMA_VERSION_KEY}'")
    table.add(
        [
            {
                "key": SCHEMA_VERSION_KEY,
                "value_json": json.dumps(version),
            }
        ]
    )


def check_schema_compat(on_disk: int | None) -> None:
    """Compare an on-disk schema version against the build's current version.

    - ``None`` means a fresh store — caller will write the current version.
    - Equal means we're good.
    - On-disk > current means the store was written by a newer build; we
      refuse to open it rather than silently mangle the data.
    - On-disk < current is a future migration path; v0 has no older versions
      so this path raises — when v0.1 lands, replace with migration dispatch.
    """

    if on_disk is None:
        return
    if on_disk == CURRENT_SCHEMA_VERSION:
        return
    if on_disk > CURRENT_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"Store schema v{on_disk} is newer than this library (v{CURRENT_SCHEMA_VERSION}). "
            "Upgrade vemem before opening."
        )
    # on_disk < current: v0 has no earlier versions, so anything here is a bug
    # or a hand-written file. Fail loud.
    raise SchemaVersionError(
        f"Store schema v{on_disk} is older than this library (v{CURRENT_SCHEMA_VERSION}) "
        "and no migration is registered. Run `vm migrate` (v0.1+)."
    )
