"""Tests that domain errors are importable and carry useful payloads."""

from vemem.core.errors import (
    EntityUnavailableError,
    KindMismatchError,
    ModalityMismatchError,
    NoCompatibleEncoderError,
    OperationNotReversibleError,
    SchemaVersionError,
    VemError,
)


def test_all_errors_inherit_vemerror() -> None:
    for cls in (
        ModalityMismatchError,
        KindMismatchError,
        EntityUnavailableError,
        OperationNotReversibleError,
        NoCompatibleEncoderError,
        SchemaVersionError,
    ):
        assert issubclass(cls, VemError)
        assert issubclass(cls, Exception)


def test_errors_accept_message() -> None:
    err = ModalityMismatchError("face vs object")
    assert "face" in str(err)
