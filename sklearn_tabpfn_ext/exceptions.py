"""Exceptions raised by vldm.preprocessing."""

from __future__ import annotations


class PreprocessingError(Exception):
    """Base class for vldm.preprocessing errors."""


class ArtifactNotFoundError(PreprocessingError):
    pass


class ArtifactSchemaError(PreprocessingError):
    """meta.json or pipeline.json fails schema validation."""


class ArtifactSecurityError(PreprocessingError):
    """Refusing to load artifact because it would execute Python (pickle)."""


class UnknownOperatorError(PreprocessingError):
    def __init__(self, op_id: str, available: list[str] | None = None):
        suffix = ""
        if available:
            suffix = f". Known: {', '.join(available[:8])}{'…' if len(available) > 8 else ''}"
        super().__init__(f"Operator not registered: {op_id}{suffix}")
        self.op_id = op_id


class OpInitError(PreprocessingError):
    def __init__(self, op_path: str, reason: str):
        super().__init__(f"OpInitError at {op_path!r}: {reason}")
        self.op_path = op_path


class OpStateError(PreprocessingError):
    def __init__(self, op_path: str, key: str, reason: str):
        super().__init__(f"OpStateError at {op_path!r} key={key!r}: {reason}")
        self.op_path = op_path
        self.key = key


class VersionMismatchWarning(UserWarning):
    pass


class UnsupportedConversionError(PreprocessingError):
    """Translator encountered a class it cannot convert."""

    def __init__(self, cls_name: str, op_path: str, suggestion: str = ""):
        msg = f"Unsupported class {cls_name} at {op_path!r}"
        if suggestion:
            msg += f". {suggestion}"
        super().__init__(msg)
        self.cls_name = cls_name
        self.op_path = op_path
