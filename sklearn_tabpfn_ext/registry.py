"""Operator registry for vldm preprocessing artifacts."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from sklearn_tabpfn_ext.exceptions import UnknownOperatorError

T = TypeVar("T", bound=type)

OPERATOR_REGISTRY: dict[str, type] = {}
OP_ALIASES: dict[str, str] = {}  # legacy op_id -> canonical op_id


def register(op_id: str) -> Callable[[T], T]:
    """Class decorator: register `cls` under the fully-qualified op_id."""

    def deco(cls: T) -> T:
        if op_id in OPERATOR_REGISTRY:
            raise ValueError(f"Operator already registered: {op_id}")
        OPERATOR_REGISTRY[op_id] = cls
        cls._canonical_op_id = op_id  # NEW: stable wire-format id, decoupled from __module__
        return cls

    return deco


def get(op_id: str) -> type:
    canonical = OP_ALIASES.get(op_id, op_id)
    cls = OPERATOR_REGISTRY.get(canonical)
    if cls is None:
        raise UnknownOperatorError(op_id, available=sorted(OPERATOR_REGISTRY))
    return cls
