"""VldmEstimatorMixin: state-dict + init-params hooks for sklearn-style ops."""

from __future__ import annotations

from typing import Any, ClassVar, TypeVar

import numpy as np

from sklearn_tabpfn_ext.exceptions import OpStateError

# Self-like bound for the _from_state_dict restoration hook: ``obj`` is an
# instance of the class the classmethod is called on, so subclasses may annotate
# it with their own concrete type without an LSP contravariance violation. (Uses
# a TypeVar rather than typing.Self, which is 3.11+ / requires typing_extensions.)
_E = TypeVar("_E", bound="VldmEstimatorMixin")


class VldmEstimatorMixin:
    """Mix into any sklearn-compatible estimator that participates in vldm artifacts.

    Subclasses MUST declare:
      - _state_keys: tuple of fitted attribute names persisted in state.npz
      - _init_param_keys: tuple of __init__ kwarg names persisted in pipeline.json

    Subclasses MAY declare:
      - _child_attrs: tuple of fitted attribute names whose value is itself a vldm
        operator (i.e. a VldmEstimatorMixin instance). codec.save/load handles these
        recursively: each child is serialised as a child OpSpec and restored via
        _build_op before transform() is called. Order matches the declaration order.
        Children in _child_attrs are NOT included in _state_keys (they are not
        plain numpy arrays).
    """

    _state_keys: ClassVar[tuple[str, ...]] = ()
    _init_param_keys: ClassVar[tuple[str, ...]] = ()
    _child_attrs: ClassVar[tuple[str, ...]] = ()
    # _nan_init_param_keys: init param names whose value may be float NaN.
    # JSON does not support NaN (it serialises to null / None). codec.load uses
    # this list to convert None back to np.nan before calling cls(**init_params).
    _nan_init_param_keys: ClassVar[tuple[str, ...]] = ()
    # Forward-compat guard: ops that are NEW since a given reader version set this
    # True so their serialised artifacts always carry serialised_child_attrs,
    # which an older (extra="forbid") reader rejects at load. See spec §6.
    _forward_guard: ClassVar[bool] = False

    def _to_state_dict(self) -> dict[str, np.ndarray]:
        """Extract every fitted attribute named in _state_keys as ndarray."""
        out: dict[str, np.ndarray] = {}
        for key in self._state_keys:
            value = getattr(self, key)  # AttributeError if not fitted -- intended
            out[key] = np.asarray(value)
        return out

    @classmethod
    def _from_state_dict(cls: type[_E], obj: _E, state: dict[str, np.ndarray]) -> None:
        """Restore fitted attributes onto an already-constructed instance.

        Override when sklearn parent requires special restoration (e.g.
        QuantileTransformer needs n_quantiles_ alongside quantiles_).
        """
        for key in cls._state_keys:
            if key not in state:
                raise OpStateError(
                    op_path=type(obj).__qualname__,
                    key=key,
                    reason="missing in state.npz",
                )
            value = state[key]
            # 0-d ndarray -> python scalar for ints/floats so sklearn internals are happy.
            if value.shape == () and value.dtype.kind in "iuf":
                value = value.item()
            setattr(obj, key, value)

    def _init_params_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._init_param_keys}
