"""vldm wrapper for sklearn.impute.SimpleImputer."""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer as _Sk

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.exceptions import OpStateError, UnsupportedConversionError
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.sklearn_wrappers.simple_imputer.SimpleImputer")
class SimpleImputer(VldmEstimatorMixin, _Sk):
    # Note: indicator_ is omitted from _state_keys because the audit showed add_indicator=False
    # only. When add_indicator=False (default), indicator_ is None, which would fail serialization
    # via np.asarray(None) producing a 0-d object array that np.load(allow_pickle=False) rejects.
    # Phase A scope is add_indicator=False cases only; if add_indicator=True is needed later,
    # override _to_state_dict / _from_state_dict to handle the MaskMaker object.
    _state_keys = ("statistics_", "n_features_in_", "_fit_dtype")
    _init_param_keys = (
        "missing_values",
        "strategy",
        "fill_value",
        "copy",
        "add_indicator",
        "keep_empty_features",
    )
    # missing_values is commonly np.nan, which JSON serialises as null (None).
    # Declare it here so codec.load converts None -> np.nan before construction.
    _nan_init_param_keys = ("missing_values",)

    def _to_state_dict(self) -> dict[str, np.ndarray]:
        """Override to serialize _fit_dtype as a fixed-width unicode string.

        We must NOT use dtype=object here: numpy stores object arrays via pickle,
        which np.load(allow_pickle=False) at codec.load (Task 11) would reject.
        Use a fixed-width unicode dtype (str) instead — it serialises cleanly.
        Also guard add_indicator=True (out of Phase A scope per audit).
        """
        if getattr(self, "indicator_", None) is not None:
            raise UnsupportedConversionError(
                "sklearn.impute.SimpleImputer(add_indicator=True)",
                op_path=type(self).__qualname__,
                suggestion=(
                    "Phase A audit showed add_indicator=False only; "
                    "extend _state_keys + _to/_from_state_dict to support "
                    "the MissingIndicator child."
                ),
            )
        state = super()._to_state_dict()
        if "_fit_dtype" in state:
            # Fixed-width unicode 0-d array (no pickle).
            state["_fit_dtype"] = np.asarray(str(self._fit_dtype))
        return state

    @classmethod
    def _from_state_dict(cls, obj: SimpleImputer, state: dict[str, np.ndarray]) -> None:
        """Override to deserialize _fit_dtype from string representation."""
        # Handle regular state keys
        for key in cls._state_keys:
            if key == "_fit_dtype":
                # Special handling for _fit_dtype: deserialize from string
                if key not in state:
                    raise OpStateError(
                        op_path=cls.__qualname__,
                        key=key,
                        reason="missing in state.npz",
                    )
                dtype_str = str(state[key].item())  # 0-d unicode array
                setattr(obj, key, np.dtype(dtype_str))
            else:
                # Regular attributes
                if key not in state:
                    raise OpStateError(
                        op_path=cls.__qualname__,
                        key=key,
                        reason="missing in state.npz",
                    )
                value = state[key]
                # 0-d ndarray -> python scalar for ints/floats so sklearn internals are happy.
                if value.shape == () and value.dtype.kind in "iuf":
                    value = value.item()
                setattr(obj, key, value)
