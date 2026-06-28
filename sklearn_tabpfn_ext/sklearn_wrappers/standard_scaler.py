"""vldm wrapper for sklearn.preprocessing.StandardScaler."""

from __future__ import annotations

from sklearn.preprocessing import StandardScaler as _Sk

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.exceptions import UnsupportedConversionError
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.sklearn_wrappers.standard_scaler.StandardScaler")
class StandardScaler(VldmEstimatorMixin, _Sk):
    _state_keys = ("mean_", "scale_", "var_", "n_samples_seen_", "n_features_in_")
    _init_param_keys = ("with_mean", "with_std", "copy")

    @classmethod
    def from_sklearn(cls, sk: _Sk) -> StandardScaler:
        """Build a fully-fitted StandardScaler from a fitted sklearn StandardScaler."""
        init = {k: getattr(sk, k) for k in cls._init_param_keys if hasattr(sk, k)}
        obj = cls(**init)
        for k in cls._state_keys:
            if not hasattr(sk, k):
                raise UnsupportedConversionError(
                    type(sk).__qualname__,
                    "from_sklearn",
                    f"missing fitted attribute {k!r}; was the object fit?",
                )
            setattr(obj, k, getattr(sk, k))
        return obj
