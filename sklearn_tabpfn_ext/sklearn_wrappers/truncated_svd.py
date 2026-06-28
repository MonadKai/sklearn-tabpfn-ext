"""vldm wrapper for sklearn.decomposition.TruncatedSVD."""

from __future__ import annotations

from sklearn.decomposition import TruncatedSVD as _Sk

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.exceptions import UnsupportedConversionError
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.sklearn_wrappers.truncated_svd.TruncatedSVD")
class TruncatedSVD(VldmEstimatorMixin, _Sk):
    _state_keys = (
        "components_",
        "explained_variance_",
        "explained_variance_ratio_",
        "singular_values_",
        "n_features_in_",
    )
    _init_param_keys = (
        "n_components",
        "algorithm",
        "n_iter",
        "n_oversamples",
        "power_iteration_normalizer",
        "random_state",
        "tol",
    )

    @classmethod
    def from_sklearn(cls, sk: _Sk) -> TruncatedSVD:
        """Build a fully-fitted TruncatedSVD from a fitted sklearn TruncatedSVD."""
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
