"""vldm wrapper for sklearn.decomposition.TruncatedSVD."""

from __future__ import annotations

from sklearn.decomposition import TruncatedSVD as _Sk

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
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
