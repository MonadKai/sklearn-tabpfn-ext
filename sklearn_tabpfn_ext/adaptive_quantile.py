"""vldm equivalent of tabpfn.preprocessing.steps.AdaptiveQuantileTransformer.

We subclass sklearn.preprocessing.QuantileTransformer; transform behavior is
inherited unchanged. The only purpose of this class is to provide a
non-tabpfn home for the fitted state so artifacts contain no tabpfn refs.
"""

from __future__ import annotations

from sklearn.preprocessing import QuantileTransformer

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.adaptive_quantile.AdaptiveQuantileTransformer")
class AdaptiveQuantileTransformer(VldmEstimatorMixin, QuantileTransformer):
    _state_keys = ("quantiles_", "references_", "n_features_in_", "n_quantiles_")
    _init_param_keys = (
        "n_quantiles",
        "output_distribution",
        "subsample",
        "random_state",
        "ignore_implicit_zeros",
    )
