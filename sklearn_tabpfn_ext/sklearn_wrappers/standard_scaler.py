"""vldm wrapper for sklearn.preprocessing.StandardScaler."""

from __future__ import annotations

from sklearn.preprocessing import StandardScaler as _Sk

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.sklearn_wrappers.standard_scaler.StandardScaler")
class StandardScaler(VldmEstimatorMixin, _Sk):
    _state_keys = ("mean_", "scale_", "var_", "n_samples_seen_", "n_features_in_")
    _init_param_keys = ("with_mean", "with_std", "copy")
