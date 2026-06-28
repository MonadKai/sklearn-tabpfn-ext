"""Drop columns with zero variance over the fit set.

Mirrors the semantics of ``tabpfn.preprocessing.steps.RemoveConstantFeaturesStep``
without importing tabpfn. Behaves like an sklearn transformer: ``fit`` learns
``sel_`` (a boolean mask of "non-constant" columns); ``transform`` applies it.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.constant_filter.ConstantFeatureFilter")
class ConstantFeatureFilter(VldmEstimatorMixin, BaseEstimator, TransformerMixin):
    """Selects columns that are not constant in the fit data.

    Parameters
    ----------
    keep_constant : bool, default False
        If True, keep all columns regardless. Useful for debugging / round-trip
        tests.

    Attributes
    ----------
    sel_ : ndarray of shape (n_features_in_,), dtype=bool
        ``True`` for columns kept, ``False`` for constant columns dropped.
    n_features_in_ : int
    n_features_out_ : int
    """

    _state_keys = ("sel_", "n_features_in_", "n_features_out_")
    _init_param_keys = ()

    def __init__(self, *, keep_constant: bool = False) -> None:
        self.keep_constant = keep_constant

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        if self.keep_constant or X.shape[0] <= 1:
            self.sel_ = np.ones(X.shape[1], dtype=bool)
        else:
            # Column is non-constant if any element differs from the first row.
            self.sel_ = np.any(X[0:1] != X, axis=0)
        self.n_features_out_ = int(self.sel_.sum())
        return self

    def transform(self, X):
        if not hasattr(self, "sel_"):
            raise RuntimeError("ConstantFeatureFilter not fitted")
        X = np.asarray(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return X[:, self.sel_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        else:
            input_features = np.asarray(input_features)
        return input_features[self.sel_]
