"""vldm equivalent of tabpfn.preprocessing.steps.DifferentiableZNormStep.

Algorithm: per-column mean and ddof=1 std (matching torch.Tensor.std default),
then z = (X - mean) / std. No epsilon guard is added; columns with zero std
will produce NaN/inf, matching the reference behaviour.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.differentiable_znorm.DifferentiableZNorm")
class DifferentiableZNorm(VldmEstimatorMixin, BaseEstimator, TransformerMixin):
    """Per-feature z-normalisation.

    Matches DifferentiableZNormStep in TabPFN: mean and std (ddof=1) are
    computed column-wise at fit time and applied at transform time.

    Parameters
    ----------
    None.  No configurable init parameters.
    """

    _state_keys = ("mean_", "std_", "n_features_in_")
    _init_param_keys = ()

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        # Matches torch.Tensor.std default (ddof=1, Bessel correction); needed
        # for parity with TabPFN's DifferentiableZNormStep.
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=1)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_
