"""vldm equivalent of tabpfn.preprocessing.steps.NanHandlingPolynomialFeaturesStep.

Algorithm (ported from TabPFN reference):

1. Fit a StandardScaler(with_mean=False) on the training data.
2. At fit time, determine how many polynomial pair features to generate:
       n_polynomials = min(max_features, n_choose_2 + n)  (if max_features set)
3. Randomly assign poly_factor_1_idx from [0, n_features) with replacement.
4. For each i, assign poly_factor_2_idx[i] from indices >= poly_factor_1_idx[i]
   that have not yet been paired with poly_factor_1_idx[i] (de-dup within same
   factor-1 group).  If no candidate remains, resample poly_factor_1_idx[i].
5. At transform time, standardise X then hstack(X, X[:, idx1] * X[:, idx2]).

No NaN-to-zero replacement is applied (the reference does not do that).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.polynomial_features.PolynomialFeaturesAdder")
class PolynomialFeaturesAdder(VldmEstimatorMixin, BaseEstimator, TransformerMixin):
    """Append random pairwise polynomial features (standardised-input products).

    Matches NanHandlingPolynomialFeaturesStep from the TabPFN reference.

    Parameters
    ----------
    max_features : int or None, default=None
        Maximum number of polynomial features to append.  None means use all
        possible pairs (n*(n-1)//2 + n).
    random_state : int or np.random.Generator or None, default=None
        RNG seed / generator.
    """

    _state_keys = (
        "poly_factor_1_idx_",
        "poly_factor_2_idx_",
        # StandardScaler(with_mean=False) fitted state
        "std_scale_",  # scaler.scale_  (var_)
        "n_features_in_",
    )
    _init_param_keys = ("max_features", "random_state")

    def __init__(
        self,
        max_features: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.max_features = max_features
        self.random_state = random_state

    # ------------------------------------------------------------------

    @staticmethod
    def _make_rng(seed) -> np.random.Generator:
        if isinstance(seed, np.random.Generator):
            return seed
        return np.random.default_rng(seed)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        if n_samples == 0 or n_features == 0:
            self.poly_factor_1_idx_ = np.empty(0, dtype=np.int64)
            self.poly_factor_2_idx_ = np.empty(0, dtype=np.int64)
            self.std_scale_ = np.ones(n_features, dtype=np.float64)
            self.n_features_in_ = n_features
            return self

        # Fit scaler (with_mean=False, matching the reference). Only scale_ is
        # needed here; transform() recomputes the standardised matrix.
        scaler = StandardScaler(with_mean=False)
        scaler.fit(X)
        self.std_scale_ = scaler.scale_.astype(np.float64)

        # Compute number of polynomials.
        n_max = (n_features * (n_features - 1)) // 2 + n_features
        n_polynomials = min(self.max_features, n_max) if self.max_features is not None else n_max

        rng = self._make_rng(self.random_state)

        # Replicate the TabPFN reference pair-selection loop.
        poly_factor_1_idx = rng.choice(
            np.arange(0, n_features),
            size=n_polynomials,
            replace=True,
        )
        poly_factor_2_idx = np.full(n_polynomials, -1, dtype=np.int64)

        for i in range(n_polynomials):
            while poly_factor_2_idx[i] == -1:
                factor1 = poly_factor_1_idx[i]
                # Indices already assigned as factor-2 for the same factor-1 value.
                used = poly_factor_2_idx[poly_factor_1_idx == factor1]
                # Candidates: index >= factor1 and not already used.
                candidates = sorted(set(range(factor1, n_features)) - set(used.tolist()))
                if len(candidates) == 0:
                    # Resample factor-1 and retry.
                    poly_factor_1_idx[i] = rng.choice(np.arange(0, n_features))
                    continue
                poly_factor_2_idx[i] = rng.choice(candidates)

        self.poly_factor_1_idx_ = poly_factor_1_idx.astype(np.int64)
        self.poly_factor_2_idx_ = poly_factor_2_idx.astype(np.int64)
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        check_is_fitted(self, ["poly_factor_1_idx_", "poly_factor_2_idx_", "std_scale_"])
        X = np.asarray(X, dtype=np.float64)

        if X.shape[1] == 0 or len(self.poly_factor_1_idx_) == 0:
            return X

        # Standardise (with_mean=False: divide by scale only).
        X_std = X / self.std_scale_

        poly = X_std[:, self.poly_factor_1_idx_] * X_std[:, self.poly_factor_2_idx_]
        return np.hstack((X_std, poly))
