"""Permute / rotate columns of the feature matrix.

Mirrors ``tabpfn.preprocessing.steps.ShuffleFeaturesStep``. Two modes:

- ``"shuffle"``: arbitrary permutation drawn at ``fit`` time.
- ``"rotate"``: cyclic rotation by ``shuffle_index`` columns.

The fitted ``index_permutation_`` is just a length-``n_features`` int array;
``transform`` is a single fancy-index lookup.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.shuffle.FeatureShuffler")
class FeatureShuffler(VldmEstimatorMixin, BaseEstimator, TransformerMixin):
    """Permutes columns according to a fitted index permutation.

    Parameters
    ----------
    shuffle_method : {"shuffle", "rotate"}, default "shuffle"
    shuffle_index : int, default 0
        Used as the rotation amount in ``"rotate"`` mode and as a seed offset
        in ``"shuffle"`` mode.
    random_state : int or None
        Used to draw the permutation in ``"shuffle"`` mode.

    Attributes
    ----------
    index_permutation_ : ndarray of shape (n_features_in_,)
    n_features_in_ : int
    """

    _state_keys = ("index_permutation_", "n_features_in_")
    _init_param_keys = ("shuffle_method", "shuffle_index")

    def __init__(
        self,
        *,
        shuffle_method: Literal["shuffle", "rotate"] = "shuffle",
        shuffle_index: int = 0,
        random_state: int | None = None,
    ) -> None:
        self.shuffle_method = shuffle_method
        self.shuffle_index = shuffle_index
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X)
        n = X.shape[1]
        self.n_features_in_ = n
        if self.shuffle_method == "shuffle":
            seed = self.random_state if self.random_state is not None else self.shuffle_index
            rng = np.random.default_rng(seed)
            self.index_permutation_ = rng.permutation(n)
        elif self.shuffle_method == "rotate":
            shift = int(self.shuffle_index) % max(n, 1)
            self.index_permutation_ = np.concatenate([np.arange(shift, n), np.arange(0, shift)])
        else:
            raise ValueError(f"unknown shuffle_method: {self.shuffle_method!r}")
        self.index_permutation_ = self.index_permutation_.astype(np.int64, copy=False)
        return self

    def transform(self, X):
        if not hasattr(self, "index_permutation_"):
            raise RuntimeError("FeatureShuffler not fitted")
        X = np.asarray(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return X[:, self.index_permutation_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        else:
            input_features = np.asarray(input_features)
        return input_features[self.index_permutation_]
