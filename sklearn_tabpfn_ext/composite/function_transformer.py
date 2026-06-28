"""vldm FunctionTransformer — named-function enum, never an arbitrary callable.

We intentionally reject callables: they cannot be structurally serialised in
pipeline.json. The named-function registry below covers everything the
Phase A audit found in tabpfn ckpts. To add a new function, register it
here AND in scripts/tabpfn_translator.py's qualname map.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.registry import register


def _identity(X: np.ndarray) -> np.ndarray:
    return X


def _inf_to_nan(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X, nan=np.nan, neginf=np.nan, posinf=np.nan)


def _exp_minus_1(X: np.ndarray) -> np.ndarray:
    return np.expm1(X)


_NAMED_FUNCS = {
    "identity": _identity,
    "passthrough": _identity,  # alias
    "inf_to_nan": _inf_to_nan,
    "exp_minus_1": _exp_minus_1,
}


@register("vldm.preprocessing.composite.function_transformer.FunctionTransformer")
class FunctionTransformer(VldmEstimatorMixin, BaseEstimator, TransformerMixin):
    _state_keys = ()
    _init_param_keys = ("func",)

    def __init__(self, func: str = "identity"):
        if not isinstance(func, str) or func not in _NAMED_FUNCS:
            raise ValueError(
                f"FunctionTransformer.func must be one of {sorted(_NAMED_FUNCS)}; got {func!r}"
            )
        self.func = func

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1] if X.ndim == 2 else 1
        return self

    def transform(self, X):
        return _NAMED_FUNCS[self.func](np.asarray(X, dtype=np.float64))
