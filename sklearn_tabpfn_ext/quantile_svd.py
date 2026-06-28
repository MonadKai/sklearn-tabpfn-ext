"""Quantile-uniform reshaping plus optional SVD reduction.

This estimator is a thin sklearn-compatible wrapper around a *fitted*
``sklearn.pipeline.Pipeline`` (typically QuantileTransformer +
TruncatedSVD). It exists for two reasons:

1. To give the conversion script (``scripts/extract_model.py``) a stable
   entry point for storing tabpfn's
   ``ReshapeFeatureDistributionsStep.transformer_`` — the underlying
   ``sklearn.Pipeline`` is reused as-is (sklearn primitives, not
   reimplemented), but vldm wraps it so we can record extra fitted state
   like ``subsampled_features_`` and ``append_to_original``.
2. To support sklearn's ``Pipeline`` composition seamlessly: this class can
   appear directly in a ``Pipeline`` alongside :class:`ConstantFeatureFilter`
   etc.

We do **not** reimplement QuantileTransformer or TruncatedSVD — those are
stable sklearn primitives. The conversion script extracts the fitted
``transformer_`` Pipeline directly.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.quantile_svd.QuantileSVDReshaper")
class QuantileSVDReshaper(VldmEstimatorMixin, BaseEstimator, TransformerMixin):
    """Wraps a fitted sklearn Pipeline (QuantileTransformer + optional TruncatedSVD).

    Parameters
    ----------
    transformer : sklearn.pipeline.Pipeline or None
        The fitted pipeline. Required at ``transform`` time; optional at
        construction time so the object can be ``__init__``'d empty before
        ``fit`` is called.
    subsampled_features : ndarray or None
        Indices into the input columns selecting which columns get reshaped.
        ``None`` means use all columns.
    append_to_original : bool, default False
        Recorded for inspection only. When wrapping a tabpfn-fitted
        ``transformer_`` the "append originals" behaviour is **already baked
        into** the inner ColumnTransformer (a passthrough branch); we do not
        re-append here. The flag is preserved so round-tripping the saved
        artifact retains the metadata.

    Attributes
    ----------
    transformer_ : sklearn.pipeline.Pipeline
    subsampled_features_ : ndarray of int64 or None
    n_features_in_ : int
    n_features_out_ : int
    """

    _state_keys = ("subsampled_features_", "n_features_in_", "n_features_out_")
    _init_param_keys = ("append_to_original",)
    # transformer_ is a fitted vldm operator (SequentialPipeline post-translation);
    # codec.save/load handles it recursively via _child_attrs rather than state.npz.
    _child_attrs = ("transformer_",)

    def __init__(
        self,
        *,
        transformer: Pipeline | None = None,
        subsampled_features: np.ndarray | None = None,
        append_to_original: bool = False,
    ) -> None:
        self.transformer = transformer
        self.subsampled_features = subsampled_features
        self.append_to_original = append_to_original

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]

        if self.subsampled_features is not None:
            sub = np.asarray(self.subsampled_features, dtype=np.int64)
            X_sub = X[:, sub]
            self.subsampled_features_ = sub
        else:
            X_sub = X
            self.subsampled_features_ = None

        if self.transformer is None:
            from sklearn.preprocessing import QuantileTransformer

            self.transformer_ = Pipeline(
                steps=[("quantile", QuantileTransformer(output_distribution="uniform"))]
            )
            self.transformer_.fit(X_sub)
        else:
            self.transformer_ = self.transformer
            self.transformer_.fit(X_sub)

        out = self.transformer_.transform(X_sub[:1])
        self.n_features_out_ = int(out.shape[1])
        return self

    def transform(self, X):
        if not hasattr(self, "transformer_"):
            raise RuntimeError("QuantileSVDReshaper not fitted")
        X = np.asarray(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        X_sub = X[:, self.subsampled_features_] if self.subsampled_features_ is not None else X
        return self.transformer_.transform(X_sub)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"reshape_{i}" for i in range(self.n_features_out_)], dtype=object)
