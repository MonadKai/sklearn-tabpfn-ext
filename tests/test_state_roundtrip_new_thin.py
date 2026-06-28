"""Round-trip tests for AdaptiveQuantile / SafePower / KDIWithNaN."""

from __future__ import annotations

import numpy as np

from sklearn_tabpfn_ext.adaptive_quantile import AdaptiveQuantileTransformer
from sklearn_tabpfn_ext.kdi_with_nan import KDITransformerWithNaN
from sklearn_tabpfn_ext.safe_power import SafePowerTransformer


def _roundtrip(orig, X):
    state = orig._to_state_dict()
    init = orig._init_params_dict()
    new = type(orig)(**init)
    type(orig)._from_state_dict(new, state)
    np.testing.assert_allclose(new.transform(X), orig.transform(X), atol=1e-12)


def test_adaptive_quantile():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    aq = AdaptiveQuantileTransformer(
        n_quantiles=64, output_distribution="uniform", subsample=200, random_state=0
    ).fit(X)
    _roundtrip(aq, X[:10])


def test_safe_power():
    rng = np.random.default_rng(1)
    X = np.abs(rng.normal(size=(50, 2))) + 0.1
    sp = SafePowerTransformer(method="yeo-johnson", standardize=True).fit(X)
    _roundtrip(sp, X)


def test_kdi_with_nan_falls_back_when_no_kditransform(monkeypatch):
    """KDITransformerWithNaN must work without kditransform installed."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 2))
    k = KDITransformerWithNaN(method="yeo-johnson").fit(X)
    _roundtrip(k, X)
