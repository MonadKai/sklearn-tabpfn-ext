"""Round-trip tests for SquashingScaler / PolynomialFeaturesAdder / DifferentiableZNorm."""

from __future__ import annotations

import numpy as np

from sklearn_tabpfn_ext.differentiable_znorm import DifferentiableZNorm
from sklearn_tabpfn_ext.polynomial_features import PolynomialFeaturesAdder
from sklearn_tabpfn_ext.squashing_scaler import SquashingScaler


def _roundtrip(orig, X):
    state = orig._to_state_dict()
    init = orig._init_params_dict()
    new = type(orig)(**init)
    type(orig)._from_state_dict(new, state)
    np.testing.assert_allclose(new.transform(X), orig.transform(X), atol=1e-12)


def test_differentiable_znorm():
    rng = np.random.default_rng(0)
    n = 50
    X = rng.normal(size=(n, 4))
    z = DifferentiableZNorm().fit(X)
    out = z.transform(X)
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-10)
    # std uses ddof=1 (matches TabPFN's torch.Tensor.std default);
    # post-transform numpy-default (ddof=0) std on the same data is sqrt((n-1)/n).
    expected_post_std = np.sqrt((n - 1) / n)
    np.testing.assert_allclose(out.std(axis=0), expected_post_std, atol=1e-10)
    _roundtrip(z, X)


def test_squashing_scaler():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 3))
    s = SquashingScaler().fit(X)
    out = s.transform(X)
    # Output should be roughly bounded due to squashing.
    assert np.all(np.isfinite(out))
    _roundtrip(s, X)


def test_polynomial_features_adder():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    p = PolynomialFeaturesAdder(max_features=5, random_state=0).fit(X)
    out = p.transform(X)
    assert out.shape[1] == X.shape[1] + p.max_features  # original + appended
    _roundtrip(p, X)
