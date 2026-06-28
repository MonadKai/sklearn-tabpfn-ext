"""Round-trip tests for vldm.preprocessing.sklearn_wrappers.

For every wrapper class on §4.5 of the spec, add a small fitter+roundtrip test.
"""

from __future__ import annotations

import numpy as np


def _roundtrip(orig, X):
    state = orig._to_state_dict()
    init = orig._init_params_dict()
    new = type(orig)(**init)
    type(orig)._from_state_dict(new, state)
    np.testing.assert_allclose(new.transform(X), orig.transform(X), atol=1e-12)


def test_standard_scaler_wrapper_roundtrip():
    from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    s = StandardScaler().fit(X)
    _roundtrip(s, X)


def test_simple_imputer_wrapper_roundtrip():
    from sklearn_tabpfn_ext.sklearn_wrappers.simple_imputer import SimpleImputer

    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4))
    # Insert some NaN values
    X[0, 0] = np.nan
    X[5, 2] = np.nan
    X[10, 1] = np.nan
    si = SimpleImputer(strategy="mean").fit(X)
    _roundtrip(si, X)


def test_truncated_svd_wrapper_roundtrip():
    from sklearn_tabpfn_ext.sklearn_wrappers.truncated_svd import TruncatedSVD

    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 10))
    ts = TruncatedSVD(n_components=5, random_state=42).fit(X)
    _roundtrip(ts, X)


def test_simple_imputer_state_npz_safe(tmp_path):
    """SimpleImputer._fit_dtype must survive np.savez_compressed +
    np.load(allow_pickle=False) — codec.load uses the latter as a security gate.
    A naive dtype=object encoding would break this contract."""
    from sklearn_tabpfn_ext.sklearn_wrappers.simple_imputer import SimpleImputer

    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 4))
    X[0, 0] = np.nan
    si = SimpleImputer(strategy="mean").fit(X)
    state = si._to_state_dict()

    p = tmp_path / "state.npz"
    np.savez_compressed(p, **state)
    loaded = np.load(p, allow_pickle=False)
    assert "_fit_dtype" in loaded.files
    # Reconstruct; transform must still match.
    new = SimpleImputer(strategy="mean")
    SimpleImputer._from_state_dict(new, {k: loaded[k] for k in loaded.files})
    np.testing.assert_allclose(new.transform(X), si.transform(X), atol=1e-12)


def test_simple_imputer_rejects_add_indicator():
    """add_indicator=True is out of Phase A scope; _to_state_dict must raise."""
    import pytest

    from sklearn_tabpfn_ext.exceptions import UnsupportedConversionError
    from sklearn_tabpfn_ext.sklearn_wrappers.simple_imputer import SimpleImputer

    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 4))
    X[0, 0] = np.nan
    si = SimpleImputer(strategy="mean", add_indicator=True).fit(X)
    with pytest.raises(UnsupportedConversionError):
        si._to_state_dict()
