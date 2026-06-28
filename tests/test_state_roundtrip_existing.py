"""State round-trip tests for the 5 migrated vldm operators.

These tests verify that _to_state_dict / _from_state_dict preserves
transform output exactly, without going through the codec (which is
not yet built).  Where fit() is not convenient, fitted state is
assigned directly.
"""

from __future__ import annotations

import numpy as np

from sklearn_tabpfn_ext.categorical import (
    CategoricalOrdinalEncoder,
    IdentityCategoricalEncoder,
)
from sklearn_tabpfn_ext.constant_filter import ConstantFeatureFilter
from sklearn_tabpfn_ext.fingerprint import FingerprintFeatureAdder
from sklearn_tabpfn_ext.quantile_svd import QuantileSVDReshaper
from sklearn_tabpfn_ext.shuffle import FeatureShuffler


def _roundtrip(orig, X):
    """Generic round-trip helper: serialize, reconstruct, assert equal."""
    state = orig._to_state_dict()
    init = orig._init_params_dict()
    new = type(orig)(**init)
    type(orig)._from_state_dict(new, state)
    np.testing.assert_array_equal(new.transform(X), orig.transform(X))


def test_constant_filter_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    X[:, 2] = 1.0  # constant column
    cf = ConstantFeatureFilter().fit(X)
    _roundtrip(cf, X)


def test_fingerprint_roundtrip():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 4))
    fp = FingerprintFeatureAdder(salt=42)
    fp.rnd_salt_ = 42
    fp.n_features_in_ = X.shape[1]
    _roundtrip(fp, X)


def test_shuffle_roundtrip():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 6))
    sh = FeatureShuffler(shuffle_method="rotate", shuffle_index=0)
    sh.index_permutation_ = np.array([1, 0, 2, 3, 4, 5], dtype=np.int64)
    sh.n_features_in_ = X.shape[1]
    _roundtrip(sh, X)


def test_quantile_svd_roundtrip():
    """Simulate a fitted QuantileSVDReshaper by setting state directly.

    We use a concrete subsampled_features_ (not None) so that the ndarray
    round-trip through _to_state_dict / _from_state_dict is well-typed.
    The None-subsampling path is exercised separately by fit() in unit tests;
    here we focus on the mixin contract.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import QuantileTransformer

    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 4))
    sub_idx = np.array([0, 1, 3], dtype=np.int64)
    X_sub = X[:, sub_idx]

    # Build and fit the inner pipeline on the sub-selected columns
    inner = Pipeline(
        steps=[("qt", QuantileTransformer(output_distribution="uniform", random_state=0))]
    )
    inner.fit(X_sub)

    # Manually assign fitted state (simulating what extract_model.py would do)
    qsvd = QuantileSVDReshaper(append_to_original=False)
    qsvd.transformer_ = inner
    qsvd.subsampled_features_ = sub_idx
    qsvd.n_features_in_ = X.shape[1]
    qsvd.n_features_out_ = int(inner.transform(X_sub[:1]).shape[1])

    # Round-trip: only the three declared _state_keys go through state dict;
    # transformer_ is a child and NOT in _state_keys.
    state = qsvd._to_state_dict()
    assert set(state.keys()) == {"subsampled_features_", "n_features_in_", "n_features_out_"}

    # Reconstruct: init params only contain append_to_original.
    init = qsvd._init_params_dict()
    assert set(init.keys()) == {"append_to_original"}

    new = QuantileSVDReshaper(**init)
    QuantileSVDReshaper._from_state_dict(new, state)
    # Manually restore the child transformer_ (codec would do this).
    new.transformer_ = inner

    np.testing.assert_array_equal(new.transform(X), qsvd.transform(X))


def test_identity_categorical_encoder_roundtrip():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(15, 3))
    enc = IdentityCategoricalEncoder().fit(X)
    _roundtrip(enc, X)


def test_categorical_ordinal_encoder_roundtrip_no_transformer():
    """CategoricalOrdinalEncoder with column_transformer=None (pure pass-through).

    column_transformer_ is a child handled by the codec; only n_features_in_
    goes through _state_keys at this level.
    """
    rng = np.random.default_rng(5)
    X = rng.normal(size=(15, 3))

    enc = CategoricalOrdinalEncoder()
    enc.n_features_in_ = X.shape[1]
    enc.column_transformer_ = None
    enc.random_mappings_ = {}

    state = enc._to_state_dict()
    assert "n_features_in_" in state

    init = enc._init_params_dict()
    new = CategoricalOrdinalEncoder(**init)
    CategoricalOrdinalEncoder._from_state_dict(new, state)
    # Restore child state manually (codec responsibility).
    new.column_transformer_ = None
    new.random_mappings_ = {}

    np.testing.assert_array_equal(new.transform(X), enc.transform(X))
