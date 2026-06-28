"""Codec round-trip tests for FingerprintFeatureAdder v63 and v64.

Verifies that _to_state_dict / _init_params_dict / _from_state_dict
correctly persists and restores algo-appropriate state keys.
"""

from __future__ import annotations

import numpy as np
import pytest

from sklearn_tabpfn_ext.exceptions import OpStateError
from sklearn_tabpfn_ext.fingerprint import FingerprintFeatureAdder


def _clone_via_roundtrip(fa):
    """Round-trip: serialize, construct new instance, restore state."""
    state = fa._to_state_dict()
    init = fa._init_params_dict()
    clone = FingerprintFeatureAdder(**init)
    FingerprintFeatureAdder._from_state_dict(clone, state)
    return clone


class TestV63Roundtrip:
    def test_rnd_salt_restored(self):
        X = np.arange(12.0).reshape(3, 4)
        fa = FingerprintFeatureAdder(salt=42).fit(X)
        clone = _clone_via_roundtrip(fa)
        assert clone.rnd_salt_ == 42

    def test_n_features_in_restored(self):
        X = np.arange(12.0).reshape(3, 4)
        fa = FingerprintFeatureAdder(salt=42).fit(X)
        clone = _clone_via_roundtrip(fa)
        assert clone.n_features_in_ == 4

    def test_transform_equal(self):
        X = np.random.default_rng(1).normal(size=(20, 5))
        fa = FingerprintFeatureAdder(salt=99).fit(X)
        clone = _clone_via_roundtrip(fa)
        np.testing.assert_array_equal(clone.transform(X), fa.transform(X))

    def test_rnd_salt_in_state_keys(self):
        X = np.arange(12.0).reshape(3, 4)
        fa = FingerprintFeatureAdder(salt=1).fit(X)
        state = fa._to_state_dict()
        assert "rnd_salt_" in state
        assert "n_cells_" not in state


class TestV64Roundtrip:
    def test_n_cells_restored(self):
        X = np.arange(12.0).reshape(3, 4)
        fa = FingerprintFeatureAdder(algo="v64").fit(X)
        clone = _clone_via_roundtrip(fa)
        assert clone.n_cells_ == 12

    def test_n_features_in_restored(self):
        X = np.arange(12.0).reshape(3, 4)
        fa = FingerprintFeatureAdder(algo="v64").fit(X)
        clone = _clone_via_roundtrip(fa)
        assert clone.n_features_in_ == 4

    def test_no_rnd_salt_in_state(self):
        X = np.arange(12.0).reshape(3, 4)
        fa = FingerprintFeatureAdder(algo="v64").fit(X)
        state = fa._to_state_dict()
        assert "rnd_salt_" not in state
        assert "n_cells_" in state

    def test_transform_equal(self):
        X = np.random.default_rng(2).normal(size=(20, 5))
        fa = FingerprintFeatureAdder(algo="v64").fit(X)
        clone = _clone_via_roundtrip(fa)
        np.testing.assert_array_equal(clone.transform(X), fa.transform(X))

    def test_algo_preserved_in_init_params(self):
        X = np.arange(12.0).reshape(3, 4)
        fa = FingerprintFeatureAdder(algo="v64").fit(X)
        init = fa._init_params_dict()
        assert init["algo"] == "v64"


class TestMissingAlgoDefaultsToV63:
    """Loading a legacy artifact (no 'algo' in init-params) round-trips as v63."""

    def test_missing_algo_in_init_params_defaults_to_v63(self):
        X = np.arange(12.0).reshape(3, 4)
        fa_orig = FingerprintFeatureAdder(salt=5).fit(X)
        state = fa_orig._to_state_dict()

        # Simulate legacy init-params dict without 'algo'
        init_without_algo = {"salt": 5}
        clone = FingerprintFeatureAdder(**init_without_algo)
        FingerprintFeatureAdder._from_state_dict(clone, state)

        assert clone.algo == "v63"
        assert clone.rnd_salt_ == 5
        np.testing.assert_array_equal(clone.transform(X), fa_orig.transform(X))


class TestFromStateDictErrors:
    def test_v63_missing_rnd_salt_raises(self):
        X = np.arange(12.0).reshape(3, 4)
        fa = FingerprintFeatureAdder(salt=1).fit(X)
        state = fa._to_state_dict()
        del state["rnd_salt_"]
        clone = FingerprintFeatureAdder(salt=1)
        with pytest.raises(OpStateError):
            FingerprintFeatureAdder._from_state_dict(clone, state)

    def test_v64_missing_n_cells_raises(self):
        X = np.arange(12.0).reshape(3, 4)
        fa = FingerprintFeatureAdder(algo="v64").fit(X)
        state = fa._to_state_dict()
        del state["n_cells_"]
        clone = FingerprintFeatureAdder(algo="v64")
        with pytest.raises(OpStateError):
            FingerprintFeatureAdder._from_state_dict(clone, state)
