"""Tests for FingerprintFeatureAdder v64 algorithm variant.

Covers:
(a) default algo == "v63"
(b) v64 determinism, +1 column, originals preserved, values in [0,1]
(c) v63 path unchanged (salt=7 -> rnd_salt_==7, shape ok)
(d) oracle fixture byte-exact match
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sklearn_tabpfn_ext.fingerprint import FingerprintFeatureAdder

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fingerprint_v64_expected.json"


def test_default_algo_is_v63():
    fa = FingerprintFeatureAdder()
    assert fa.algo == "v63"


def test_v64_deterministic():
    X = np.random.default_rng(42).normal(size=(10, 5))
    fa1 = FingerprintFeatureAdder(algo="v64").fit(X)
    fa2 = FingerprintFeatureAdder(algo="v64").fit(X)
    out1 = fa1.transform(X)
    out2 = fa2.transform(X)
    np.testing.assert_array_equal(out1, out2)


def test_v64_appends_exactly_one_column():
    X = np.arange(20.0).reshape(4, 5)
    fa = FingerprintFeatureAdder(algo="v64").fit(X)
    out = fa.transform(X)
    assert out.shape == (4, 6)


def test_v64_originals_preserved():
    X = np.arange(20.0).reshape(4, 5)
    fa = FingerprintFeatureAdder(algo="v64").fit(X)
    out = fa.transform(X)
    np.testing.assert_array_equal(out[:, :5], X)


def test_v64_fingerprint_values_in_0_1():
    X = np.random.default_rng(7).normal(size=(50, 8))
    fa = FingerprintFeatureAdder(algo="v64").fit(X)
    out = fa.transform(X)
    fp = out[:, -1]
    assert np.all(fp >= 0.0)
    assert np.all(fp <= 1.0)


def test_v63_salt_stored():
    """v63 path: explicit salt -> rnd_salt_ == salt."""
    X = np.arange(12.0).reshape(3, 4)
    fa = FingerprintFeatureAdder(salt=7).fit(X)
    assert fa.rnd_salt_ == 7
    out = fa.transform(X)
    assert out.shape == (3, 5)


def test_v63_salt_none_draws_from_rng():
    """v63 path: salt=None draws a random salt."""
    X = np.arange(12.0).reshape(3, 4)
    fa = FingerprintFeatureAdder(algo="v63", random_state=0).fit(X)
    assert hasattr(fa, "rnd_salt_")
    assert 0 <= fa.rnd_salt_ < 2**16


def test_v64_matches_oracle_fixture():
    """Byte-exact match against oracle captured from tabpfn 6.4.1."""
    with open(_FIXTURE_PATH) as f:
        fixture = json.load(f)
    X = np.array(fixture["X"])
    expected_fp = np.array(fixture["fingerprint"])

    fa = FingerprintFeatureAdder(algo="v64").fit(X)
    out = fa.transform(X)
    actual_fp = out[:, -1]

    np.testing.assert_array_equal(actual_fp, expected_fp)


def test_v64_hash_is_float32_invariant():
    """The hash must use the float64 byte layout regardless of input dtype, so a
    float32-served row hashes identically to float64 (Gemini fix). Uses values
    exactly representable in float32 so the cast is lossless."""
    from sklearn_tabpfn_ext.fingerprint import (
        _float_hash_arr_v63,
        _float_hash_arr_v64,
    )

    row = np.array([1.5, 2.25, 3.125, 4.0])
    assert _float_hash_arr_v64(row.astype(np.float32), 12) == _float_hash_arr_v64(
        row.astype(np.float64), 12
    )
    assert _float_hash_arr_v63(row.astype(np.float32)) == _float_hash_arr_v63(
        row.astype(np.float64)
    )
