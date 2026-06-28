"""FunctionTransformer — named-function enum (identity / passthrough / inf_to_nan / exp_minus_1)."""

from __future__ import annotations

import numpy as np
import pytest

from sklearn_tabpfn_ext.composite.function_transformer import FunctionTransformer


def test_identity():
    ft = FunctionTransformer(func="identity")
    X = np.array([[1.0, 2.0]])
    np.testing.assert_array_equal(ft.fit_transform(X), X)


def test_passthrough_alias_for_identity():
    ft = FunctionTransformer(func="passthrough")
    X = np.array([[1.0, 2.0]])
    np.testing.assert_array_equal(ft.fit_transform(X), X)


def test_inf_to_nan():
    ft = FunctionTransformer(func="inf_to_nan")
    X = np.array([[1.0, np.inf, -np.inf, np.nan, 2.5]])
    out = ft.fit_transform(X)
    expected = np.array([[1.0, np.nan, np.nan, np.nan, 2.5]])
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    np.testing.assert_array_equal(out[~np.isnan(out)], expected[~np.isnan(expected)])


def test_exp_minus_1():
    ft = FunctionTransformer(func="exp_minus_1")
    X = np.array([[0.0, 1.0]])
    np.testing.assert_allclose(ft.fit_transform(X), np.expm1(X), atol=1e-12)


def test_callable_rejected():
    with pytest.raises(ValueError, match="must be one of"):
        FunctionTransformer(func=lambda x: x)


def test_unknown_string_rejected():
    with pytest.raises(ValueError, match="must be one of"):
        FunctionTransformer(func="logarithm")
