"""Order-preserving CT keeps original column positions for one-to-one blocks;
fails loud on a width-changing block or a dropped column. Pure CPU, no tabpfn."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import FunctionTransformer

from sklearn_tabpfn_ext.composite.column_transformer import ColumnTransformer


def _plus_ten(X):
    return X + 10.0


def test_order_preserving_restores_positions():
    ct = ColumnTransformer(
        transformers=[("enc", FunctionTransformer(func=_plus_ten), [1])],
        remainder="passthrough",
        order_preserving=True,
    )
    ct.n_features_in_ = 3
    ct._from_state_dict(ct, {"n_features_in_": np.asarray(3)})
    X = np.array([[0.0, 1.0, 2.0]])
    out = ct.transform(X)
    np.testing.assert_array_equal(out, np.array([[0.0, 11.0, 2.0]]))


def test_non_one_to_one_block_fails_loud():
    def _widen(X):
        return np.concatenate([X, X], axis=1)

    ct = ColumnTransformer(
        transformers=[("enc", FunctionTransformer(func=_widen), [1])],
        remainder="passthrough",
        order_preserving=True,
    )
    ct.n_features_in_ = 3
    ct._from_state_dict(ct, {"n_features_in_": np.asarray(3)})
    with pytest.raises(ValueError):
        ct.transform(np.array([[0.0, 1.0, 2.0]]))


def test_explicit_drop_fails_loud():
    ct = ColumnTransformer(
        transformers=[("enc", FunctionTransformer(func=_plus_ten), [1]), ("gone", "drop", [0])],
        remainder="passthrough",
        order_preserving=True,
    )
    ct.n_features_in_ = 3
    ct._from_state_dict(ct, {"n_features_in_": np.asarray(3)})
    with pytest.raises(ValueError):
        ct.transform(np.array([[0.0, 1.0, 2.0]]))


def test_default_is_v63_passthrough_order():
    ct = ColumnTransformer(
        transformers=[("enc", FunctionTransformer(func=_plus_ten), [1])],
        remainder="passthrough",
    )
    assert ct.order_preserving is False
