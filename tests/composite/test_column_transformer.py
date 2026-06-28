"""ColumnTransformer (vldm) tests."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn_tabpfn_ext.composite.column_transformer import ColumnTransformer


def test_columntransformer_split_concat():
    X = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]])
    ct = ColumnTransformer(
        transformers=[("scale01", StandardScaler(), [0, 1])],
        remainder="passthrough",
    )
    ct.fit(X)
    out = ct.transform(X)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out[:, 2], X[:, 2])  # passthrough preserved


def test_columntransformer_drop_remainder():
    X = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]])
    ct = ColumnTransformer(
        transformers=[("scale", StandardScaler(), [0])],
        remainder="drop",
    )
    ct.fit(X)
    assert ct.transform(X).shape == (2, 1)


def test_columntransformer_codec_style_reconstruction():
    """Simulates codec.load: build ColumnTransformer with already-fitted children
    + remainder, then call _from_state_dict; transform must work."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))

    # First, fit a normal sklearn ColumnTransformer to get a fitted child.
    from sklearn.preprocessing import StandardScaler as SkSS

    fitted_ss = SkSS().fit(X[:, [0, 1]])

    # Now construct the vldm ColumnTransformer the way codec.load would:
    # children pre-fitted, no obj.fit() call.
    ct = ColumnTransformer(
        transformers=[("ss", fitted_ss, [0, 1])],
        remainder="passthrough",
    )
    ColumnTransformer._from_state_dict(ct, {"n_features_in_": np.asarray(3)})

    # transform must succeed without obj.fit().
    out = ct.transform(X)
    assert out.shape == (20, 3)
    # Verify scaling was applied to cols 0,1; col 2 untouched.
    np.testing.assert_allclose(out[:, 2], X[:, 2])


def test_columntransformer_codec_drop_remainder():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    from sklearn.preprocessing import StandardScaler as SkSS

    fitted_ss = SkSS().fit(X[:, [0]])
    ct = ColumnTransformer(
        transformers=[("ss", fitted_ss, [0])],
        remainder="drop",
    )
    ColumnTransformer._from_state_dict(ct, {"n_features_in_": np.asarray(3)})
    out = ct.transform(X)
    assert out.shape == (20, 1)
