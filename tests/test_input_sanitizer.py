import numpy as np
import pytest
import sklearn.compose
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder as SkOrdinalEncoder

from sklearn_tabpfn_ext.exceptions import UnsupportedConversionError
from sklearn_tabpfn_ext.tabpfn.translate import translate_input_sanitizer

pytestmark = pytest.mark.tabpfn


def _tabpfn_like_ord_encoder(cat_indices, n_features):
    """Mimic get_ordinal_encoder(): vanilla ColumnTransformer (reorders),
    OrdinalEncoder on cat columns, passthrough remainder."""
    oe = SkOrdinalEncoder(
        categories="auto",
        dtype=np.float64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,
    )
    return ColumnTransformer(
        transformers=[("ordinal", oe, list(cat_indices))],
        remainder="passthrough",
    )


class _FakeClf:
    pass


def test_non_identity_reproduces_sklearn_columntransformer_byte_exact():
    rng = np.random.default_rng(0)
    n = 200
    X = np.column_stack(
        [
            rng.normal(size=n),  # col0 numeric
            rng.integers(0, 4, size=n).astype(float),  # col1 categorical
            rng.normal(size=n),  # col2 numeric
        ]
    )
    ct = _tabpfn_like_ord_encoder([1], n_features=3).fit(X)

    clf = _FakeClf()
    clf.n_features_in_ = 3
    clf.ordinal_encoder_ = ct
    clf.inferred_categorical_indices_ = [1]

    san = translate_input_sanitizer(clf)

    X_test = np.array([[0.5, 2.0, -0.5], [1.0, 99.0, 0.0]], dtype=np.float32)
    got = san.transform(X_test)
    exp = ct.transform(X_test.astype(np.float64))
    assert got.shape == exp.shape
    assert np.array_equal(np.isnan(got), np.isnan(exp))
    m = ~np.isnan(exp)
    assert np.allclose(got[m], exp[m], rtol=0, atol=0)


def test_identity_is_literal_passthrough():
    clf = _FakeClf()
    clf.n_features_in_ = 3
    clf.ordinal_encoder_ = None
    clf.inferred_categorical_indices_ = []
    san = translate_input_sanitizer(clf)
    X = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = san.transform(X)
    assert out is X
    assert out.dtype == X.dtype  # no cast for identity


def test_subclass_ordinal_encoder_raises_unsupported():
    """A ColumnTransformer *subclass* (e.g. tabpfn v7) must hard-fail."""

    class _Sub(sklearn.compose.ColumnTransformer):
        pass

    rng = np.random.default_rng(1)
    n = 50
    X = np.column_stack(
        [
            rng.normal(size=n),
            rng.integers(0, 3, size=n).astype(float),
        ]
    )
    oe = SkOrdinalEncoder(
        categories="auto",
        dtype=np.float64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,
    )
    # Build a subclass instance and fit it
    sub = _Sub(transformers=[("ordinal", oe, [1])], remainder="passthrough")
    sub.fit(X)

    clf = _FakeClf()
    clf.n_features_in_ = 2
    clf.ordinal_encoder_ = sub
    clf.inferred_categorical_indices_ = [1]

    with pytest.raises(UnsupportedConversionError):
        translate_input_sanitizer(clf)
