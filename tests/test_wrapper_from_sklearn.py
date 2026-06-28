"""Parity tests: Op.from_sklearn(sk).transform(X) == sk.transform(X)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("name", ["standard_scaler", "truncated_svd", "simple_imputer"])
def test_wrapper_from_sklearn_matches_source(name):
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    if name == "standard_scaler":
        from sklearn.preprocessing import StandardScaler as Sk

        from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import (
            StandardScaler as Op,
        )

        sk = Sk().fit(X)
    elif name == "truncated_svd":
        from sklearn.decomposition import TruncatedSVD as Sk

        from sklearn_tabpfn_ext.sklearn_wrappers.truncated_svd import (
            TruncatedSVD as Op,
        )

        sk = Sk(n_components=2).fit(X)
    else:
        from sklearn.impute import SimpleImputer as Sk

        from sklearn_tabpfn_ext.sklearn_wrappers.simple_imputer import (
            SimpleImputer as Op,
        )

        sk = Sk().fit(X)
    op = Op.from_sklearn(sk)
    np.testing.assert_allclose(op.transform(X), sk.transform(X))


def test_wrapper_from_sklearn_simple_imputer_non_default_missing_values():
    """SimpleImputer with non-default missing_values must be transform-identical."""
    from sklearn.impute import SimpleImputer as Sk

    from sklearn_tabpfn_ext.sklearn_wrappers.simple_imputer import SimpleImputer as Op

    # -1.0 is the sentinel for missing
    X = np.array([[1.0, 2.0, -1.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    sk = Sk(missing_values=-1.0).fit(X)
    op = Op.from_sklearn(sk)
    np.testing.assert_allclose(op.transform(X), sk.transform(X))
