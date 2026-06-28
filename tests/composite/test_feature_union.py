"""FeatureUnion (vldm) tests."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn_tabpfn_ext.composite.feature_union import FeatureUnion


def test_feature_union_concats():
    X = np.array([[1.0], [2.0], [3.0]])
    fu = FeatureUnion(transformer_list=[("ss", StandardScaler()), ("mm", MinMaxScaler())])
    fu.fit(X)
    out = fu.transform(X)
    assert out.shape == (3, 2)
