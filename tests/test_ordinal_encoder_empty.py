"""OrdinalEncoder.from_sklearn must treat an unfitted/zero-column sklearn
OrdinalEncoder (numeric-only models whose ColumnTransformer selects no
categorical columns) as a no-op instead of crashing on missing categories_."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import OrdinalEncoder as SkOrdinalEncoder

from sklearn_tabpfn_ext.ordinal_encoder import OrdinalEncoder


def test_from_sklearn_unfitted_empty_is_noop():
    sk = SkOrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1.0)
    enc = OrdinalEncoder.from_sklearn(sk)  # never fitted -> no categories_
    assert enc.categories_ == []
    assert enc.n_features_in_ == 0
    out = enc.transform(np.zeros((3, 0)))
    assert out.shape == (3, 0)
