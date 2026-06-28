"""SequentialPipeline tests."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn_tabpfn_ext.composite.sequential import SequentialPipeline


def test_sequential_runs_steps_in_order():
    a = StandardScaler()
    b = StandardScaler()
    p = SequentialPipeline(steps=[("a", a), ("b", b)])
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    p.fit(X)
    out = p.transform(X)
    assert out.shape == X.shape


def test_sequential_named_step_access():
    a = StandardScaler()
    p = SequentialPipeline(steps=[("a", a)])
    assert p.named_steps["a"] is a
