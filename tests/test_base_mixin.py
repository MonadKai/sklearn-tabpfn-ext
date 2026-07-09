"""Contract tests for VldmEstimatorMixin."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_tabpfn_ext.base import VldmEstimatorMixin


class _Demo(VldmEstimatorMixin, BaseEstimator, TransformerMixin):
    _state_keys = ("mean_", "n_features_in_")
    _init_param_keys = ("with_mean",)

    def __init__(self, with_mean: bool = True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X - self.mean_ if self.with_mean else X


def test_to_state_dict_returns_ndarrays():
    d = _Demo(with_mean=True).fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    state = d._to_state_dict()
    assert set(state.keys()) == {"mean_", "n_features_in_"}
    assert isinstance(state["mean_"], np.ndarray)
    assert isinstance(state["n_features_in_"], np.ndarray)
    assert state["n_features_in_"].shape == ()  # 0-d


def test_from_state_dict_reconstructs():
    d = _Demo(with_mean=False).fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    X = np.array([[5.0, 7.0]])
    expected = d.transform(X)

    init = d._init_params_dict()
    new = _Demo(**init)
    state = d._to_state_dict()
    _Demo._from_state_dict(new, state)

    np.testing.assert_array_equal(new.transform(X), expected)
    # 0-d int ndarray must be unwrapped to a Python int (sklearn internals
    # typecheck against int, not 0-d ndarray).
    assert isinstance(new.n_features_in_, int)


def test_init_params_dict_extracts_only_declared():
    d = _Demo(with_mean=True)
    assert d._init_params_dict() == {"with_mean": True}


def test_to_state_dict_raises_when_unfitted():
    d = _Demo()
    with pytest.raises(AttributeError):
        d._to_state_dict()


def test_from_state_dict_raises_on_missing_key():
    """Missing state key must surface as OpStateError with the key + op_path."""
    from sklearn_tabpfn_ext.exceptions import OpStateError

    new = _Demo()
    with pytest.raises(OpStateError) as exc_info:
        _Demo._from_state_dict(new, {})  # empty state, mean_ missing
    assert exc_info.value.key == "mean_"
    assert exc_info.value.op_path == "_Demo"


def test_schema_validates_minimal_pipeline_spec():
    from sklearn_tabpfn_ext.schema import OpSpec, PipelineSpec

    spec = PipelineSpec.model_validate(
        {
            "schema_version": 1,
            "root": {
                "op_id": "vldm.preprocessing.ordinal_encoder.OrdinalEncoder",
                "name": "root",
                "init_params": {},
                "state_keys": [],
                "children": [],
            },
        }
    )
    assert isinstance(spec.root, OpSpec)
