"""Codec _child_attrs None-safety + forward-guard tests (spec §3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_tabpfn_ext import codec
from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.composite.sequential import SequentialPipeline
from sklearn_tabpfn_ext.exceptions import ArtifactSchemaError
from sklearn_tabpfn_ext.registry import register
from sklearn_tabpfn_ext.schema import OpSpec


def test_opspec_accepts_serialised_child_attrs():
    spec = OpSpec(
        op_id="a.B",
        name="root",
        init_params={},
        state_keys=[],
        children=[],
        serialised_child_attrs=("x_",),
    )
    assert spec.serialised_child_attrs == ("x_",)


def test_opspec_field_unset_when_omitted():
    spec = OpSpec(op_id="a.B", name="root", init_params={}, state_keys=[], children=[])
    assert spec.serialised_child_attrs == ()
    assert "serialised_child_attrs" not in spec.model_fields_set


def test_base_forward_guard_defaults_false():
    assert VldmEstimatorMixin._forward_guard is False


@register(f"{__name__}.TwoChildOp")  # op_id == f"{cls.__module__}.{cls.__qualname__}"
class TwoChildOp(VldmEstimatorMixin, BaseEstimator, TransformerMixin):
    """Test-only op with two child-attrs; b_ may be None."""

    _state_keys = ("n_features_in_",)
    _init_param_keys = ()
    _child_attrs = ("a_", "b_")

    def __init__(self, *, a=None, b=None):
        self.a = a
        self.b = b

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        self.a_ = self.a.fit(X) if self.a is not None else None
        self.b_ = self.b.fit(X) if self.b is not None else None
        return self

    def transform(self, X):
        return np.asarray(X)


def _wrap(op):
    p = SequentialPipeline(steps=[("op", op)])
    p.fit(np.zeros((3, 2)))
    return p


def test_serialiser_writes_field_when_child_skipped(tmp_path: Path):
    from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import StandardScaler

    p = _wrap(TwoChildOp(a=StandardScaler(), b=None))  # b_ -> None, skipped
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": None,
            "extracted_at": "2026-06-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    import json

    spec = json.loads((tmp_path / "pipeline.json").read_text())
    op = spec["root"]["children"][0]
    assert op["serialised_child_attrs"] == ["a_"]


def test_serialiser_omits_field_for_non_guard_all_present(tmp_path: Path):
    from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import StandardScaler

    p = _wrap(TwoChildOp(a=StandardScaler(), b=StandardScaler()))  # both present
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": None,
            "extracted_at": "2026-06-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    import json

    spec = json.loads((tmp_path / "pipeline.json").read_text())
    op = spec["root"]["children"][0]
    assert "serialised_child_attrs" not in op


def test_roundtrip_with_skipped_child_restores_none(tmp_path: Path):
    from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import StandardScaler

    p = _wrap(TwoChildOp(a=StandardScaler(), b=None))
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": None,
            "extracted_at": "2026-06-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    p2 = codec.load(tmp_path)
    op2 = p2.steps[0][1]
    assert op2.a_ is not None
    assert op2.b_ is None


def test_legacy_spec_without_field_uses_tail_counting(tmp_path: Path):
    """An all-present op with NO serialised_child_attrs (legacy form) still
    restores its child via tail-counting."""
    from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import StandardScaler

    p = _wrap(TwoChildOp(a=StandardScaler(), b=StandardScaler()))
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": None,
            "extracted_at": "2026-06-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    import json

    spec = json.loads((tmp_path / "pipeline.json").read_text())
    assert "serialised_child_attrs" not in spec["root"]["children"][0]  # omitted
    p2 = codec.load(tmp_path)
    op2 = p2.steps[0][1]
    assert op2.a_ is not None and op2.b_ is not None


def test_unknown_child_attr_name_raises(tmp_path: Path):
    from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import StandardScaler

    p = _wrap(TwoChildOp(a=StandardScaler(), b=None))
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": None,
            "extracted_at": "2026-06-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    import json

    pj = tmp_path / "pipeline.json"
    spec = json.loads(pj.read_text())
    spec["root"]["children"][0]["serialised_child_attrs"] = ["does_not_exist_"]
    pj.write_text(json.dumps(spec))
    with pytest.raises(ArtifactSchemaError):
        codec.load(tmp_path)


# ---------------------------------------------------------------------------
# Task 4: Forward-guard wiring + bidirectional compat test
# ---------------------------------------------------------------------------


@register(f"{__name__}.GuardOp")
class GuardOp(TwoChildOp):
    """Forward-guard variant: always writes serialised_child_attrs."""

    _forward_guard = True


def test_guard_op_always_writes_field_even_when_all_present(tmp_path: Path):
    from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import StandardScaler

    p = _wrap(GuardOp(a=StandardScaler(), b=StandardScaler()))
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": None,
            "extracted_at": "2026-06-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    import json

    spec = json.loads((tmp_path / "pipeline.json").read_text())
    assert spec["root"]["children"][0]["serialised_child_attrs"] == ["a_", "b_"]


def test_legacy_strict_schema_rejects_new_field():
    """The forward guard relies on older readers using extra='forbid'. Lock that
    premise: a strict model WITHOUT the field rejects it (as an old OpSpec would),
    so an old reader fails loud at load on a guard artifact."""
    from pydantic import BaseModel, ConfigDict, ValidationError

    class _OldOpSpec(BaseModel):
        model_config = ConfigDict(extra="forbid")
        op_id: str
        name: str
        init_params: dict
        state_keys: list
        children: list = []

    with pytest.raises(ValidationError):
        _OldOpSpec(
            op_id="a.B",
            name="root",
            init_params={},
            state_keys=[],
            children=[],
            serialised_child_attrs=["x_"],
        )


def test_categorical_encoder_is_forward_guard():
    from sklearn_tabpfn_ext.categorical import CategoricalOrdinalEncoder

    assert CategoricalOrdinalEncoder._forward_guard is True


# ---------------------------------------------------------------------------
# Task 5: Full regression — existing pipeline byte-identity
# ---------------------------------------------------------------------------


def test_existing_pipeline_roundtrips_unchanged(tmp_path: Path):
    """A pipeline with a real composite child-attr (no skips, non-guard) must
    serialize without the new field and round-trip byte-identically."""
    from sklearn_tabpfn_ext.composite.column_transformer import ColumnTransformer
    from sklearn_tabpfn_ext.constant_filter import ConstantFeatureFilter
    from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    X[:, 1] = 7.0
    p = SequentialPipeline(
        steps=[
            ("cf", ConstantFeatureFilter()),
            (
                "ct",
                ColumnTransformer(
                    transformers=[("ss", StandardScaler(), [0, 1])], remainder="passthrough"
                ),
            ),
        ]
    )
    p.fit(X)
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": None,
            "extracted_at": "2026-06-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    pj = (tmp_path / "pipeline.json").read_text()
    assert "serialised_child_attrs" not in pj
    p2 = codec.load(tmp_path)
    out1 = p.transform(X)
    out2 = p2.transform(X)
    np.testing.assert_array_equal(out1, out2)


def test_explicit_restore_too_few_children_raises(tmp_path: Path):
    """Corrupt artifact: serialised_child_attrs lists more names than children."""
    from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import StandardScaler

    p = _wrap(TwoChildOp(a=StandardScaler(), b=None))
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": None,
            "extracted_at": "2026-06-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    import json

    pj = tmp_path / "pipeline.json"
    spec = json.loads(pj.read_text())
    # claim both child-attrs present but drop the one serialised child
    spec["root"]["children"][0]["serialised_child_attrs"] = ["a_", "b_"]
    spec["root"]["children"][0]["children"] = []
    pj.write_text(json.dumps(spec))
    with pytest.raises(ArtifactSchemaError):
        codec.load(tmp_path)


def test_legacy_too_few_children_raises(tmp_path: Path):
    """Corrupt legacy artifact (no field): fewer children than declared _child_attrs."""
    from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import StandardScaler

    p = _wrap(TwoChildOp(a=StandardScaler(), b=StandardScaler()))  # all present -> legacy form
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": None,
            "extracted_at": "2026-06-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    import json

    pj = tmp_path / "pipeline.json"
    spec = json.loads(pj.read_text())
    assert "serialised_child_attrs" not in spec["root"]["children"][0]  # legacy
    spec["root"]["children"][0]["children"] = []  # corrupt: drop both children
    pj.write_text(json.dumps(spec))
    with pytest.raises(ArtifactSchemaError):
        codec.load(tmp_path)


@register(f"{__name__}.EmptyGuardOp")
class EmptyGuardOp(VldmEstimatorMixin, BaseEstimator, TransformerMixin):
    """Forward-guard op with NO _child_attrs — must still emit the field ()."""

    _state_keys = ("n_features_in_",)
    _init_param_keys = ()
    _forward_guard = True

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)


def test_guard_op_without_child_attrs_emits_empty_field(tmp_path: Path):
    p = _wrap(EmptyGuardOp())
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": None,
            "extracted_at": "2026-06-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    import json

    spec = json.loads((tmp_path / "pipeline.json").read_text())
    op = spec["root"]["children"][0]
    assert op["serialised_child_attrs"] == []  # present, empty -> guards old readers
    # and it still loads fine in the new reader
    codec.load(tmp_path)
