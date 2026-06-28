"""Pydantic schema tests for pipeline.json / meta.json."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sklearn_tabpfn_ext.schema import MetaJson, OpSpec, PipelineSpec


def test_metajson_roundtrip():
    meta = MetaJson(
        schema_version=1,
        vldm_version="0.5.0",
        sklearn_version="1.5.2",
        numpy_version="1.26.4",
        source={
            "kind": "tabpfn",
            "tabpfn_version": "2.5.1",
            "extracted_at": "2026-05-08T00:00:00Z",
            "estimator_index": 0,
        },
        estimator_id="estimator_0",
        n_features_in=147,
        n_features_out_after_pipeline=256,
    )
    s = meta.model_dump_json()
    again = MetaJson.model_validate_json(s)
    assert again == meta


def test_opspec_minimal():
    spec = OpSpec(
        op_id="vldm.preprocessing.constant_filter.ConstantFeatureFilter",
        name="constant_filter",
        init_params={},
        state_keys=["sel_", "n_features_in_"],
    )
    assert spec.children == []


def test_opspec_with_children():
    spec = OpSpec(
        op_id="vldm.preprocessing.composite.sequential.SequentialPipeline",
        name="root",
        init_params={"steps": [["a", {"child_ref": 0}]]},
        state_keys=[],
        children=[
            OpSpec(
                op_id="vldm.preprocessing.constant_filter.ConstantFeatureFilter",
                name="a",
                init_params={},
                state_keys=[],
            ),
        ],
    )
    assert len(spec.children) == 1


def test_pipeline_spec_roundtrip():
    inner = OpSpec(
        op_id="vldm.preprocessing.constant_filter.ConstantFeatureFilter",
        name="constant_filter",
        init_params={},
        state_keys=["sel_"],
    )
    root = OpSpec(
        op_id="vldm.preprocessing.composite.sequential.SequentialPipeline",
        name="root",
        init_params={"steps": [["constant_filter", {"child_ref": 0}]]},
        state_keys=[],
        children=[inner],
    )
    spec = PipelineSpec(schema_version=1, root=root)

    s = spec.model_dump_json()
    again = PipelineSpec.model_validate_json(s)
    assert again == spec


def test_schema_version_must_be_one():
    with pytest.raises(ValidationError):
        PipelineSpec(
            schema_version=2, root=OpSpec(op_id="x", name="r", init_params={}, state_keys=[])
        )


def test_invalid_op_id_rejected():
    with pytest.raises(ValidationError):
        OpSpec(op_id="", name="r", init_params={}, state_keys=[])
