"""Tests: canonical op_id decoupled from __module__ (Task 3)."""

from __future__ import annotations

import pytest


def test_register_stores_canonical_id_independent_of_module():
    from sklearn_tabpfn_ext.base import VldmEstimatorMixin
    from sklearn_tabpfn_ext.registry import OPERATOR_REGISTRY, register

    @register("vldm.preprocessing.demo.Demo")
    class Demo(VldmEstimatorMixin):  # actual __module__ is this test module
        pass

    # the registered canonical id is stored on the class, not derived from __module__
    assert Demo._canonical_op_id == "vldm.preprocessing.demo.Demo"
    assert OPERATOR_REGISTRY["vldm.preprocessing.demo.Demo"] is Demo
    # cleanup so re-runs don't hit "already registered"
    del OPERATOR_REGISTRY["vldm.preprocessing.demo.Demo"]


@pytest.mark.tabpfn  # needs sklearn; gate so it runs in the dev+wrappers env
def test_written_op_id_is_canonical_not_module_path(tmp_path):
    import json

    import numpy as np
    from sklearn.preprocessing import OrdinalEncoder as SkOrd
    from sklearn_tabpfn_ext.ordinal_encoder import OrdinalEncoder

    from sklearn_tabpfn_ext.codec import save

    sk = SkOrd(handle_unknown="use_encoded_value", unknown_value=-1,
               encoded_missing_value=np.nan).fit(np.array([[1.0], [2.0]]))
    op = OrdinalEncoder.from_sklearn(sk)
    save(op, tmp_path / "op", source_meta={"kind": "tabpfn", "tabpfn_version": "6.3.2", "estimator_index": 0})
    spec = json.loads((tmp_path / "op" / "pipeline.json").read_text())
    assert spec["root"]["op_id"] == "vldm.preprocessing.ordinal_encoder.OrdinalEncoder"
    assert "sklearn_tabpfn_ext" not in spec["root"]["op_id"]
