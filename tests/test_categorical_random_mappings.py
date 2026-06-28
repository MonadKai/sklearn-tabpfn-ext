"""CategoricalOrdinalEncoder random_mappings_ codec round-trip + validation (spec §4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import FunctionTransformer

from sklearn_tabpfn_ext import OrdinalEncoder, codec
from sklearn_tabpfn_ext.categorical import CategoricalOrdinalEncoder
from sklearn_tabpfn_ext.composite.column_transformer import ColumnTransformer
from sklearn_tabpfn_ext.composite.sequential import SequentialPipeline
from sklearn_tabpfn_ext.exceptions import ArtifactSchemaError


def _make_fitted_ordinal_encoder(n_categories: int) -> OrdinalEncoder:
    """Return a vldm OrdinalEncoder fitted with n_categories integer codes."""
    enc = OrdinalEncoder()
    enc.categories_ = [np.arange(n_categories, dtype=np.float64)]
    enc.n_features_in_ = 1
    enc._cats_has_nan_ = [False]
    return enc


def _make_fitted_column_transformer(n_features_in: int) -> ColumnTransformer:
    """Return a vldm ColumnTransformer fitted on column 0 (ordinal) + remainder passthrough."""
    oe = _make_fitted_ordinal_encoder(n_categories=2)
    ct = ColumnTransformer(
        transformers=[("ordinal_encoder", oe, [0])],
        remainder="passthrough",
    )
    ct.n_features_in_ = n_features_in
    # _columns: required by _iter(fitted=False) on every transform() call.
    ct._columns = [[0]]
    leftover = [i for i in range(n_features_in) if i != 0]
    # _remainder: required by _iter(fitted=False).
    ct._remainder = ("remainder", "passthrough", leftover)
    # transformers_: post-fit list for _iter(fitted=True).
    passthrough_ft = FunctionTransformer(
        accept_sparse=True,
        check_inverse=False,
        feature_names_out="one-to-one",
    )
    ct.transformers_ = [("ordinal_encoder", oe, [0])]
    if leftover:
        ct.transformers_.append(("remainder", passthrough_ft, leftover))
    ct.sparse_output_ = False
    return ct


def _make_encoder(random_mappings, n_features_in: int = 2) -> CategoricalOrdinalEncoder:
    enc = CategoricalOrdinalEncoder()
    enc.n_features_in_ = n_features_in
    enc.column_transformer_ = _make_fitted_column_transformer(n_features_in)
    enc.random_mappings_ = dict(random_mappings)
    return enc


def _wrap(enc):
    return SequentialPipeline(steps=[("cat", enc)])


def test_to_state_dict_emits_rm_keys_and_vals():
    enc = _make_encoder({0: np.array([2, 0, 1], dtype=np.int64)})
    state = enc._to_state_dict()
    assert set(state) >= {"n_features_in_", "rm_keys", "rm_val_0"}
    assert state["rm_keys"].tolist() == [0]
    assert state["rm_val_0"].tolist() == [2, 0, 1]


def test_roundtrip_restores_random_mappings(tmp_path: Path):
    enc = _make_encoder({0: np.array([1, 0, 2], dtype=np.int64)})
    codec.save(
        _wrap(enc),
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": None,
            "extracted_at": "2026-06-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    p2 = codec.load(tmp_path)
    enc2 = p2.steps[0][1]
    assert set(enc2.random_mappings_) == {0}
    np.testing.assert_array_equal(enc2.random_mappings_[0], [1, 0, 2])
    assert enc2.random_mappings_[0].dtype == np.int64


def test_empty_random_mappings_roundtrip(tmp_path: Path):
    enc = _make_encoder({})
    codec.save(
        _wrap(enc),
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": None,
            "extracted_at": "2026-06-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    p2 = codec.load(tmp_path)
    assert p2.steps[0][1].random_mappings_ == {}


@pytest.mark.parametrize(
    "tamper,exc",
    [
        ("dup_keys", ArtifactSchemaError),
        ("unsorted_keys", ArtifactSchemaError),
        ("orphan_val", ArtifactSchemaError),
        ("missing_val", ArtifactSchemaError),
        ("out_of_range", ArtifactSchemaError),
    ],
)
def test_corrupt_random_mappings_fail_loud(tmp_path: Path, tamper, exc):
    import json as _json

    enc = _make_encoder(
        {0: np.array([1, 0], dtype=np.int64), 1: np.array([0, 1], dtype=np.int64)},
    )
    codec.save(
        _wrap(enc),
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": None,
            "extracted_at": "2026-06-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    npz_path = tmp_path / "state.npz"
    arrs = dict(np.load(npz_path, allow_pickle=False))
    rk = next(k for k in arrs if k.endswith("/rm_keys") or k == "rm_keys")
    prefix = rk[: -len("rm_keys")]
    if tamper == "dup_keys":
        arrs[prefix + "rm_keys"] = np.array([0, 0], dtype=np.int64)
    elif tamper == "unsorted_keys":
        arrs[prefix + "rm_keys"] = np.array([1, 0], dtype=np.int64)
    elif tamper == "orphan_val":
        arrs[prefix + "rm_keys"] = np.array([0], dtype=np.int64)
    elif tamper == "missing_val":
        # Remove rm_val_1 from both the npz AND pipeline.json state_keys so the
        # codec passes state-loading and _from_state_dict raises ArtifactSchemaError.
        del arrs[prefix + "rm_val_1"]
        pj_path = tmp_path / "pipeline.json"
        pj = _json.loads(pj_path.read_text())
        cat_node = pj["root"]["children"][0]
        cat_node["state_keys"] = [k for k in cat_node["state_keys"] if k != "rm_val_1"]
        pj_path.write_text(_json.dumps(pj))
    elif tamper == "out_of_range":
        arrs[prefix + "rm_val_0"] = np.array([5, 0], dtype=np.int64)
    np.savez(npz_path, **arrs)
    with pytest.raises(exc):
        codec.load(tmp_path)
