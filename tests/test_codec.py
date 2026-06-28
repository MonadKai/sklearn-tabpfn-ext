"""codec.save / codec.load tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sklearn_tabpfn_ext import codec
from sklearn_tabpfn_ext.composite.column_transformer import ColumnTransformer
from sklearn_tabpfn_ext.composite.sequential import SequentialPipeline
from sklearn_tabpfn_ext.constant_filter import ConstantFeatureFilter
from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import StandardScaler


def _make_pipeline():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    X[:, 1] = 7.0  # constant column
    p = SequentialPipeline(
        steps=[
            ("cf", ConstantFeatureFilter()),
            (
                "ct",
                ColumnTransformer(
                    transformers=[("ss", StandardScaler(), [0, 1])],
                    remainder="passthrough",
                ),
            ),
        ]
    )
    p.fit(X)
    return p, X


def test_save_writes_three_files(tmp_path: Path):
    p, _ = _make_pipeline()
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": "2.5.1",
            "extracted_at": "2026-05-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    assert (tmp_path / "meta.json").exists()
    assert (tmp_path / "pipeline.json").exists()
    assert (tmp_path / "state.npz").exists()


def test_save_meta_contents(tmp_path: Path):
    p, _ = _make_pipeline()
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": "2.5.1",
            "extracted_at": "2026-05-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["schema_version"] == 1
    assert meta["source"]["kind"] == "tabpfn"
    assert meta["source"]["estimator_index"] == 0


def test_save_pipeline_json_op_ids(tmp_path: Path):
    p, _ = _make_pipeline()
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": "2.5.1",
            "extracted_at": "2026-05-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    spec = json.loads((tmp_path / "pipeline.json").read_text())
    assert spec["root"]["op_id"] == "vldm.preprocessing.composite.sequential.SequentialPipeline"
    # children are listed in order
    assert spec["root"]["children"][0]["name"] == "cf"
    assert spec["root"]["children"][1]["name"] == "ct"


def test_save_state_npz_loads_without_pickle(tmp_path: Path):
    """state.npz must be readable with allow_pickle=False (security gate)."""
    p, _ = _make_pipeline()
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": "2.5.1",
            "extracted_at": "2026-05-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    npz = np.load(tmp_path / "state.npz", allow_pickle=False, mmap_mode="r")
    # Just access a few keys to ensure load doesn't lazily fail.
    assert len(npz.files) > 0
    for k in npz.files:
        _ = npz[k]


def test_save_load_roundtrip(tmp_path: Path):
    p, X = _make_pipeline()
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": "2.5.1",
            "extracted_at": "2026-05-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    p2 = codec.load(tmp_path)
    np.testing.assert_allclose(p2.transform(X), p.transform(X), atol=1e-12)


def test_order_preserving_survives_codec_roundtrip(tmp_path: Path):
    """order_preserving must round-trip through codec save/load. The CT special-
    case builds init_params explicitly, so the flag is dropped unless serialised
    there — reloading as False would silently produce wrong column order (#issue-2).
    StandardScaler on [0,1] is one-to-one, so the order_preserving guard is satisfied."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    p = SequentialPipeline(
        steps=[
            (
                "ct",
                ColumnTransformer(
                    transformers=[("ss", StandardScaler(), [0, 1])],
                    remainder="passthrough",
                    order_preserving=True,
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
            "tabpfn_version": "6.4.1",
            "extracted_at": "2026-06-12T00:00:00Z",
            "estimator_index": 0,
        },
    )
    p2 = codec.load(tmp_path)
    ct2 = dict(p2.steps)["ct"]
    assert ct2.order_preserving is True, "order_preserving lost on codec round-trip"
    np.testing.assert_allclose(p2.transform(X), p.transform(X), atol=1e-12)


def test_load_rejects_pickle_in_npz(tmp_path: Path):
    """np.load(allow_pickle=False) must reject any pickle-encoded array."""
    p, _ = _make_pipeline()
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": "2.5.1",
            "extracted_at": "2026-05-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    # Corrupt: overwrite state.npz preserving key names but replacing the first
    # array with an object-dtype (pickle-encoded) array.  np.load with
    # allow_pickle=False will raise ValueError when that key is accessed.
    real_npz = np.load(tmp_path / "state.npz", allow_pickle=False)
    keys = real_npz.files
    arrays = {k: real_npz[k] for k in keys}
    bad = np.array([{"a": 1}], dtype=object)
    arrays[keys[0]] = bad  # inject a pickle-encoded array under the first key
    np.savez(tmp_path / "state.npz", **arrays)  # overwrite with bad data
    from sklearn_tabpfn_ext.exceptions import ArtifactSecurityError

    with pytest.raises(ArtifactSecurityError):
        codec.load(tmp_path)


def test_load_rejects_bad_schema(tmp_path: Path):
    p, _ = _make_pipeline()
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": "2.5.1",
            "extracted_at": "2026-05-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    (tmp_path / "pipeline.json").write_text('{"schema_version": 99}')
    from sklearn_tabpfn_ext.exceptions import ArtifactSchemaError

    with pytest.raises(ArtifactSchemaError):
        codec.load(tmp_path)


def test_codec_roundtrip_quantile_svd_with_inner_transformer(tmp_path: Path):
    """The full path: vldm pipeline w/ QuantileSVDReshaper(transformer_=inner pipeline)
    must roundtrip through codec.save -> codec.load -> transform with numerical equality.

    This catches the issue where the inner transformer_ is silently dropped, which
    is the very thing issue #14 is about.
    """
    from sklearn_tabpfn_ext.adaptive_quantile import AdaptiveQuantileTransformer
    from sklearn_tabpfn_ext.quantile_svd import QuantileSVDReshaper

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 10))

    # Build a vldm pipeline where the inner transformer_ is itself a vldm
    # SequentialPipeline (post-translation shape). Use vldm-wrapped AdaptiveQuantileTransformer.
    inner = SequentialPipeline(
        steps=[
            (
                "q",
                AdaptiveQuantileTransformer(
                    n_quantiles=20, output_distribution="uniform", subsample=200, random_state=0
                ),
            ),
        ]
    ).fit(X)

    qs = QuantileSVDReshaper(transformer=inner)
    qs.transformer_ = inner
    qs.subsampled_features_ = np.arange(10, dtype=np.int64)
    qs.n_features_in_ = 10
    qs.n_features_out_ = 10

    p = SequentialPipeline(steps=[("qs", qs)])
    expected = p.transform(X[:5])

    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": "2.5.1",
            "extracted_at": "2026-05-08T00:00:00Z",
            "estimator_index": 0,
        },
    )
    p2 = codec.load(tmp_path)

    # Verify the inner transformer_ is restored.
    assert hasattr(p2.named_steps["qs"], "transformer_")
    inner2 = p2.named_steps["qs"].transformer_
    # And it's a library op (no tabpfn / sklearn leak):
    assert type(inner2).__module__.startswith("sklearn_tabpfn_ext")

    # And transform numerical equivalence:
    np.testing.assert_allclose(p2.transform(X[:5]), expected, atol=1e-12)


def test_metajson_schema_accepts_tabpfn_pickle_sha256(tmp_path: Path):
    """SourceMeta supports an optional tabpfn_pickle_sha256 field for
    fixture-replay identity verification (sigma-server parity slice)."""
    p, _ = _make_pipeline()
    sha = "a" * 64
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": "6.3.2",
            "extracted_at": "2026-05-08T00:00:00Z",
            "estimator_index": 0,
            "tabpfn_pickle_sha256": sha,
        },
    )
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["source"]["tabpfn_pickle_sha256"] == sha


def test_metajson_schema_accepts_inference_precision_fields(tmp_path: Path):
    """SourceMeta supports optional inference_precision_original /
    inference_precision_resolved fields (vldm extract_model.py records the
    source pkl's clf.inference_precision for audit / boundary defense)."""
    p, _ = _make_pipeline()
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": "6.3.2",
            "extracted_at": "2026-05-09T00:00:00Z",
            "estimator_index": 0,
            "inference_precision_original": "auto",
            "inference_precision_resolved": "float32",
        },
    )
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["source"]["inference_precision_original"] == "auto"
    assert meta["source"]["inference_precision_resolved"] == "float32"
    # codec.load drives MetaJson.model_validate — schema acceptance smoke test
    codec.load(tmp_path)


def test_metajson_schema_inference_precision_fields_default_none(tmp_path: Path):
    """Legacy meta.json (no inference_precision_* keys) loads cleanly with
    both fields defaulting to None — backward compat for artifacts produced
    before this PR."""
    p, _ = _make_pipeline()
    codec.save(
        p,
        tmp_path,
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": "6.3.2",
            "extracted_at": "2026-05-09T00:00:00Z",
            "estimator_index": 0,
        },
    )
    # Optional fields with None defaults: pydantic by default omits None
    # from model_dump / serialization. Either absent OR present-as-null is
    # acceptable on disk; what matters is that codec.load works.
    codec.load(tmp_path)
