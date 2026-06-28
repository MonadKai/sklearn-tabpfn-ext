"""End-to-end (vldm venv, no tabpfn): the committed categorical fixture reproduces
tabpfn's cpu_preprocessor byte-exactly, round-trips through the codec, carries a
real CategoricalOrdinalEncoder with random_mappings_, and stays within a size cap.
Fixture built by tests/scripts/make_categorical_fixture.py under conda tabpfn."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from sklearn_tabpfn_ext import codec

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "categorical"
EST = FIX / "estimators"
GOLDEN = FIX / "golden.npz"

pytestmark = pytest.mark.skipif(not GOLDEN.exists(), reason="categorical fixture not built")


def _estimator_dirs():
    return sorted(p for p in EST.glob("estimator_*") if p.is_dir())


def test_fixture_has_real_categorical_encoder_with_random_mappings():
    found = 0
    for d in _estimator_dirs():
        pj = (d / "pipeline.json").read_text()
        if "CategoricalOrdinalEncoder" in pj:
            arrs = np.load(d / "state.npz", allow_pickle=False)
            if any(k.endswith("rm_keys") for k in arrs.files):
                assert '"serialised_child_attrs"' in pj  # forward-guard marker
                found += 1
    assert found >= 1, "no CategoricalOrdinalEncoder with random_mappings_ in fixture"


def test_loaded_pipeline_matches_tabpfn_golden_byte_exact():
    """vldm codec-loaded pipeline.transform == tabpfn cpu_preprocessor output."""
    g = np.load(GOLDEN)
    Xt = g["X_test"]
    for i, d in enumerate(_estimator_dirs()):
        p = codec.load(d)
        out = np.asarray(p.transform(Xt), dtype=np.float64)
        ref = g[f"cpp_out_{i}"]
        assert out.shape == ref.shape
        assert float(np.nanmax(np.abs(out - ref))) == 0.0, f"estimator {i} drift"


def test_categorical_codec_roundtrip_byte_identical():
    g = np.load(GOLDEN)
    Xt = g["X_test"]
    for d in _estimator_dirs():
        p = codec.load(d)
        with tempfile.TemporaryDirectory() as td:
            codec.save(
                p,
                td,
                source_meta={
                    "kind": "tabpfn",
                    "tabpfn_version": None,
                    "extracted_at": "2026-06-08T00:00:00Z",
                    "estimator_index": 0,
                },
            )
            p2 = codec.load(td)
        np.testing.assert_array_equal(p.transform(Xt), p2.transform(Xt))


def test_fixture_size_budget():
    total = sum(f.stat().st_size for f in FIX.rglob("*") if f.is_file())
    assert total < 1 * 1024 * 1024, f"categorical fixture is {total / 1e6:.2f} MB (>1 MB cap)"
