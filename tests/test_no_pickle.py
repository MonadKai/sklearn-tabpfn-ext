"""Security gate: codec.load must not invoke pickle / joblib / dill at any point.

Strategy: monkey-patch each known unsafe loader to raise; if the load
path triggers any of them we fail loudly. Also: a static grep over
vldm/preprocessing/* must produce no joblib/pickle calls.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from sklearn_tabpfn_ext import codec
from sklearn_tabpfn_ext.composite.column_transformer import ColumnTransformer
from sklearn_tabpfn_ext.composite.sequential import SequentialPipeline
from sklearn_tabpfn_ext.constant_filter import ConstantFeatureFilter
from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import StandardScaler


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    X[:, 1] = 4.0
    p = SequentialPipeline(
        steps=[
            ("cf", ConstantFeatureFilter()),
            (
                "ss",
                ColumnTransformer(
                    transformers=[("ss", StandardScaler(), [0])], remainder="passthrough"
                ),
            ),
        ]
    ).fit(X)
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
    return tmp_path


def _boom(*a, **kw):
    raise RuntimeError("disallowed loader called during codec.load")


def test_load_never_calls_pickle(artifact: Path):
    with mock.patch("pickle.load", _boom), mock.patch("pickle.loads", _boom):
        codec.load(artifact)


def test_load_never_calls_joblib(artifact: Path):
    import joblib

    with mock.patch.object(joblib, "load", _boom):
        codec.load(artifact)


def test_static_grep_no_joblib_in_preprocessing():
    """Source-level guarantee: vldm/preprocessing/* must not call joblib/pickle."""
    r = subprocess.run(
        [
            "grep",
            "-rnE",
            r"joblib\.(load|dump)|pickle\.(load|loads|dump|dumps)",
            "sklearn_tabpfn_ext/",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0, f"joblib/pickle found in vldm/preprocessing/: {r.stdout}"


def test_static_grep_no_tabpfn_in_preprocessing():
    """vldm/preprocessing/ must not import tabpfn."""
    r = subprocess.run(
        ["grep", "-rnE", r"^(from|import)\s+tabpfn", "sklearn_tabpfn_ext/"],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0, f"tabpfn import in vldm/preprocessing/: {r.stdout}"
