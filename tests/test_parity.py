"""End-to-end parity: tabpfn cpu_preprocessor vs vldm-translated pipeline.

Runs only when VLDM_TABPFN_BINARY_PICKLE points at a fitted TabPFNClassifier.
Marked gpu_ladder per the project's CI convention (CPU CI does not run this).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu_ladder, pytest.mark.tabpfn]


def _ckpt_or_skip() -> Path:
    p = os.environ.get("VLDM_TABPFN_BINARY_PICKLE")
    if not p:
        pytest.skip("set VLDM_TABPFN_BINARY_PICKLE to a tabpfn binary pickle path")
    return Path(p)


@pytest.fixture(scope="module")
def members_and_x():
    import joblib

    clf = joblib.load(_ckpt_or_skip())
    members = clf.executor_.ensemble_members
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, clf.n_features_in_))
    return members, X


@pytest.mark.parametrize("variant", ["normal", "all_zero", "all_nan", "constant_columns"])
def test_parity_per_member(members_and_x, variant):
    from sklearn_tabpfn_ext.tabpfn import translate_member

    members, X = members_and_x
    if variant == "all_zero":
        X = np.zeros_like(X)
    elif variant == "all_nan":
        X = np.full_like(X, np.nan)
    elif variant == "constant_columns":
        X = np.broadcast_to(X[0], X.shape).copy()

    for i, member in enumerate(members):
        ref_result = member.cpu_preprocessor.transform(X)
        # tabpfn cpu_preprocessor returns a TransformResult named-tuple;
        # extract the numeric array via .X (element 0).
        ref = ref_result.X if hasattr(ref_result, "X") else np.asarray(ref_result[0])
        vldm_pipe = translate_member(member)
        out = vldm_pipe.transform(X)
        np.testing.assert_allclose(
            out,
            ref,
            atol=1e-6,
            rtol=1e-6,
            err_msg=f"member {i} variant={variant} mismatch",
        )
