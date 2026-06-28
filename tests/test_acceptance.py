"""Library acceptance tests — Phase 1 final gate.

Three acceptance tests:

1. ``test_codec_roundtrip_transform_identical_after_move`` — an artifact written
   by the library loads + transforms identically to itself.  Guards the module
   move, import rewrites, and canonical op_id change end-to-end.  No tabpfn
   dependency required (core-only).

2. ``test_all_registered_op_ids_are_canonical_vldm_namespace`` — runs in a
   clean subprocess (no pytest test modules loaded) so ``OPERATOR_REGISTRY``
   holds exactly the 20 production operators.  No filter needed: every entry
   is asserted to start with ``"vldm.preprocessing."`` and satisfy
   ``cls._canonical_op_id == op_id``.  This is the keystone guarantee:
   on-disk ids stay ``vldm.preprocessing.*`` even though the classes live in
   ``sklearn_tabpfn_ext.*``.

3. ``test_core_imports_no_tabpfn_no_torch`` — subprocess test asserting that
   importing ``sklearn_tabpfn_ext.codec``, ``.pipeline``, and
   ``.input_sanitizer`` pulls in no ``tabpfn`` and no ``torch`` module.  Core
   is tabpfn/torch-free.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np


def test_codec_roundtrip_transform_identical_after_move(tmp_path):
    """Artifact written + loaded transforms identically (guards import rewrites + op_id end to end)."""
    from sklearn.preprocessing import OrdinalEncoder as SkOrd

    from sklearn_tabpfn_ext.codec import load, save
    from sklearn_tabpfn_ext.ordinal_encoder import OrdinalEncoder

    sk = SkOrd(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,
    ).fit(np.array([[1.0], [2.0], [np.nan]]))
    op = OrdinalEncoder.from_sklearn(sk)
    save(
        op,
        tmp_path / "op",
        source_meta={
            "kind": "tabpfn",
            "tabpfn_version": "6.3.2",
            "extracted_at": "2026-01-01T00:00:00Z",
            "estimator_index": 0,
        },
    )
    probe = np.array([[2.0], [np.nan], [9.0]], dtype=np.float64)
    got = load(tmp_path / "op").transform(probe)
    np.testing.assert_array_equal(
        np.nan_to_num(got, nan=-7),
        np.nan_to_num(op.transform(probe), nan=-7),
    )


def test_all_registered_op_ids_are_canonical_vldm_namespace():
    """All production operators use the vldm.preprocessing.* namespace (clean subprocess).

    Runs in a fresh interpreter — no pytest test modules loaded, so
    ``OPERATOR_REGISTRY`` holds exactly the production operators registered by
    ``sklearn_tabpfn_ext.__init__``.  No filter needed; every entry is asserted.

    Checks:
    - registry is non-empty
    - at least 20 operators (non-vacuous lower bound)
    - every op_id starts with ``"vldm.preprocessing."``
    - ``cls._canonical_op_id == op_id`` for every entry
    """
    code = (
        "import sklearn_tabpfn_ext\n"
        "from sklearn_tabpfn_ext.registry import OPERATOR_REGISTRY as R\n"
        "assert R, 'registry empty'\n"
        "assert len(R) >= 20, f'expected >=20 production ops, got {len(R)}: {sorted(R)}'\n"
        "bad = [(i, getattr(c, '_canonical_op_id', None), c.__module__) for i, c in R.items()\n"
        "       if not i.startswith('vldm.preprocessing.') or getattr(c, '_canonical_op_id', None) != i]\n"
        "assert not bad, f'non-canonical/mis-namespaced ops: {bad}'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_imports_no_tabpfn_no_torch():
    """Core modules import cleanly — no tabpfn or torch pulled into sys.modules."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sklearn_tabpfn_ext.codec, sklearn_tabpfn_ext.pipeline, "
                "sklearn_tabpfn_ext.input_sanitizer, sys; "
                "assert 'tabpfn' not in sys.modules, "
                "sorted(m for m in sys.modules if 'tabpfn' in m); "
                "assert 'torch' not in sys.modules, "
                "sorted(m for m in sys.modules if 'torch' in m)"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
