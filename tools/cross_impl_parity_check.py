"""Cross-impl backward/forward-compat parity check: vldm <-> sklearn-tabpfn-ext.

Migration-period verification (NOT part of the package, NOT run by default CI).
Proves the keystone guarantee concretely and torch-free: an artifact written by
vldm's codec (on-disk op_id ``vldm.preprocessing.*``) loads in this library and
transforms byte-identically, and vice versa — for leaf operators and for a
nested ``SequentialPipeline`` (the real ``cpu_preprocessor`` shape). This is the
intended Phase B backward-compatibility gate.

vldm's ``preprocessing`` subpackage is pure-python (sklearn/numpy/pydantic) and
imports without torch/tabpfn, so this check needs no fitted TabPFN.

Run (from anywhere, in the project's uv env)::

    VLDM_SRC=/path/to/vldm uv run --extra dev python tools/cross_impl_parity_check.py

``VLDM_SRC`` defaults to a sibling ``../vldm`` next to this repo. Exits non-zero
on any parity mismatch.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# vldm source on path BEFORE importing it (pure-python preprocessing, torch-free).
_DEFAULT_VLDM = Path(__file__).resolve().parent.parent.parent / "vldm"
VLDM_SRC = os.environ.get("VLDM_SRC", str(_DEFAULT_VLDM))
sys.path.insert(0, VLDM_SRC)

import numpy as np  # noqa: E402

try:
    import vldm.preprocessing.codec as vcodec
    from vldm.preprocessing.composite.column_transformer import (
        ColumnTransformer as VColumnTransformer,
    )
    from vldm.preprocessing.composite.sequential import SequentialPipeline as VSeq
    from vldm.preprocessing.input_sanitizer import InputSanitizer as VInputSanitizer
    from vldm.preprocessing.ordinal_encoder import OrdinalEncoder as VOrdinalEncoder
    from vldm.preprocessing.sklearn_wrappers.simple_imputer import (
        SimpleImputer as VImp,
    )
    from vldm.preprocessing.sklearn_wrappers.standard_scaler import (
        StandardScaler as Vss,
    )
    from vldm.preprocessing.sklearn_wrappers.truncated_svd import TruncatedSVD as Vsvd
except ImportError as exc:  # pragma: no cover - operator/setup guidance
    sys.exit(
        f"cannot import vldm.preprocessing from {VLDM_SRC!r}: {exc}\n"
        "Set VLDM_SRC=/path/to/vldm (the repo root containing the 'vldm' package)."
    )

from sklearn.decomposition import TruncatedSVD as SkSVD  # noqa: E402
from sklearn.impute import SimpleImputer as SkImp  # noqa: E402
from sklearn.pipeline import Pipeline as SkPipe  # noqa: E402
from sklearn.preprocessing import OrdinalEncoder as SkOrd  # noqa: E402
from sklearn.preprocessing import StandardScaler as SkSS  # noqa: E402

import sklearn_tabpfn_ext.codec as ncodec  # noqa: E402
from sklearn_tabpfn_ext.composite.column_transformer import (  # noqa: E402
    ColumnTransformer as NColumnTransformer,
)
from sklearn_tabpfn_ext.composite.sequential import SequentialPipeline as NSeq  # noqa: E402
from sklearn_tabpfn_ext.input_sanitizer import InputSanitizer as NInputSanitizer  # noqa: E402
from sklearn_tabpfn_ext.ordinal_encoder import OrdinalEncoder as NOrdinalEncoder  # noqa: E402
from sklearn_tabpfn_ext.sklearn_wrappers.simple_imputer import SimpleImputer as NImp  # noqa: E402
from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import StandardScaler as Nss  # noqa: E402
from sklearn_tabpfn_ext.sklearn_wrappers.truncated_svd import TruncatedSVD as Nsvd  # noqa: E402

assert "torch" not in sys.modules, "torch must NOT be imported (core is torch-free)"
assert "tabpfn" not in sys.modules, "tabpfn must NOT be imported (this path is core-only)"

SRC = {
    "kind": "tabpfn",
    "tabpfn_version": "6.3.2",
    "estimator_index": 0,
    "extracted_at": "2026-06-01T00:00:00Z",
}
X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [2.0, 1.0, 0.0]], dtype=np.float64)
PROBE = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0], [2.0, 6.0, 5.0]], dtype=np.float64)
SAN_X = np.array([[0.1, 1.0], [0.2, 2.0], [0.3, np.nan], [0.4, 1.0]], dtype=np.float64)
SAN_PROBE = np.array([[9.0, 2.0], [8.0, 99.0], [7.0, np.nan]], dtype=np.float64)


def build(op_cls, sk):
    """Reconstruct a fitted library op from a fitted sklearn estimator (mirrors _instantiate_with_state)."""
    init = {k: getattr(sk, k) for k in op_cls._init_param_keys if hasattr(sk, k)}
    op = op_cls(**init)
    for k in op_cls._state_keys:
        op.__dict__[k] = getattr(sk, k)
    return op


def nan_eq(a, b):
    return np.array_equal(
        np.nan_to_num(np.asarray(a, float), nan=-7.0), np.nan_to_num(np.asarray(b, float), nan=-7.0)
    )


def build_sanitizer(input_sanitizer_cls, column_transformer_cls, ordinal_encoder_cls):
    sk = SkOrd(
        dtype=np.float64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,
    ).fit(SAN_X[:, [1]])
    enc = ordinal_encoder_cls.from_sklearn(sk)
    ct = column_transformer_cls(
        transformers=[("ordinal", enc, [1])],
        remainder="passthrough",
    )
    column_transformer_cls._from_state_dict(ct, {"n_features_in_": np.asarray(2)})
    return input_sanitizer_cls(
        n_features_in=2,
        inferred_categorical_indices=[1],
        column_transformer=ct,
    )


def main() -> int:
    failures = []

    leaf_cases = [
        ("StandardScaler", Vss, Nss, SkSS().fit(X)),
        ("TruncatedSVD", Vsvd, Nsvd, SkSVD(n_components=2, random_state=0).fit(X)),
        ("SimpleImputer", VImp, NImp, SkImp().fit(X)),
    ]
    for name, vcls, ncls, sk in leaf_cases:
        sk_t = sk.transform(PROBE)
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "vldm_written"
            vcodec.save(build(vcls, sk), a, source_meta=dict(SRC))
            op_id = json.loads((a / "pipeline.json").read_text())["root"]["op_id"]
            t_newlib = ncodec.load(a).transform(PROBE)
            t_vldm = vcodec.load(a).transform(PROBE)
            b = Path(d) / "newlib_written"
            ncodec.save(build(ncls, sk), b, source_meta=dict(SRC))
            op_id_n = json.loads((b / "pipeline.json").read_text())["root"]["op_id"]
            t_vldm_from_new = vcodec.load(b).transform(PROBE)
        checks = {
            "vldm-written op_id is vldm.preprocessing.*": op_id.startswith("vldm.preprocessing."),
            "newlib-written op_id is vldm.preprocessing.*": op_id_n.startswith(
                "vldm.preprocessing."
            ),
            "vldm-written -> newlib-load == sklearn": nan_eq(t_newlib, sk_t),
            "vldm-written -> newlib-load == vldm-load": nan_eq(t_newlib, t_vldm),
            "newlib-written -> vldm-load == sklearn": nan_eq(t_vldm_from_new, sk_t),
        }
        bad = [k for k, ok in checks.items() if not ok]
        print(f"[{'OK' if not bad else 'FAIL'}] {name}: op_id={op_id}")
        for k in bad:
            print(f"      x {k}")
        if bad:
            failures.append((name, bad))

    # Nested composite: SequentialPipeline (representative cpu_preprocessor shape).
    skpipe = SkPipe([("ss", SkSS()), ("imp", SkImp())]).fit(X)  # 3->3->3, dimensionally chainable
    sk_seq_t = skpipe.transform(PROBE)

    def build_seq(seq_cls, wrap_ss, wrap_imp):
        return seq_cls(
            steps=[
                ("ss", build(wrap_ss, skpipe.named_steps["ss"])),
                ("imp", build(wrap_imp, skpipe.named_steps["imp"])),
            ]
        )

    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "vldm_seq"
        vcodec.save(build_seq(VSeq, Vss, VImp), a, source_meta=dict(SRC))
        spec = json.loads((a / "pipeline.json").read_text())["root"]
        root_id = spec["op_id"]
        child_ids = [c["op_id"] for c in spec.get("children", [])]
        seq_newlib = ncodec.load(a).transform(PROBE)
        seq_vldm = vcodec.load(a).transform(PROBE)
        b = Path(d) / "newlib_seq"
        ncodec.save(build_seq(NSeq, Nss, NImp), b, source_meta=dict(SRC))
        seq_vldm_from_new = vcodec.load(b).transform(PROBE)
    seq_checks = {
        "root op_id is vldm.preprocessing.*": root_id.startswith("vldm.preprocessing."),
        "all child op_ids are vldm.preprocessing.*": bool(child_ids)
        and all(c.startswith("vldm.preprocessing.") for c in child_ids),
        "vldm-written seq -> newlib-load == sklearn": nan_eq(seq_newlib, sk_seq_t),
        "vldm-written seq -> newlib-load == vldm-load": nan_eq(seq_newlib, seq_vldm),
        "newlib-written seq -> vldm-load == sklearn": nan_eq(seq_vldm_from_new, sk_seq_t),
    }
    seq_bad = [k for k, ok in seq_checks.items() if not ok]
    print(
        f"[{'OK' if not seq_bad else 'FAIL'}] SequentialPipeline(nested): "
        f"root={root_id} children={len(child_ids)}"
    )
    for k in seq_bad:
        print(f"      x {k}")
    if seq_bad:
        failures.append(("SequentialPipeline", seq_bad))

    # Classifier-level InputSanitizer sidecar, including nested encoder artifact.
    v_san = build_sanitizer(VInputSanitizer, VColumnTransformer, VOrdinalEncoder)
    n_san = build_sanitizer(NInputSanitizer, NColumnTransformer, NOrdinalEncoder)
    expected_san_t = v_san.transform(SAN_PROBE)
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "vldm_sanitizer"
        vcodec.save_input_sanitizer(v_san, a, source_tabpfn_version=SRC["tabpfn_version"])
        san_from_vldm = ncodec.load_input_sanitizer(a)
        t_newlib_san = san_from_vldm.transform(SAN_PROBE)

        b = Path(d) / "newlib_sanitizer"
        ncodec.save_input_sanitizer(n_san, b, source_tabpfn_version=SRC["tabpfn_version"])
        san_from_newlib = vcodec.load_input_sanitizer(b)
        t_vldm_san = san_from_newlib.transform(SAN_PROBE)

    san_checks = {
        "vldm-written sanitizer -> newlib-load == vldm": nan_eq(t_newlib_san, expected_san_t),
        "newlib-written sanitizer -> vldm-load == vldm": nan_eq(t_vldm_san, expected_san_t),
        "vldm-written sanitizer metadata preserved": san_from_vldm.n_features_in == 2
        and san_from_vldm.inferred_categorical_indices == [1],
        "newlib-written sanitizer metadata preserved": san_from_newlib.n_features_in == 2
        and san_from_newlib.inferred_categorical_indices == [1],
    }
    san_bad = [k for k, ok in san_checks.items() if not ok]
    print(f"[{'OK' if not san_bad else 'FAIL'}] InputSanitizer(sidecar)")
    for k in san_bad:
        print(f"      x {k}")
    if san_bad:
        failures.append(("InputSanitizer", san_bad))

    print()
    if failures:
        print(f"PARITY FAILED for {len(failures)} case(s): {[n for n, _ in failures]}")
        return 1
    print(
        f"CROSS-IMPL PARITY: all {len(leaf_cases)} leaf cases + nested SequentialPipeline "
        "+ InputSanitizer sidecar identical bidirectionally (vldm <-> sklearn-tabpfn-ext), "
        "op_ids vldm.preprocessing.* at every level."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
