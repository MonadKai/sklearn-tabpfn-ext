"""Round-trip tests for InputSanitizer sidecar save/load (codec.py)."""

import json

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder as SkOrdinalEncoder

from sklearn_tabpfn_ext.codec import load_input_sanitizer, save_input_sanitizer
from sklearn_tabpfn_ext.input_sanitizer import InputSanitizer
from sklearn_tabpfn_ext.tabpfn import translate_input_sanitizer

pytestmark = pytest.mark.tabpfn


class _FakeClf:
    pass


def test_sidecar_roundtrip_non_identity(tmp_path):
    rng = np.random.default_rng(3)
    n = 120
    X = np.column_stack([rng.normal(size=n), rng.integers(0, 5, size=n).astype(float)])
    ct = ColumnTransformer(
        transformers=[
            (
                "ordinal",
                SkOrdinalEncoder(
                    categories="auto",
                    dtype=np.float64,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=np.nan,
                ),
                [1],
            )
        ],
        remainder="passthrough",
    ).fit(X)

    clf = _FakeClf()
    clf.n_features_in_ = 2
    clf.ordinal_encoder_ = ct
    clf.inferred_categorical_indices_ = [1]
    san = translate_input_sanitizer(clf)

    out = tmp_path / "input_sanitizer"
    save_input_sanitizer(san, out, source_tabpfn_version="6.3.2")
    loaded = load_input_sanitizer(out)

    X_test = np.array([[0.1, 3.0], [0.2, 99.0]], dtype=np.float32)
    g, e = loaded.transform(X_test), san.transform(X_test)
    assert np.array_equal(np.isnan(g), np.isnan(e))
    m = ~np.isnan(e)
    assert np.array_equal(g[m], e[m])
    assert loaded.inferred_categorical_indices == [1]
    assert loaded.n_features_in == 2


def test_sidecar_roundtrip_identity(tmp_path):
    out = tmp_path / "input_sanitizer"
    save_input_sanitizer(InputSanitizer.identity(5), out, source_tabpfn_version="6.3.2")
    loaded = load_input_sanitizer(out)
    assert loaded.is_identity
    X = np.ones((2, 5), dtype=np.float32)
    assert np.array_equal(loaded.transform(X), X)


def test_sidecar_roundtrip_real_encoder_shape(tmp_path):
    """Roundtrip with FunctionTransformer() remainder — mirrors real tabpfn get_ordinal_encoder().

    tabpfn's get_ordinal_encoder() sets remainder=FunctionTransformer() (an object, not
    the string "passthrough"). The translator must normalise this to "passthrough" before
    saving; the loaded sanitizer must produce byte-exact output (NaN-position-aware).
    """
    rng = np.random.default_rng(7)
    n = 80
    X = np.column_stack(
        [
            rng.normal(size=n),  # col 0: numeric passthrough
            rng.integers(0, 4, size=n).astype(float),  # col 1: categorical
        ]
    )
    # Mirror real tabpfn get_ordinal_encoder() shape: OrdinalEncoder on col 1,
    # remainder=FunctionTransformer() (identity object, NOT the string).
    ct = ColumnTransformer(
        transformers=[
            (
                "ordinal",
                SkOrdinalEncoder(
                    categories="auto",
                    dtype=np.float64,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=np.nan,
                ),
                [1],
            )
        ],
        remainder=FunctionTransformer(),
    ).fit(X)

    clf = _FakeClf()
    clf.n_features_in_ = 2
    clf.ordinal_encoder_ = ct
    clf.inferred_categorical_indices_ = [1]
    san = translate_input_sanitizer(clf)

    out = tmp_path / "input_sanitizer"
    save_input_sanitizer(san, out, source_tabpfn_version="6.3.2")
    loaded = load_input_sanitizer(out)

    # Probe: known value, unknown value (99.0), and NaN
    X_probe = np.array(
        [
            [0.5, 2.0],  # known categorical
            [1.3, 99.0],  # unknown -> should encode as -1
            [2.1, np.nan],  # missing -> should encode as NaN
        ],
        dtype=np.float32,
    )

    g = loaded.transform(X_probe)
    e = san.transform(X_probe)

    # NaN positions must match exactly
    assert np.array_equal(np.isnan(g), np.isnan(e)), (
        f"NaN positions differ:\npre-save:\n{e}\nloaded:\n{g}"
    )
    # Non-NaN values must be byte-exact
    m = ~np.isnan(e)
    assert np.array_equal(g[m], e[m]), f"Non-NaN values differ:\npre-save:\n{e}\nloaded:\n{g}"
    assert loaded.inferred_categorical_indices == [1]
    assert loaded.n_features_in == 2


def test_load_rejects_unknown_format_version(tmp_path):
    out = tmp_path / "input_sanitizer"
    save_input_sanitizer(InputSanitizer.identity(3), out, source_tabpfn_version="6.3.2")
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["format_version"] = 999
    meta_path.write_text(json.dumps(meta, indent=2))
    with pytest.raises(ValueError):
        load_input_sanitizer(out)
