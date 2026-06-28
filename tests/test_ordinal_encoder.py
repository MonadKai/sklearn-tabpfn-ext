import numpy as np
from sklearn.preprocessing import OrdinalEncoder as SkOrdinalEncoder

from sklearn_tabpfn_ext import OrdinalEncoder


def _fitted_sklearn(X):
    oe = SkOrdinalEncoder(
        categories="auto",
        dtype=np.float64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,
    )
    oe.fit(X)
    return oe


def test_matches_sklearn_including_unknown_and_missing():
    rng = np.random.default_rng(0)
    X_fit = np.column_stack(
        [
            rng.integers(0, 5, size=200).astype(np.float64),
            rng.integers(10, 13, size=200).astype(np.float64),
        ]
    )
    sk = _fitted_sklearn(X_fit)

    v = OrdinalEncoder.from_sklearn(sk)

    X_test = np.array(
        [
            [3.0, 11.0],  # known
            [99.0, 12.0],  # unknown in col0 -> -1
            [np.nan, 10.0],  # missing in col0 -> nan
            [0.0, 999.0],  # unknown in col1 -> -1
        ],
        dtype=np.float64,
    )

    got = v.transform(X_test)
    expect = sk.transform(X_test)
    assert got.dtype == np.float64
    assert np.array_equal(np.isnan(got), np.isnan(expect))
    m = ~np.isnan(expect)
    assert np.array_equal(got[m], expect[m])


def test_state_roundtrip_ragged_categories():
    rng = np.random.default_rng(1)
    X_fit = np.column_stack(
        [
            rng.integers(0, 7, size=100).astype(np.float64),  # 7 cats
            rng.integers(0, 2, size=100).astype(np.float64),  # 2 cats (ragged vs col0)
        ]
    )
    v = OrdinalEncoder.from_sklearn(_fitted_sklearn(X_fit))
    state = v._to_state_dict()
    restored = OrdinalEncoder(**v._init_params_dict())
    OrdinalEncoder._from_state_dict(restored, state)
    X_test = X_fit[:10]
    assert np.array_equal(restored.transform(X_test), v.transform(X_test))


def test_codec_roundtrip_nan_safe(tmp_path):
    """Codec save/load round-trip: encoded_missing_value=nan must survive JSON
    serialisation (JSON null -> None) and be restored as float nan on reload.

    col0 has NaN in training data (nan ends up in fitted categories_).
    col1 has no NaN (plain integer categories only).

    This test would FAIL if _nan_init_param_keys were absent (C1) because
    codec.load would pass None as encoded_missing_value, breaking the NaN path.
    """
    from sklearn_tabpfn_ext.codec import load, save
    from sklearn_tabpfn_ext.composite.sequential import SequentialPipeline

    rng = np.random.default_rng(42)
    # col0: integers 0-4 + some NaN rows; col1: integers 10-12 (no NaN)
    col0 = rng.integers(0, 5, size=80).astype(np.float64)
    col1 = rng.integers(10, 13, size=80).astype(np.float64)
    col0[[3, 17, 55]] = np.nan  # NaN in col0 -> encoded_missing_value path
    X_fit = np.column_stack([col0, col1])

    sk = SkOrdinalEncoder(
        categories="auto",
        dtype=np.float64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,
    )
    sk.fit(X_fit)
    enc = OrdinalEncoder.from_sklearn(sk)

    # codec.save requires a SequentialPipeline root.
    pipeline = SequentialPipeline(steps=[("enc", enc)])

    save(pipeline, tmp_path / "artifact")
    reloaded_pipeline = load(tmp_path / "artifact")

    # Extract the reloaded encoder from the pipeline.
    reloaded_enc = reloaded_pipeline.steps[0][1]

    # encoded_missing_value must be nan, not None.
    assert np.isnan(reloaded_enc.encoded_missing_value), (
        "encoded_missing_value was not restored as nan after codec load; "
        "likely _nan_init_param_keys is missing (C1)"
    )

    # Probe: known value, unknown value, NaN per column.
    X_probe = np.array(
        [
            [2.0, 11.0],  # both known
            [99.0, 12.0],  # col0 unknown -> -1
            [np.nan, 10.0],  # col0 NaN    -> encoded_missing_value (nan)
            [1.0, 999.0],  # col1 unknown -> -1
        ],
        dtype=np.float64,
    )

    got_orig = enc.transform(X_probe)
    got_reloaded = reloaded_enc.transform(X_probe)

    # NaN positions must match.
    assert np.array_equal(np.isnan(got_orig), np.isnan(got_reloaded)), (
        "NaN positions differ between original and reloaded encoder"
    )
    # Non-NaN positions must be byte-identical.
    mask = ~np.isnan(got_orig)
    assert np.array_equal(got_orig[mask], got_reloaded[mask]), (
        "Non-NaN values differ between original and reloaded encoder"
    )
