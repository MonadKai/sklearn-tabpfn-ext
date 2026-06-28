"""SourceMeta accepts the stage-2 provenance fields (#38) and stays
backward-compatible with stage-1 artifacts (all new fields Optional)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sklearn_tabpfn_ext.schema import SourceMeta


def test_source_meta_accepts_provenance_fields():
    m = SourceMeta(
        extracted_at="2026-05-25T00:00:00+00:00",
        estimator_index=0,
        sklearn_version="1.6.1",
        numpy_version="2.1.0",
        torch_version="2.5.0",
        python_version="3.12.7",
        n_estimators=1,
        n_features=147,
        n_train_rows=6000,
        training_hyperparams={"softmax_temperature": 0.9},
    )
    assert m.sklearn_version == "1.6.1"
    assert m.n_train_rows == 6000
    assert m.training_hyperparams == {"softmax_temperature": 0.9}


def test_source_meta_provenance_fields_default_none():
    """Stage-1 artifacts (no provenance) still validate; new fields are None."""
    m = SourceMeta(extracted_at="2026-05-25T00:00:00+00:00", estimator_index=0)
    assert m.sklearn_version is None
    assert m.n_estimators is None
    assert m.training_hyperparams is None


def test_source_meta_still_rejects_unknown_key():
    """extra='forbid' is preserved: an unknown key alongside provenance fails."""
    with pytest.raises(ValidationError):
        SourceMeta(
            extracted_at="2026-05-25T00:00:00+00:00",
            estimator_index=0,
            sklearn_version="1.6.1",
            totally_unknown_field="nope",
        )
