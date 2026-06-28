"""Regression test: PreprocessingPipeline.subset must carry clf-level fields.

When ``from_pretrained`` slices estimators for a multi-worker memory-split
the ``input_sanitizer`` and ``n_jobs`` are clf-level (estimator-independent)
and must be preserved on the sliced pipeline.  Without the fix the sanitizer
was silently dropped, causing the #114 argmax-flip bug to recur on f200
models served multi-worker.

CPU-only — no GPU or tabpfn dependency required.
"""

from __future__ import annotations

import numpy as np

from sklearn_tabpfn_ext.pipeline import PreprocessingPipeline

# ---------------------------------------------------------------------------
# Minimal stand-ins
# ---------------------------------------------------------------------------


class _Echo:
    """Identity EstimatorPipeline stand-in."""

    def transform(self, X):
        return np.asarray(X)


class _SwapDtype:
    """Non-identity sanitizer stand-in (is_identity=False)."""

    is_identity = False

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_subset_preserves_input_sanitizer_and_njobs():
    """subset() must carry input_sanitizer and n_jobs to the sliced pipeline."""
    sanitizer = _SwapDtype()
    pp = PreprocessingPipeline(
        ensemble_pipelines=[_Echo(), _Echo(), _Echo()],
        input_sanitizer=sanitizer,
        n_jobs=2,
    )

    sub = pp.subset([0, 2])

    # sanitizer identity — same object, not a copy
    assert sub.input_sanitizer is pp.input_sanitizer
    # n_jobs preserved
    assert sub.n_jobs == 2
    # correct subset of pipelines, in caller order
    assert len(sub.ensemble_pipelines) == 2
    assert sub.ensemble_pipelines[0] is pp.ensemble_pipelines[0]
    assert sub.ensemble_pipelines[1] is pp.ensemble_pipelines[2]


def test_subset_drops_sanitizer_if_original_has_none():
    """When input_sanitizer is None, the subset must also have None."""
    pp = PreprocessingPipeline(
        ensemble_pipelines=[_Echo(), _Echo()],
        input_sanitizer=None,
        n_jobs=0,
    )
    sub = pp.subset([1])
    assert sub.input_sanitizer is None
    assert sub.n_jobs == 0


def test_subset_regression_broken_pattern():
    """Demonstrates that the old inline re-wrap (missing sanitizer=) would fail.

    This test constructs the broken equivalent inline and asserts the sanitizer
    IS lost — proving the old code path was wrong — and then verifies the fixed
    ``subset()`` method restores it.
    """
    sanitizer = _SwapDtype()
    pp = PreprocessingPipeline(
        ensemble_pipelines=[_Echo(), _Echo(), _Echo()],
        input_sanitizer=sanitizer,
        n_jobs=3,
    )

    # Simulate the OLD broken re-wrap (no input_sanitizer= / no n_jobs=)
    broken = PreprocessingPipeline(ensemble_pipelines=[pp.ensemble_pipelines[i] for i in [0, 2]])
    assert broken.input_sanitizer is None, "old pattern silently drops sanitizer"
    assert broken.n_jobs == 0, "old pattern silently drops n_jobs"

    # The fixed path
    fixed = pp.subset([0, 2])
    assert fixed.input_sanitizer is sanitizer
    assert fixed.n_jobs == 3
