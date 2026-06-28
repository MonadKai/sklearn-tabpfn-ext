"""Pipeline glue for the ensemble preprocessing layer.

- :func:`build_estimator_pipeline` constructs a single
  :class:`sklearn.pipeline.Pipeline` for one ensemble member, composing the
  five vldm-specific transformers in the canonical order tabpfn uses
  (constant filter → quantile/SVD reshape → categorical encode → fingerprint
  → shuffle).
- :class:`EstimatorPipeline` is a typing alias for ``sklearn.pipeline.Pipeline``
  carrying our convention; vldm code uses this alias.
- :class:`PreprocessingPipeline` is the multi-estimator wrapper that holds
  ``n_estimators`` independent Pipelines. ``transform(X)`` returns one ndarray
  per estimator, ready to be stacked into the model's batch dim by the runner.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from sklearn.pipeline import Pipeline

from sklearn_tabpfn_ext.input_sanitizer import InputSanitizer

# Convention: an EstimatorPipeline is just an sklearn Pipeline composed of
# vldm preprocessing transformers. Reusing the type keeps joblib round-trips
# trivial — we never need a bespoke serialiser.
EstimatorPipeline: TypeAlias = Pipeline


def build_estimator_pipeline(
    *,
    constant_filter,
    quantile_svd_reshaper,
    categorical_encoder,
    fingerprint_adder,
    feature_shuffler,
) -> EstimatorPipeline:
    """Assemble the canonical 5-step pipeline for one ensemble member.

    All five components must already be fitted (or be no-op stubs) before
    calling ``transform`` on the returned Pipeline.
    """
    return Pipeline(
        steps=[
            ("constant_filter", constant_filter),
            ("quantile_svd", quantile_svd_reshaper),
            ("categorical", categorical_encoder),
            ("fingerprint", fingerprint_adder),
            ("shuffle", feature_shuffler),
        ]
    )


@dataclass
class PreprocessingPipeline:
    """Top-level CPU preprocessing — one ``EstimatorPipeline`` per ensemble member.

    Use :meth:`transform` to produce one preprocessed ndarray per estimator,
    in the same order as ``ensemble_pipelines``. The returned list is the
    contract between :mod:`vldm.preprocessing` and :mod:`vldm.worker.model_runner`.
    """

    ensemble_pipelines: list[EstimatorPipeline]
    n_jobs: int = 0  # 0 means inline (no thread pool); >0 enables ThreadPoolExecutor
    input_sanitizer: InputSanitizer | None = None

    @property
    def n_estimators(self) -> int:
        return len(self.ensemble_pipelines)

    def transform(self, X) -> list[np.ndarray]:
        if self.input_sanitizer is not None:
            X = self.input_sanitizer.transform(X)
        X = np.asarray(X, dtype=np.float64)

        if self.n_jobs <= 1 or self.n_estimators <= 1:
            return [np.asarray(p.transform(X), dtype=np.float32) for p in self.ensemble_pipelines]

        with ThreadPoolExecutor(max_workers=min(self.n_jobs, self.n_estimators)) as ex:
            outputs = list(
                ex.map(
                    lambda p: np.asarray(p.transform(X), dtype=np.float32),
                    self.ensemble_pipelines,
                )
            )
        return outputs

    def subset(self, indices: list[int]) -> PreprocessingPipeline:
        """Return a new :class:`PreprocessingPipeline` restricted to *indices*.

        The caller's order is preserved.  Clf-level fields (``input_sanitizer``,
        ``n_jobs``) are carried over unchanged — the sanitizer is
        estimator-independent and must not be silently dropped when slicing
        for a multi-worker memory-split (#114).
        """
        return PreprocessingPipeline(
            ensemble_pipelines=[self.ensemble_pipelines[i] for i in indices],
            input_sanitizer=self.input_sanitizer,
            n_jobs=self.n_jobs,
        )

    def __len__(self) -> int:
        return self.n_estimators
