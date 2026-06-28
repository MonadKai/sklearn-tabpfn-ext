"""Classifier-level input sanitizer (tabpfn fix_dtypes + ordinal_encoder_).

Runs once on raw X before per-estimator preprocessing. Identity (no inferred
categoricals) is a literal passthrough so unaffected models stay byte-identical.
See docs/superpowers/specs/2026-06-05-vldm-input-sanitizer-design.md.
"""

from __future__ import annotations

import numpy as np

from sklearn_tabpfn_ext.composite import ColumnTransformer


class InputSanitizer:
    """Holds the translated ordinal-encoder ColumnTransformer (or identity)."""

    def __init__(
        self,
        *,
        n_features_in: int,
        inferred_categorical_indices: list[int],
        column_transformer: ColumnTransformer | None,
    ) -> None:
        self.n_features_in = int(n_features_in)
        self.inferred_categorical_indices = list(inferred_categorical_indices)
        self.column_transformer = column_transformer  # None => identity

    @property
    def is_identity(self) -> bool:
        return self.column_transformer is None

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.is_identity:
            return X  # literal passthrough: byte-identical to pre-fix behaviour
        # tabpfn fix_dtypes casts to float64 before the ColumnTransformer; the
        # OrdinalEncoder does exact-match lookup, so the cast must precede it.
        # float64 cast also covers tabpfn's numeric fix_dtypes path; vldm serves
        # numeric inputs only — text/NaN-string sanitization is out of scope.
        # column_transformer is guaranteed non-None here: is_identity (== column_transformer is None)
        # is checked above and returns early, so execution only reaches this line when non-None.
        assert self.column_transformer is not None
        return np.asarray(self.column_transformer.transform(np.asarray(X, dtype=np.float64)))

    @classmethod
    def identity(cls, n_features_in: int) -> InputSanitizer:
        return cls(
            n_features_in=n_features_in, inferred_categorical_indices=[], column_transformer=None
        )
