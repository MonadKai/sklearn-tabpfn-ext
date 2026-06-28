"""vldm-native OrdinalEncoder leaf.

Byte-faithful reimplementation of
``sklearn.preprocessing.OrdinalEncoder(categories="auto",
handle_unknown="use_encoded_value", unknown_value=-1,
encoded_missing_value=np.nan, dtype=float64)`` — the exact config used by
``tabpfn ... get_ordinal_encoder``. We reimplement (rather than subclass
sklearn) because ``categories_`` is a ragged list of arrays that does not
round-trip through the rectangular npz codec; here it is serialised as a flat
values array + per-column lengths. vldm imports zero tabpfn at runtime.
"""

from __future__ import annotations

import numpy as np

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.ordinal_encoder.OrdinalEncoder")
class OrdinalEncoder(VldmEstimatorMixin):
    """Ordinal-encode columns against fitted sorted categories.

    Attributes
    ----------
    categories_ : list[np.ndarray]   # one sorted float64 array per column
    n_features_in_ : int
    """

    _init_param_keys = ("unknown_value", "encoded_missing_value")
    # encoded_missing_value defaults to float('nan'), which JSON serialises as
    # null (None).  Declare it here so codec.load converts None -> np.nan before
    # constructing the object.  See simple_imputer.py for the same pattern.
    _nan_init_param_keys = ("encoded_missing_value",)
    # categories_ is ragged -> custom (de)serialisation below; n_features_in_
    # is derived from categories_ so it is not a separate state key.

    # Fitted attributes set by from_sklearn / _from_state_dict.
    # Declared here (annotation-only, no default) so mypy can resolve them.
    categories_: list[np.ndarray]
    n_features_in_: int
    _cats_has_nan_: list[bool]

    def __init__(
        self, *, unknown_value: float = -1.0, encoded_missing_value: float = float("nan")
    ) -> None:
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value

    @classmethod
    def from_sklearn(cls, sk) -> OrdinalEncoder:
        obj = cls(
            unknown_value=float(sk.unknown_value),
            encoded_missing_value=float(sk.encoded_missing_value),
        )
        # An OrdinalEncoder applied to ZERO categorical columns (common for
        # numeric-only models whose ColumnTransformer selects no columns) is
        # never fitted, so it has no ``categories_``. Translate it as a no-op
        # (encodes nothing) instead of crashing on the missing attribute.
        cats = getattr(sk, "categories_", None)
        if cats is None:
            obj.categories_ = []
            obj.n_features_in_ = 0
            obj._cats_has_nan_ = []
            return obj
        obj.categories_ = [np.asarray(c, dtype=np.float64) for c in cats]
        obj.n_features_in_ = int(sk.n_features_in_)
        obj._cats_has_nan_ = [bool(np.any(np.isnan(c))) for c in obj.categories_]
        return obj

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape, dtype=np.float64)
        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            # Guard: empty categories -> all unknown.
            if len(cats) == 0:
                out[:, j] = self.unknown_value
                continue
            # exact-match lookup against sorted categories
            idx = np.searchsorted(cats, col)
            idx_clipped = np.clip(idx, 0, len(cats) - 1)
            matched = cats[idx_clipped] == col
            codes = np.where(matched, idx_clipped.astype(np.float64), self.unknown_value)
            # missing (NaN input) -> encoded_missing_value, but only when NaN
            # was seen during fit (i.e. NaN is in cats). When NaN is absent
            # from the fitted categories, a NaN input falls through to the
            # unknown_value path (same as sklearn behaviour).
            # Use the precomputed per-column flag to avoid recomputing each call.
            if self._cats_has_nan_[j]:
                codes = np.where(np.isnan(col), self.encoded_missing_value, codes)
            out[:, j] = codes
        return out

    # ragged-safe (de)serialisation: flat values + per-column lengths
    def _to_state_dict(self) -> dict[str, np.ndarray]:
        cats = [np.asarray(c, dtype=np.float64) for c in self.categories_]
        lengths = np.array([len(c) for c in cats], dtype=np.int64)
        values = np.concatenate(cats) if cats else np.empty(0, dtype=np.float64)
        return {"categories_lengths_": lengths, "categories_values_": values}

    @classmethod
    def _from_state_dict(cls, obj: OrdinalEncoder, state: dict[str, np.ndarray]) -> None:
        lengths = np.asarray(state["categories_lengths_"], dtype=np.int64)
        values = np.asarray(state["categories_values_"], dtype=np.float64)
        cats, pos = [], 0
        for n in lengths:
            cats.append(values[pos : pos + int(n)])
            pos += int(n)
        obj.categories_ = cats
        obj.n_features_in_ = len(cats)
        obj._cats_has_nan_ = [bool(np.any(np.isnan(c))) for c in cats]
