"""vldm equivalent of tabpfn.preprocessing.steps.SquashingScaler.

Algorithm: adapted from skrub SquashingScaler (used in RealMLP, arXiv:2407.04491).

Each column is categorised at fit time into one of three regimes:

* zero_cols   – max == min (constant column): output set to 0.
* minmax_cols – upper/lower quantiles are equal but col is non-constant:
                use a custom MinMaxScaler (median-centred, range-scaled).
* robust_cols – general case: use RobustScaler (median + IQR).

After scaling, a smooth soft-clip is applied column-wise:

    x_out = x / sqrt(1 + (x / max_absolute_value)^2)

Infinite inputs are mapped to ±max_absolute_value; NaN is preserved.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_is_fitted

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.exceptions import OpStateError
from sklearn_tabpfn_ext.registry import register

# ---------------------------------------------------------------------------
# Internal helpers (ported from TabPFN reference, no tabpfn import)
# ---------------------------------------------------------------------------


def _mask_inf(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace infinite values with NaN and return their sign mask."""
    mask_inf = np.isinf(X)
    if mask_inf.any():
        sign = np.sign(X)
        X = np.where(mask_inf, np.nan, X)
        mask_inf = mask_inf.astype(X.dtype) * sign
    return X, mask_inf


def _set_zeros(X: np.ndarray, zero_cols: np.ndarray) -> np.ndarray:
    """Set finite values in constant columns to zero."""
    mask = np.isfinite(X)
    mask[:, ~zero_cols] = False
    X[mask] = 0.0
    return X


def _soft_clip(X: np.ndarray, max_absolute_value: float, mask_inf: np.ndarray) -> np.ndarray:
    """Smooth squashing: x / sqrt(1 + (x/B)^2), inf -> ±B."""
    X = X / np.sqrt(1 + (X / max_absolute_value) ** 2)
    X = np.where(mask_inf == 1, max_absolute_value, X)
    return np.where(mask_inf == -1, -max_absolute_value, X)


class _MinMaxScaler:
    """Median-centred min-max scaler (ported from TabPFN reference).

    Transform: scale * (X - median)
    where scale = 2 / (max - min + eps).
    """

    def fit(self, X: np.ndarray) -> _MinMaxScaler:
        eps = np.finfo("float32").tiny
        self.median_ = np.nanmedian(X, axis=0)
        self.scale_ = 2.0 / (np.nanmax(X, axis=0) - np.nanmin(X, axis=0) + eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scale_ * (X - self.median_)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


@register("vldm.preprocessing.squashing_scaler.SquashingScaler")
class SquashingScaler(VldmEstimatorMixin, BaseEstimator, TransformerMixin):
    """Robust scaling + smooth soft-clip squashing.

    Matches the behaviour of the TabPFN SquashingScaler reference implementation.

    Parameters
    ----------
    max_absolute_value : float, default=3.0
        Maximum absolute value the output can take (boundary of the soft-clip).
    quantile_range : tuple of float, default=(25.0, 75.0)
        Percentile range (0-100 scale) used by the RobustScaler and for
        detecting constant-quantile columns.
    """

    # Fitted attributes serialised to state.npz.
    # Scaler sub-objects are decomposed into raw arrays to avoid pickling.
    _state_keys = (
        "robust_cols_",
        "minmax_cols_",
        "zero_cols_",
        # RobustScaler internals (None → stored as 0-len sentinel)
        "robust_center_",
        "robust_scale_",
        # _MinMaxScaler internals (None → stored as 0-len sentinel)
        "minmax_median_",
        "minmax_scale_",
        "n_features_in_",
    )
    _init_param_keys = ("max_absolute_value", "quantile_range")

    def __init__(
        self,
        max_absolute_value: float = 3.0,
        quantile_range: tuple[float, float] = (25.0, 75.0),
    ) -> None:
        self.max_absolute_value = max_absolute_value
        self.quantile_range = quantile_range

    # ------------------------------------------------------------------
    # sklearn fit / transform
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        X, _ = _mask_inf(X)  # replace inf with NaN for fitting

        # Categorise columns.
        zero_cols = np.nanmax(X, axis=0) == np.nanmin(X, axis=0)

        quantiles = np.nanpercentile(X, self.quantile_range, axis=0)
        minmax_cols = (quantiles[0, :] == quantiles[1, :]) & ~zero_cols
        robust_cols = ~(minmax_cols | zero_cols)

        self.zero_cols_ = zero_cols
        self.minmax_cols_ = minmax_cols
        self.robust_cols_ = robust_cols
        self.n_features_in_ = X.shape[1]

        if robust_cols.any():
            rs = RobustScaler(
                with_centering=True,
                with_scaling=True,
                quantile_range=self.quantile_range,
                copy=True,
            ).fit(X[:, robust_cols])
            self.robust_center_ = rs.center_.astype(np.float64)
            self.robust_scale_ = rs.scale_.astype(np.float64)
        else:
            self.robust_center_ = np.empty(0, dtype=np.float64)
            self.robust_scale_ = np.empty(0, dtype=np.float64)

        if minmax_cols.any():
            mms = _MinMaxScaler().fit(X[:, minmax_cols])
            self.minmax_median_ = mms.median_.astype(np.float64)
            self.minmax_scale_ = mms.scale_.astype(np.float64)
        else:
            self.minmax_median_ = np.empty(0, dtype=np.float64)
            self.minmax_scale_ = np.empty(0, dtype=np.float64)

        return self

    def transform(self, X):
        check_is_fitted(self, ["robust_cols_", "minmax_cols_", "zero_cols_"])
        X = np.asarray(X, dtype=np.float64)
        X, mask_inf = _mask_inf(X)

        X_tr = X.copy()

        if self.robust_cols_.any():
            X_tr[:, self.robust_cols_] = (
                X[:, self.robust_cols_] - self.robust_center_
            ) / self.robust_scale_

        if self.minmax_cols_.any():
            X_tr[:, self.minmax_cols_] = self.minmax_scale_ * (
                X[:, self.minmax_cols_] - self.minmax_median_
            )

        if self.zero_cols_.any():
            X_tr = _set_zeros(X_tr, self.zero_cols_)

        return _soft_clip(X_tr, self.max_absolute_value, mask_inf)

    # ------------------------------------------------------------------
    # State-dict overrides: flatten / restore sub-scaler arrays
    # ------------------------------------------------------------------

    def _to_state_dict(self) -> dict[str, np.ndarray]:
        """Serialise: booleans stored as uint8, sub-scaler arrays as float64."""
        state = {}
        bool_keys = {"robust_cols_", "minmax_cols_", "zero_cols_"}
        for key in self._state_keys:
            val = getattr(self, key)
            arr = np.asarray(val)
            if key in bool_keys:
                state[key] = arr.astype(np.uint8)
            else:
                state[key] = arr
        return state

    @classmethod
    def _from_state_dict(cls, obj: SquashingScaler, state: dict[str, np.ndarray]) -> None:
        """Restore: booleans recovered from uint8."""
        bool_keys = {"robust_cols_", "minmax_cols_", "zero_cols_"}
        for key in cls._state_keys:
            if key not in state:
                raise OpStateError(op_path=cls.__qualname__, key=key, reason="missing in state.npz")
            val = state[key]
            if key in bool_keys:
                setattr(obj, key, val.astype(bool))
            elif val.shape == () and val.dtype.kind in "iuf":
                setattr(obj, key, val.item())
            else:
                setattr(obj, key, val)
