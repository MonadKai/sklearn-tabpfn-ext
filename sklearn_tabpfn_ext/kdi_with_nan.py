"""vldm equivalent of tabpfn.preprocessing.steps.KDITransformerWithNaN.

The original class falls back to sklearn.preprocessing.PowerTransformer
when the optional `kditransform` package is unavailable. We don't carry
the kditransform path — by the time a checkpoint reaches vldm conversion
the fit has already happened, and runtime only needs transform. We keep
PowerTransformer's behavior so transform output matches.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.exceptions import OpStateError
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.kdi_with_nan.KDITransformerWithNaN")
class KDITransformerWithNaN(VldmEstimatorMixin, PowerTransformer):
    # See note in vldm/preprocessing/safe_power.py — same rationale for
    # persisting PowerTransformer's internal _scaler state.
    _state_keys = (
        "lambdas_",
        "n_features_in_",
        "_scaler_mean_",
        "_scaler_scale_",
        "_scaler_var_",
        "_scaler_n_features_in_",
        "_scaler_n_samples_seen_",
    )
    _init_param_keys = ("method", "standardize")

    def _to_state_dict(self) -> dict[str, np.ndarray]:
        """Extract fitted state, including _scaler if present."""
        out = {}
        for key in ("lambdas_", "n_features_in_"):
            value = getattr(self, key)
            out[key] = np.asarray(value)

        # If standardize=True, _scaler exists and we need to save its state.
        if self.standardize and hasattr(self, "_scaler"):
            scaler = self._scaler
            out["_scaler_mean_"] = np.asarray(scaler.mean_)
            out["_scaler_scale_"] = np.asarray(scaler.scale_)
            out["_scaler_var_"] = np.asarray(scaler.var_)
            out["_scaler_n_features_in_"] = np.asarray(scaler.n_features_in_)
            out["_scaler_n_samples_seen_"] = np.asarray(scaler.n_samples_seen_)

        return out

    @classmethod
    def _from_state_dict(cls, obj: KDITransformerWithNaN, state: dict[str, np.ndarray]) -> None:
        """Restore fitted state, reconstructing _scaler if needed."""
        # Restore main attributes.
        for key in ("lambdas_", "n_features_in_"):
            if key not in state:
                raise OpStateError(
                    op_path=type(obj).__qualname__,
                    key=key,
                    reason="missing in state.npz",
                )
            value = state[key]
            if value.shape == () and value.dtype.kind in "iuf":
                value = value.item()
            setattr(obj, key, value)

        # Restore _scaler if standardize=True and _scaler_mean_ is in state.
        if obj.standardize and "_scaler_mean_" in state:
            scaler = StandardScaler()
            scaler.mean_ = state["_scaler_mean_"].astype(np.float64)
            scaler.scale_ = state["_scaler_scale_"].astype(np.float64)
            scaler.var_ = state["_scaler_var_"].astype(np.float64)
            scaler.n_features_in_ = int(state["_scaler_n_features_in_"].item())
            scaler.n_samples_seen_ = int(state["_scaler_n_samples_seen_"].item())
            obj._scaler = scaler
