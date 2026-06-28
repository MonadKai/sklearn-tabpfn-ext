"""Categorical-feature encoder, vldm flavour.

For models with no categorical features (like the tab2.5 binary tabular family
we currently target), :class:`IdentityCategoricalEncoder` is a safe no-op that
preserves the canonical 5-step pipeline shape.

For models with categorical features the conversion script wraps tabpfn's
fitted ``categorical_transformer_`` (typically an
``sklearn.compose.ColumnTransformer``) inside
:class:`CategoricalOrdinalEncoder` — vldm doesn't reimplement the
ordinal-encoding math, just adopts the fitted sklearn transformer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.exceptions import ArtifactSchemaError, OpStateError
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.categorical.IdentityCategoricalEncoder")
class IdentityCategoricalEncoder(VldmEstimatorMixin, BaseEstimator, TransformerMixin):
    """No-op placeholder for tabular models without categorical features."""

    _state_keys = ("n_features_in_", "n_features_out_")
    _init_param_keys = ()

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features)


@register("vldm.preprocessing.categorical.CategoricalOrdinalEncoder")
class CategoricalOrdinalEncoder(VldmEstimatorMixin, BaseEstimator, TransformerMixin):
    """Wraps a fitted sklearn ColumnTransformer (or similar) plus
    tabpfn's ``random_mappings_`` per-feature shuffle dict.

    Parameters
    ----------
    column_transformer : Any
        A fitted ``sklearn.compose.ColumnTransformer`` (or any sklearn
        transformer) handling the categorical columns.
    random_mappings : dict[int, np.ndarray] or None
        Per-feature category permutations applied on top of the base
        transform. Mirrors tabpfn's ``random_mappings_``.

    Attributes
    ----------
    column_transformer_ : Any
    random_mappings_ : dict[int, np.ndarray]
    n_features_in_ : int
    """

    _state_keys = ("n_features_in_",)
    _init_param_keys = ()
    # column_transformer_ is a vldm composite; codec.save/load restores it via
    # _child_attrs. random_mappings_ (dict[int, ndarray]) is not a vldm operator,
    # so it is serialised separately by this class's _to_state_dict/_from_state_dict
    # (as rm_keys + rm_val_{k}; see below). Categorical-feature support (translator
    # factory + codec round-trip + random_mappings_) is fully implemented.
    _child_attrs = ("column_transformer_",)
    # Forward-guard op: its serialised artifacts always carry serialised_child_attrs,
    # which an older (extra="forbid") reader rejects at load instead of mis-reading.
    _forward_guard = True

    def __init__(
        self,
        *,
        column_transformer: Any | None = None,
        random_mappings: dict[int, np.ndarray] | None = None,
    ) -> None:
        self.column_transformer = column_transformer
        self.random_mappings = random_mappings

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        self.column_transformer_ = self.column_transformer
        self.random_mappings_ = dict(self.random_mappings) if self.random_mappings else {}
        if self.column_transformer_ is not None:
            self.column_transformer_.fit(X, y)
        return self

    def transform(self, X):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("CategoricalOrdinalEncoder not fitted")
        if self.column_transformer_ is None:
            return np.asarray(X)
        out = self.column_transformer_.transform(X)
        for col, perm in self.random_mappings_.items():
            col = int(col)
            perm = np.asarray(perm, dtype=np.int64)
            mask = ~np.isnan(out[:, col])
            idx = out[mask, col].astype(np.int64)
            out[mask, col] = perm[np.clip(idx, 0, len(perm) - 1)].astype(out.dtype)
        return out

    def _to_state_dict(self):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("CategoricalOrdinalEncoder not fitted")
        out = {"n_features_in_": np.asarray(self.n_features_in_)}
        rm = getattr(self, "random_mappings_", None) or {}
        rm = {int(k): v for k, v in rm.items()}  # normalize (JSON/user keys may be str)
        keys = sorted(rm)
        out["rm_keys"] = np.asarray(keys, dtype=np.int64)
        for k in keys:
            out[f"rm_val_{k}"] = np.asarray(rm[k], dtype=np.int64)
        return out

    @classmethod
    def _from_state_dict(cls, obj, state):
        if "n_features_in_" not in state:
            raise OpStateError(
                op_path=cls.__qualname__, key="n_features_in_", reason="missing in state.npz"
            )
        obj.n_features_in_ = int(np.asarray(state["n_features_in_"]).item())

        if "rm_keys" not in state:
            raise OpStateError(
                op_path=cls.__qualname__, key="rm_keys", reason="missing in state.npz"
            )
        rm_keys = np.asarray(state["rm_keys"])
        if rm_keys.ndim != 1 or rm_keys.dtype.kind not in "iu":
            raise ArtifactSchemaError(
                f"{cls.__qualname__}: rm_keys must be 1-D integer, got "
                f"shape={rm_keys.shape} dtype={rm_keys.dtype}"
            )
        keys = [int(k) for k in rm_keys.tolist()]
        if len(set(keys)) != len(keys):
            raise ArtifactSchemaError(f"{cls.__qualname__}: rm_keys has duplicates: {keys}")
        if keys != sorted(keys):
            raise ArtifactSchemaError(f"{cls.__qualname__}: rm_keys not sorted ascending: {keys}")

        present = {s for s in state if s.startswith("rm_val_")}
        expected = {f"rm_val_{k}" for k in keys}
        orphan = present - expected
        if orphan:
            raise ArtifactSchemaError(
                f"{cls.__qualname__}: orphan rm_val_* keys not in rm_keys: {sorted(orphan)}"
            )

        rm: dict[int, np.ndarray] = {}
        for k in keys:
            vk = f"rm_val_{k}"
            if vk not in state:
                raise ArtifactSchemaError(f"{cls.__qualname__}: missing {vk} for rm_keys entry {k}")
            v = np.asarray(state[vk])
            if v.ndim != 1 or v.dtype.kind not in "iu":
                raise ArtifactSchemaError(
                    f"{cls.__qualname__}: {vk} must be 1-D integer, got "
                    f"shape={v.shape} dtype={v.dtype}"
                )
            n = int(v.shape[0])
            if n > 0 and (int(v.min()) < 0 or int(v.max()) >= n):
                raise ArtifactSchemaError(
                    f"{cls.__qualname__}: {vk} values out of range [0,{n - 1}]: {v.tolist()}"
                )
            rm[k] = v.astype(np.int64)
        obj.random_mappings_ = rm
