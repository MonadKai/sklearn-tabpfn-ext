"""Append a deterministic per-row fingerprint feature.

Two algorithm variants are supported, controlled by the ``algo`` parameter:

v63 (default)
    Bit-for-bit faithful to ``tabpfn.preprocessing.steps.AddFingerprintFeaturesStep``
    as shipped in tabpfn **6.3.x**.  Algorithm:

    1. Compute ``salted = X + rnd_salt_`` (broadcast scalar add).
    2. For each row, hash ``(X[i] + rnd_salt_).tobytes()`` via SHA-256.
    3. Take ``int(hex_digest, 16) % 10**12 / 10**12`` as the float fingerprint.
    4. Append as a new column to ``X`` (raw — not salted).

v64
    Mirrors the new algorithm introduced in tabpfn **6.4.0**.  Algorithm:

    1. ``n_cells_ = n_rows * n_cols`` is used as a deterministic salt (stored at fit).
    2. For each row: ``data = np.around(X[i], decimals=12).tobytes()
       + n_cells_.to_bytes(8, "little", signed=False)``
    3. ``h = int(sha256(data).hexdigest(), 16)``
    4. Fingerprint value = ``(h & (2**64 - 1)) / (2**64 - 1)``
    5. Append as a new column to ``X``.

Serialisation
    ``algo`` is stored in ``_init_param_keys`` so the codec reconstructs the
    correct variant.  State keys differ by variant:

    - v63: ``("rnd_salt_", "n_features_in_")``
    - v64: ``("n_cells_", "n_features_in_")``

    Old artifacts serialised without ``algo`` reconstruct with the default
    ``algo="v63"`` and restore v63 state keys — byte-identical to the old code.
"""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.exceptions import OpStateError
from sklearn_tabpfn_ext.registry import register

# --- v63 constants ---
_HASH_MOD = 10**12

# --- v64 constants ---
_V64_CONSTANT = 2**64 - 1  # mask for 64-bit truncation
_V64_HASH_ROUND_DECIMALS = 12


def _float_hash_arr_v63(arr: np.ndarray) -> float:
    """SHA-256 of ``arr.tobytes()`` mod 10^12 / 10^12, matching tabpfn 6.3.x."""
    # Hash the float64 byte layout (tabpfn's pipeline is float64). Casting makes
    # the digest precision-independent: a float32-served row (4-byte tobytes)
    # would otherwise hash differently and break fingerprint parity.
    arr = np.asarray(arr, dtype=np.float64)
    h = int(hashlib.sha256(arr.tobytes()).hexdigest(), 16)
    return (h % _HASH_MOD) / _HASH_MOD


def _float_hash_arr_v64(arr: np.ndarray, offset: int = 0) -> float:
    """SHA-256 with optional integer salt appended as raw 8-byte little-endian.

    Matches tabpfn 6.4.x ``_float_hash_arr``:
      - round to 12 decimal places before hashing
      - if offset != 0: append offset.to_bytes(8, "little", signed=False)
      - mask to 64 bits: ``(h & (2**64-1)) / (2**64-1)``
    """
    # float64 byte layout (precision-independent; see _float_hash_arr_v63).
    arr = np.asarray(arr, dtype=np.float64)
    data = np.around(arr, decimals=_V64_HASH_ROUND_DECIMALS).tobytes()
    if offset != 0:
        data += offset.to_bytes(8, "little", signed=False)
    h = int(hashlib.sha256(data).hexdigest(), 16)
    return (h & _V64_CONSTANT) / _V64_CONSTANT


@register("vldm.preprocessing.fingerprint.FingerprintFeatureAdder")
class FingerprintFeatureAdder(VldmEstimatorMixin, BaseEstimator, TransformerMixin):
    """Appends one fingerprint column to the feature matrix.

    Parameters
    ----------
    salt : int or None
        Per-estimator salt (v63 only). When ``None``, ``fit`` draws one in
        ``[0, 2**16)`` (matching tabpfn 6.3.x). Ignored for v64.
    random_state : int, np.random.Generator, or None
        RNG seed for v63 salt drawing.
    algo : {"v63", "v64"}
        Fingerprint algorithm variant.  ``"v63"`` (default) is byte-identical
        to tabpfn 6.3.x; ``"v64"`` mirrors tabpfn 6.4.x.

    Attributes (v63)
    ----------------
    rnd_salt_ : int
    n_features_in_ : int

    Attributes (v64)
    ----------------
    n_cells_ : int  (n_rows * n_cols of the fit matrix)
    n_features_in_ : int
    """

    # _state_keys is NOT used as a class-level constant here; the instance-level
    # overrides in _to_state_dict / _from_state_dict select keys by algo.
    # We still declare a tuple so introspection tools see a sane default.
    _state_keys = ("rnd_salt_", "n_features_in_")
    _init_param_keys = ("salt", "algo")

    def __init__(
        self,
        *,
        salt: int | None = None,
        random_state: int | None = None,
        algo: str = "v63",
    ) -> None:
        self.salt = salt
        self.random_state = random_state
        self.algo = algo

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        if self.algo == "v64":
            self.n_cells_ = int(X.shape[0] * X.shape[1])
        else:
            # v63 path (default)
            if self.salt is not None:
                self.rnd_salt_ = int(self.salt)
            else:
                rng = np.random.default_rng(self.random_state)
                self.rnd_salt_ = int(rng.integers(0, 2**16))
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, X):
        X = np.asarray(X)
        if self.algo == "v64":
            return self._transform_v64(X)
        return self._transform_v63(X)

    def _transform_v63(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "rnd_salt_"):
            raise RuntimeError("FingerprintFeatureAdder not fitted")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        # tabpfn 6.3.x _transform with is_test=True:
        #   salted_X = X + rnd_salt_
        #   for row in salted_X: h = _float_hash_arr(row + rnd_salt_)
        # i.e. hash input = X + 2*rnd_salt_
        salted = X + self.rnd_salt_
        n = salted.shape[0]
        fp = np.empty((n, 1), dtype=X.dtype)
        for i in range(n):
            fp[i, 0] = _float_hash_arr_v63(salted[i] + self.rnd_salt_)
        return np.concatenate([X, fp], axis=1)

    def _transform_v64(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "n_cells_"):
            raise RuntimeError("FingerprintFeatureAdder not fitted")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        salt = self.n_cells_
        n = X.shape[0]
        fp = np.empty((n, 1), dtype=X.dtype)
        for i in range(n):
            fp[i, 0] = _float_hash_arr_v64(X[i], salt)
        return np.concatenate([X, fp], axis=1)

    # ------------------------------------------------------------------
    # State dict overrides — algo-aware
    # ------------------------------------------------------------------

    def _state_keys_for_algo(self) -> tuple[str, ...]:
        if self.algo == "v64":
            return ("n_cells_", "n_features_in_")
        return ("rnd_salt_", "n_features_in_")

    def _to_state_dict(self) -> dict:
        out: dict = {}
        for key in self._state_keys_for_algo():
            value = getattr(self, key)  # AttributeError if not fitted -- intended
            out[key] = np.asarray(value)
        return out

    @classmethod
    def _from_state_dict(cls, obj: FingerprintFeatureAdder, state: dict) -> None:
        keys = obj._state_keys_for_algo()
        for key in keys:
            if key not in state:
                raise OpStateError(
                    op_path=type(obj).__qualname__,
                    key=key,
                    reason="missing in state.npz",
                )
            value = state[key]
            # 0-d ndarray -> python scalar for ints/floats so sklearn internals are happy.
            if isinstance(value, np.ndarray) and value.shape == () and value.dtype.kind in "iuf":
                value = value.item()
            setattr(obj, key, value)

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.array([*list(input_features), "fingerprint"], dtype=object)
