"""Translate fitted tabpfn / sklearn preprocessing objects into ext operators.

The module is intended for offline extraction and ingestion paths. Runtime
vldm loading should use :mod:`sklearn_tabpfn_ext.codec` and must not import
``tabpfn`` or ``torch``. Imports of real tabpfn internals stay lazy inside the
version-specific branches that need them.

Usage::

    from sklearn_tabpfn_ext.tabpfn import translate_member

    pipeline = translate_member(member)   # member is an ensemble member
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import numpy as np
import sklearn
import sklearn.compose
import sklearn.pipeline
import sklearn.preprocessing

import sklearn_tabpfn_ext as _vldm
from sklearn_tabpfn_ext.exceptions import UnsupportedConversionError


@dataclasses.dataclass(frozen=True)
class TranslationProfile:
    """Version dispatch token. ``family`` is the only thing translator code
    branches on; ``version`` is the exact string for diagnostics."""

    version: str
    family: str  # "v63" | "v64"


def profile_for(version: str) -> TranslationProfile:
    # The C1 extract gate (exact allowlist {6.3.2,6.4.0,6.4.1}) is the authority
    # for *which* versions are supported and runs FIRST. profile_for only maps a
    # gated version to its translation family. Parse defensively (first two
    # segments, try/except) so malformed input raises UnsupportedConversionError, not
    # a raw ValueError.
    try:
        parts = version.split(".")  # AttributeError if version is None / non-str
        major, minor = int(parts[0]), int(parts[1])
    except (ValueError, IndexError, AttributeError) as err:
        raise UnsupportedConversionError(
            "tabpfn", "version", f"unsupported tabpfn {version}"
        ) from err
    if (major, minor) == (6, 3):
        return TranslationProfile(version, "v63")
    if (major, minor) == (6, 4):
        return TranslationProfile(version, "v64")
    raise UnsupportedConversionError("tabpfn", "version", f"unsupported tabpfn {version}")


def _qualname(obj: Any) -> str:
    """Return fully-qualified class name: ``module.ClassName``."""
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


# ---------------------------------------------------------------------------
# _SK_TRANSLATIONS  — sklearn FQN -> corresponding vldm wrapper class
# ---------------------------------------------------------------------------
_SK_TRANSLATIONS: dict[str, type] = {}


def _build_sk_translation_table() -> None:
    """Map sklearn FQN -> corresponding vldm wrapper class.

    Inspects vldm.preprocessing.sklearn_wrappers and infers (sklearn parent,
    vldm wrapper) pairs by walking each wrapper's MRO.
    """
    from sklearn_tabpfn_ext import sklearn_wrappers

    for _, member in inspect.getmembers(sklearn_wrappers, inspect.isclass):
        for base in member.__mro__:
            if base.__module__.startswith("sklearn.") and base is not member:
                _SK_TRANSLATIONS[f"{base.__module__}.{base.__qualname__}"] = member
                break


_build_sk_translation_table()


# ---------------------------------------------------------------------------
# _TABPFN_TRANSLATIONS  — tabpfn custom estimator FQN -> vldm class
# ---------------------------------------------------------------------------
# NOTE: SafePowerTransformer and KDITransformerWithNaN are intentionally
# NOT in this table. They require flattening tabpfn's internal _scaler
# (a StandardScaler instance) into individual "_scaler_*" keys, which
# _instantiate_with_state cannot do (it uses getattr on the source, but
# those flat keys don't exist on the tabpfn object). They are handled by
# _TABPFN_POWER_DISPATCH below.
_TABPFN_TRANSLATIONS: dict[str, type] = {
    "tabpfn.preprocessing.steps.adaptive_quantile_transformer.AdaptiveQuantileTransformer": _vldm.AdaptiveQuantileTransformer,
    # v64: the SVD save_standard sub-pipeline uses _NoInverseImputer, a
    # SimpleImputer subclass (verified in-env: tabpfn 6.4.1, mro includes
    # sklearn.impute._base.SimpleImputer; identical fitted attrs statistics_/
    # n_features_in_/_fit_dtype and ctor params). v63 used a plain SimpleImputer
    # there (caught by _SK_TRANSLATIONS). The only difference is a disabled
    # inverse_transform, which vldm's forward never calls. Map it to the vldm
    # SimpleImputer wrapper so _instantiate_with_state copies the same state.
    "tabpfn.preprocessing.steps.utils._NoInverseImputer": _vldm.sklearn_wrappers.SimpleImputer,
    # SquashingScaler is NOT here — same flattening issue as SafePower/KDI.
    # tabpfn keeps ``robust_scaler_`` (RobustScaler) and ``minmax_scaler_``
    # (_MinMaxScaler) as nested objects; vldm flattens them into individual
    # ``robust_center_`` / ``robust_scale_`` / ``minmax_median_`` /
    # ``minmax_scale_`` state keys. Handled by _TABPFN_SQUASH_DISPATCH below.
}


# ---------------------------------------------------------------------------
# _TABPFN_POWER_DISPATCH  — SafePowerTransformer / KDITransformerWithNaN
# ---------------------------------------------------------------------------
# These classes have standardize=True by default. When standardize=True,
# tabpfn's PowerTransformer keeps an internal ``_scaler`` (StandardScaler
# instance). vldm flattens that into individual "_scaler_*" state keys.
# _instantiate_with_state cannot handle this because the flat keys don't
# exist on the tabpfn source object; we need a dedicated copier.

_TABPFN_POWER_DISPATCH: dict[str, type] = {
    "tabpfn.preprocessing.steps.safe_power_transformer.SafePowerTransformer": _vldm.SafePowerTransformer,
    "tabpfn.preprocessing.steps.kdi_transformer.KDITransformerWithNaN": _vldm.KDITransformerWithNaN,
}


def _copy_power_transformer(src, dst, ctx: dict | None = None) -> None:
    """SafePowerTransformer / KDITransformerWithNaN — flatten _scaler when standardize=True."""
    dst.lambdas_ = np.asarray(src.lambdas_)
    dst.n_features_in_ = int(src.n_features_in_)
    if dst.standardize and hasattr(src, "_scaler"):
        s = src._scaler
        dst._scaler_mean_ = np.asarray(s.mean_)
        dst._scaler_scale_ = np.asarray(s.scale_)
        dst._scaler_var_ = np.asarray(s.var_)
        dst._scaler_n_features_in_ = int(s.n_features_in_)
        dst._scaler_n_samples_seen_ = int(s.n_samples_seen_)


# ---------------------------------------------------------------------------
# _TABPFN_SQUASH_DISPATCH  — SquashingScaler
# ---------------------------------------------------------------------------
# tabpfn's SquashingScaler stores robust_scaler_ (RobustScaler instance, may be
# None) and minmax_scaler_ (_MinMaxScaler instance, may be None). vldm flattens
# these into ``robust_center_`` / ``robust_scale_`` / ``minmax_median_`` /
# ``minmax_scale_`` 1-D arrays. _instantiate_with_state cannot bridge the gap.

_TABPFN_SQUASH_DISPATCH: dict[str, type] = {
    "tabpfn.preprocessing.steps.squashing_scaler_transformer.SquashingScaler": _vldm.SquashingScaler,
}


def _copy_ordinal_encoder(src, op_path: str):
    """sklearn OrdinalEncoder -> vldm OrdinalEncoder (byte-exact lookup).

    Only the get_ordinal_encoder() config is supported; anything else
    (different handle_unknown / infrequent categories) hard-fails so it can
    never be silently mis-translated.
    """
    if getattr(src, "handle_unknown", None) != "use_encoded_value":
        raise UnsupportedConversionError(
            "sklearn.preprocessing.OrdinalEncoder",
            op_path,
            f"handle_unknown={src.handle_unknown!r} not supported (expected 'use_encoded_value')",
        )
    if getattr(src, "_infrequent_enabled", False):
        raise UnsupportedConversionError(
            "sklearn.preprocessing.OrdinalEncoder",
            op_path,
            "infrequent categories (min_frequency / max_categories) not supported",
        )
    return _vldm.OrdinalEncoder.from_sklearn(src)


def _copy_squashing_scaler(src, dst, ctx: dict | None = None) -> None:
    """Flatten tabpfn's robust_scaler_ / minmax_scaler_ into vldm's flat state keys."""
    dst.robust_cols_ = np.asarray(src.robust_cols_, dtype=bool)
    dst.minmax_cols_ = np.asarray(src.minmax_cols_, dtype=bool)
    dst.zero_cols_ = np.asarray(src.zero_cols_, dtype=bool)
    dst.n_features_in_ = int(dst.robust_cols_.shape[0])

    rs = getattr(src, "robust_scaler_", None)
    if rs is not None:
        dst.robust_center_ = np.asarray(rs.center_, dtype=np.float64)
        dst.robust_scale_ = np.asarray(rs.scale_, dtype=np.float64)
    else:
        # No robust columns; vldm class expects empty 1-D arrays per its fit logic.
        dst.robust_center_ = np.empty(0, dtype=np.float64)
        dst.robust_scale_ = np.empty(0, dtype=np.float64)

    mms = getattr(src, "minmax_scaler_", None)
    if mms is not None:
        dst.minmax_median_ = np.asarray(mms.median_, dtype=np.float64)
        dst.minmax_scale_ = np.asarray(mms.scale_, dtype=np.float64)
    else:
        dst.minmax_median_ = np.empty(0, dtype=np.float64)
        dst.minmax_scale_ = np.empty(0, dtype=np.float64)


# ---------------------------------------------------------------------------
# _TABPFN_FUNC_NAMES  — tabpfn callable FQN -> vldm FunctionTransformer.func enum
# ---------------------------------------------------------------------------
# Key format: ``module + "." + qualname`` of the callable object.
# Value: one of the string names in vldm.preprocessing.composite.function_transformer._NAMED_FUNCS.
_TABPFN_FUNC_NAMES: dict[str, str] = {
    # v63 (tabpfn 6.3.x) FQNs
    "tabpfn.preprocessing.steps.reshape_feature_distribution_step._identity": "identity",
    "tabpfn.preprocessing.steps.reshape_feature_distribution_step._inf_to_nan_func": "inf_to_nan",
    "tabpfn.preprocessing.steps.reshape_feature_distribution_step._exp_minus_1": "exp_minus_1",
    # v64 (tabpfn 6.4.x) aliases — the identity / inf->nan helpers moved to
    # steps.utils (verified in-env against tabpfn 6.4.1). _exp_minus_1 stayed in
    # reshape_feature_distribution_step. Additive: both map to the same vldm names.
    "tabpfn.preprocessing.steps.utils._identity": "identity",
    "tabpfn.preprocessing.steps.utils._replace_inf_with_nan": "inf_to_nan",
}


# ---------------------------------------------------------------------------
# tabpfn step class FQNs -> vldm class + per-step fitted-state mapping
# ---------------------------------------------------------------------------
# Each entry is a tuple: (vldm_class, state_copier_fn)
# The state_copier_fn(src, dst, upstream_context) copies fitted attrs from the
# tabpfn step object ``src`` onto the fresh vldm object ``dst``.
# ``upstream_context`` carries computed values from previous steps (e.g. n_features_out_
# from the constant filter to set n_features_in_ on the reshaper).


def _copy_constant_filter(src, dst, ctx: dict, profile=None) -> None:
    """RemoveConstantFeaturesStep -> ConstantFeatureFilter."""
    dst.sel_ = np.asarray(src.sel_, dtype=bool)
    dst.n_features_in_ = int(dst.sel_.shape[0])
    dst.n_features_out_ = int(dst.sel_.sum())
    ctx["n_features_out"] = dst.n_features_out_


def _copy_quantile_svd(src, dst, ctx: dict, profile=None) -> None:
    """ReshapeFeatureDistributionsStep -> QuantileSVDReshaper.

    Recursively translates src.transformer_ (a raw sklearn Pipeline containing
    tabpfn classes) into a vldm SequentialPipeline so that:
    1. The in-memory pipeline is tabpfn-free.
    2. codec.save can recurse through the same vldm machinery (no tabpfn
       class references leak into pipeline.json).
    """
    # op_path is not available in this callback; use a stable placeholder.
    # The codec uses the attr path, not this string, for state.npz keys.
    # v63: src.transformer_ is Pipeline([preprocess CT, global_transformer FU([passthrough, svd])]).
    # v64: src.transformer_ is the bare "preprocess" ColumnTransformer (SVD is a
    #      SEPARATE later AddSVDFeaturesStep — emitted as its own QuantileSVDReshaper
    #      step at its original position; see _add_svd_factory). Either way we just
    #      translate src.transformer_ faithfully here.
    op_path_for_child = "cpu_preprocessor.reshape.transformer_"
    dst.transformer_ = translate_sklearn_obj(src.transformer_, op_path_for_child, profile)
    dst.subsampled_features_ = np.asarray(src.subsampled_features_, dtype=np.int64)
    dst.n_features_in_ = ctx.get("n_features_out")
    # Derive n_features_out_ by probing the translated inner transformer.
    n_in = dst.n_features_in_ or 0
    probe = np.zeros((1, n_in), dtype=np.float64)
    if dst.subsampled_features_ is not None:
        probe = probe[:, dst.subsampled_features_]
    dst.n_features_out_ = int(dst.transformer_.transform(probe).shape[1])
    ctx["n_features_out"] = dst.n_features_out_


def _copy_categorical_encoder(src, dst, ctx: dict, profile=None) -> None:
    """EncodeCategoricalFeaturesStep -> IdentityCategoricalEncoder (identity for tab2.5).

    For models without categorical features the source step is a no-op, so
    IdentityCategoricalEncoder is correct, and this helper only copies state for
    that identity branch. The non-identity (ordinal) case is already handled by
    ``_categorical_factory`` (ordinal* -> CategoricalOrdinalEncoder: it translates
    src.column_transformer_, assigns it to dst.column_transformer_, and the codec
    recurses via _child_attrs — exactly the steps once sketched here as future
    work). The one remaining unsupported case is the **onehot** encoder, which
    ``_categorical_factory`` rejects fail-loud with UnsupportedConversionError (it is a
    different encoder, out of scope until a deployed model needs it).
    """
    n = ctx.get("n_features_out", 0)
    dst.n_features_in_ = n
    dst.n_features_out_ = n


def _categorical_factory(src, op_path: str, ctx: dict, profile=None):
    """EncodeCategoricalFeaturesStep -> Identity / CategoricalOrdinalEncoder.

    numeric|none (categorical_transformer_ is None) -> IdentityCategoricalEncoder.
    ordinal*      -> CategoricalOrdinalEncoder (translated ColumnTransformer + random_mappings_).
    onehot        -> UnsupportedConversionError (different encoder, out of scope).
    """
    name = src.categorical_transform_name
    ct = src.categorical_transformer_
    if ct is None:
        new = _vldm.IdentityCategoricalEncoder()
        _copy_categorical_encoder(src, new, ctx, profile)
        return new
    if not name.startswith("ordinal"):
        raise UnsupportedConversionError(
            f"categorical_transform_name={name!r}",
            op_path,
            "only ordinal* / numeric / none are supported; onehot needs a separate encoder.",
        )
    new = _vldm.CategoricalOrdinalEncoder()
    new.column_transformer_ = translate_sklearn_obj(ct, f"{op_path}.column_transformer", profile)
    # ctx carries n_features_out from the prior step; when absent (isolated/fallback
    # dispatch with empty ctx) fall back to the translated transformer's input width.
    new.n_features_in_ = ctx.get("n_features_out") or getattr(
        new.column_transformer_, "n_features_in_", 0
    )
    new.random_mappings_ = {
        int(k): np.asarray(v, dtype=np.int64) for k, v in (src.random_mappings_ or {}).items()
    }
    # Ordinal encoding is column-count-preserving (encode-in-place + passthrough).
    ctx["n_features_out"] = new.n_features_in_
    return new


def _copy_fingerprint(src, dst, ctx: dict, profile=None) -> None:
    """AddFingerprintFeaturesStep -> FingerprintFeatureAdder.

    v63 (tabpfn 6.3.x) stores a random ``rnd_salt_``; v64 (tabpfn 6.4.x)
    replaced it with a deterministic ``n_cells_`` (= n_rows * n_cols at fit).
    Verified in-env (tabpfn 6.4.1): the fitted fingerprint step exposes
    ``n_cells_`` and no ``rnd_salt_``. The vldm FingerprintFeatureAdder selects
    its serialized state key by ``algo`` (set via the init-kwargs lambda).
    """
    if profile is not None and profile.family == "v64":
        dst.n_cells_ = int(src.n_cells_)
    else:
        dst.rnd_salt_ = int(src.rnd_salt_)
    dst.n_features_in_ = ctx.get("n_features_out", 0)


def _copy_shuffler(src, dst, ctx: dict, profile=None) -> None:
    """ShuffleFeaturesStep -> FeatureShuffler."""
    dst.index_permutation_ = np.asarray(src.index_permutation_, dtype=np.int64)
    dst.n_features_in_ = int(dst.index_permutation_.shape[0])


def _copy_polynomial_features(src, dst, ctx: dict, profile=None) -> None:
    """NanHandlingPolynomialFeaturesStep -> PolynomialFeaturesAdder.

    tabpfn fitted attrs (no trailing underscore on indices):
      src.poly_factor_1_idx  — ndarray, shape (n_polynomials,)
      src.poly_factor_2_idx  — ndarray, shape (n_polynomials,)
      src.standardizer.scale_  — StandardScaler(with_mean=False).scale_
    n_features_in_ is derived from the length of the factor index arrays and
    the scaler scale array (both should be consistent; we prefer scale_).
    """
    dst.poly_factor_1_idx_ = np.asarray(src.poly_factor_1_idx, dtype=np.int64)
    dst.poly_factor_2_idx_ = np.asarray(src.poly_factor_2_idx, dtype=np.int64)
    dst.std_scale_ = np.asarray(src.standardizer.scale_, dtype=np.float64)
    dst.n_features_in_ = int(dst.std_scale_.shape[0])


def _copy_differentiable_znorm(src, dst, ctx: dict, profile=None) -> None:
    """DifferentiableZNormStep -> DifferentiableZNorm.

    tabpfn fitted attrs (torch tensors, shape [1, n_features] with keepdim=True):
      src.means  — row-vector tensor; squeeze to 1-D ndarray
      src.stds   — row-vector tensor; squeeze to 1-D ndarray
    n_features_in_ is derived from the length after squeezing.
    """
    # src.means / src.stds are torch tensors of shape [1, n_features].
    # Convert to numpy, squeeze the leading dim.
    means_arr = np.asarray(src.means).reshape(-1)
    stds_arr = np.asarray(src.stds).reshape(-1)
    dst.mean_ = means_arr.astype(np.float64)
    dst.std_ = stds_arr.astype(np.float64)
    dst.n_features_in_ = int(dst.mean_.shape[0])


# tabpfn step class FQN -> (vldm_class, init_kwargs_fn, state_copier_fn)
# init_kwargs_fn(src, profile) returns the kwargs to pass to vldm_class.__init__
# state_copier_fn(src, dst, ctx, profile) copies fitted attrs from tabpfn src onto vldm dst
_TABPFN_STEP_DISPATCH: dict[str, tuple] = {
    "tabpfn.preprocessing.steps.remove_constant_features_step.RemoveConstantFeaturesStep": (
        _vldm.ConstantFeatureFilter,
        lambda src, profile: {},
        _copy_constant_filter,
    ),
    "tabpfn.preprocessing.steps.reshape_feature_distribution_step.ReshapeFeatureDistributionsStep": (
        _vldm.QuantileSVDReshaper,
        lambda src, profile: {
            # Do NOT pass transformer=src.transformer_ here — that would store a raw
            # tabpfn/sklearn object in self.transformer. The translated transformer_ is
            # set by _copy_quantile_svd via translate_sklearn_obj (tabpfn-free).
            "subsampled_features": np.asarray(src.subsampled_features_, dtype=np.int64),
            "append_to_original": bool(src.append_to_original),
        },
        _copy_quantile_svd,
    ),
    "tabpfn.preprocessing.steps.encode_categorical_features_step.EncodeCategoricalFeaturesStep": _categorical_factory,
    "tabpfn.preprocessing.steps.add_fingerprint_features_step.AddFingerprintFeaturesStep": (
        _vldm.FingerprintFeatureAdder,
        lambda src, profile: (
            {"algo": "v64"}
            if profile is not None and profile.family == "v64"
            # v63: pass the random salt so the adder reproduces it; algo defaults v63.
            else {"salt": int(src.rnd_salt_)}
        ),
        _copy_fingerprint,
    ),
    "tabpfn.preprocessing.steps.shuffle_features_step.ShuffleFeaturesStep": (
        _vldm.FeatureShuffler,
        lambda src, profile: {
            "shuffle_method": str(src.shuffle_method),
            "shuffle_index": int(np.asarray(src.shuffle_index).item())
            if src.shuffle_index is not None
            else 0,
        },
        _copy_shuffler,
    ),
    "tabpfn.preprocessing.steps.nan_handling_polynomial_features_step.NanHandlingPolynomialFeaturesStep": (
        _vldm.PolynomialFeaturesAdder,
        lambda src, profile: {
            "max_features": int(src.max_poly_features)
            if src.max_poly_features is not None
            else None,
            "random_state": None,
        },
        _copy_polynomial_features,
    ),
    "tabpfn.preprocessing.steps.differentiable_z_norm_step.DifferentiableZNormStep": (
        _vldm.DifferentiableZNorm,
        lambda src, profile: {},
        _copy_differentiable_znorm,
    ),
}


# ---------------------------------------------------------------------------
# Core translation helpers
# ---------------------------------------------------------------------------


def _translate_function_transformer(obj: Any, op_path: str) -> _vldm.FunctionTransformer:
    """Map a sklearn FunctionTransformer to vldm.FunctionTransformer with a named func.

    Cases:
      - obj.func is None -> "identity"
      - obj.func is a callable whose FQN is in _TABPFN_FUNC_NAMES -> mapped name
      - anything else -> UnsupportedConversionError
    """
    if obj.func is None:
        return _vldm.FunctionTransformer(func="identity")
    if callable(obj.func):
        fn = obj.func
        qual = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')}"
        if qual in _TABPFN_FUNC_NAMES:
            return _vldm.FunctionTransformer(func=_TABPFN_FUNC_NAMES[qual])
        raise UnsupportedConversionError(
            f"FunctionTransformer(func={qual})",
            op_path,
            "callable not in named-function registry; "
            "add it to scripts/tabpfn_translator._TABPFN_FUNC_NAMES "
            "AND vldm.preprocessing.composite.function_transformer._NAMED_FUNCS",
        )
    raise UnsupportedConversionError(
        f"FunctionTransformer(func={obj.func!r})",
        op_path,
        "FunctionTransformer.func is neither None nor callable",
    )


def _instantiate_with_state(target_cls: type, src_obj: Any, op_path: str) -> Any:
    """Construct target_cls with init params copied from src_obj, then copy fitted state."""
    init = {k: getattr(src_obj, k) for k in target_cls._init_param_keys if hasattr(src_obj, k)}
    new = target_cls(**init)
    for k in target_cls._state_keys:
        if not hasattr(src_obj, k):
            raise UnsupportedConversionError(
                _qualname(src_obj),
                op_path,
                f"missing fitted attribute {k!r}; was the object fit?",
            )
        setattr(new, k, getattr(src_obj, k))
    return new


def _translate_tabpfn_step(obj: Any, op_path: str, ctx: dict, profile=None) -> Any:
    """Translate a single tabpfn step object using a shared context dict.

    The context dict carries ``n_features_out`` from one step to the next so
    that steps like ``QuantileSVDReshaper`` can set ``n_features_in_``
    correctly without needing to re-probe upstream steps.

    Falls back to ``translate_sklearn_obj`` for non-dispatch-table types
    (e.g. sklearn composites inside steps).
    """
    fqn = _qualname(obj)
    if fqn in _TABPFN_STEP_DISPATCH:
        entry = _TABPFN_STEP_DISPATCH[fqn]
        if callable(entry):  # factory(src, op_path, ctx, profile) -> vldm op
            return entry(obj, op_path, ctx, profile)
        vldm_cls, init_kwargs_fn, state_copier_fn = entry
        init_kwargs = init_kwargs_fn(obj, profile)
        new = vldm_cls(**init_kwargs)
        state_copier_fn(obj, new, ctx, profile)
        return new
    # Delegate to the general translator (e.g. sklearn Pipeline, ColumnTransformer …)
    return translate_sklearn_obj(obj, op_path, profile)


def _iter_pipeline_steps(obj: Any, profile=None) -> list:
    """Return the ordered tabpfn step objects for either pipeline shape.

    - v63: ``tabpfn.preprocessing.pipeline_interfaces.SequentialFeatureTransformer``
      is a ``UserList`` whose ``.steps`` are the step objects directly.
    - v64: ``tabpfn.preprocessing.pipeline_interface.PreprocessingPipeline``
      stores ``_raw_steps`` (a plain list of step objects) and ``steps`` (a list
      of ``(step_obj, modality_set)`` tuples). Verified in-env (tabpfn 6.4.1):
      ``_raw_steps`` is the clean ordered list; ``steps`` entries are 2-tuples.

    Returns a flat list of step objects in execution order.
    """
    if profile is not None and profile.family == "v64":
        # PreprocessingPipeline: prefer _raw_steps (plain step objects). Fall back
        # to unwrapping the (step, modality) tuples in .steps if absent.
        raw = getattr(obj, "_raw_steps", None)
        if raw is not None:
            return list(raw)
        return [s[0] if isinstance(s, tuple) else s for s in obj.steps]
    # v63 UserList: .steps holds step objects directly.
    return list(obj.steps)


# tabpfn top-level preprocessing pipeline FQNs, by family. Used to recognise the
# per-estimator cpu_preprocessor container (replaces the old exact-string check).
_PIPELINE_FQNS: dict[str, str] = {
    "v63": "tabpfn.preprocessing.pipeline_interfaces.SequentialFeatureTransformer",
    "v64": "tabpfn.preprocessing.pipeline_interface.PreprocessingPipeline",
}


def _is_top_pipeline(fqn: str, profile=None) -> bool:
    """True if ``fqn`` is the per-estimator preprocessing-pipeline container for
    the active profile (v63 SequentialFeatureTransformer / v64 PreprocessingPipeline).
    """
    if profile is not None:
        return fqn == _PIPELINE_FQNS.get(profile.family)
    # No profile (legacy/isolated dispatch): accept either known top container.
    return fqn in _PIPELINE_FQNS.values()


_ADD_SVD_FQN = "tabpfn.preprocessing.steps.add_svd_features_step.AddSVDFeaturesStep"


def _add_svd_factory(src, op_path: str, ctx: dict, profile=None):
    """v64 AddSVDFeaturesStep -> standalone QuantileSVDReshaper (SVD in place).

    In v64 SVD is a separate step placed AFTER the categorical step (verified
    in-env: constant -> reshape -> categorical -> add_svd -> fingerprint ->
    shuffle). Its ``_transform`` returns ``[X, transformer_.transform(X)]`` —
    i.e. it APPENDS the SVD columns to its full input (verified in source).

    The vldm ``QuantileSVDReshaper`` reproduces exactly this when its
    ``transformer_`` is a FeatureUnion([passthrough-identity, svd_sub]) over all
    columns (``subsampled_features=None``). This keeps SVD at its original
    sequence position so the preceding categorical step's column-count contract
    is preserved (folding SVD into reshape would feed the categorical step the
    wrong width — that path was rejected after in-env validation).

    ``is_no_op`` (n_features < 2): the step is a pure passthrough; emit a
    QuantileSVDReshaper whose transformer_ is identity (no SVD branch).

    Verified v6.4.0/6.4.1 contract (read in-env, fail loud on mismatch):
      src.transformer_ = sklearn Pipeline(steps=[("save_standard", Pipeline),
                                                 ("svd", TruncatedSVD)]).
    """
    n_in = ctx.get("n_features_out", 0)
    # subsampled_features_ is an identity index range over all n_in columns (the
    # SVD step operates on the full input). An explicit arange — NOT None — so the
    # codec serializes an int64 array (None becomes an object-array which the secure
    # loader rejects). arange(n_in) selects every column in order = byte-identity.
    sub = np.arange(int(n_in), dtype=np.int64)
    new = _vldm.QuantileSVDReshaper(subsampled_features=sub, append_to_original=True)

    if getattr(src, "is_no_op", False):
        # No SVD: identity passthrough. Represent transformer_ as a 1-branch
        # FeatureUnion of identity so transform returns X unchanged.
        new.transformer_ = _vldm.FeatureUnion(
            transformer_list=[("passthrough", _vldm.FunctionTransformer(func="identity"))]
        )
        new.subsampled_features_ = sub
        new.n_features_in_ = n_in
        new.n_features_out_ = n_in
        ctx["n_features_out"] = n_in
        return new

    add_tr = getattr(src, "transformer_", None)
    if add_tr is None or not hasattr(add_tr, "steps"):
        raise UnsupportedConversionError(
            "AddSVDFeaturesStep",
            op_path,
            f"expected fitted transformer_ Pipeline with save_standard/svd steps; got {add_tr!r}",
        )
    step_names = [n for n, _ in add_tr.steps]
    if "save_standard" not in step_names or "svd" not in step_names:
        raise UnsupportedConversionError(
            "AddSVDFeaturesStep",
            op_path,
            f"expected save_standard + svd steps; got {step_names!r}",
        )

    svd_sub = translate_sklearn_obj(add_tr, f"{op_path}.svd", profile)
    new.transformer_ = _vldm.FeatureUnion(
        transformer_list=[
            ("passthrough", _vldm.FunctionTransformer(func="identity")),
            ("svd", svd_sub),
        ]
    )
    new.subsampled_features_ = sub
    new.n_features_in_ = n_in
    # Probe output width through the translated FeatureUnion (faithful to the fit).
    probe = np.zeros((1, n_in), dtype=np.float64)
    new.n_features_out_ = int(new.transformer_.transform(probe).shape[1])
    ctx["n_features_out"] = new.n_features_out_
    return new


# Register the v64 SVD step factory now that _add_svd_factory is defined. v63
# never emits AddSVDFeaturesStep (SVD is folded inside reshape), so this entry is
# inert for v63 pipelines and only fires for v64.
_TABPFN_STEP_DISPATCH[_ADD_SVD_FQN] = _add_svd_factory


def translate_sklearn_obj(obj: Any, op_path: str, profile=None) -> Any:
    """Recursively translate an sklearn or tabpfn object into a vldm operator.

    Dispatch order:
    1. tabpfn SequentialFeatureTransformer (UserList of unnamed steps)
    2. sklearn composites (Pipeline, ColumnTransformer, FeatureUnion, FunctionTransformer)
    3. tabpfn step classes (via _TABPFN_STEP_DISPATCH — no tabpfn import needed)
    4. tabpfn custom estimators (via _TABPFN_TRANSLATIONS)
    5. sklearn leaf classes (via _SK_TRANSLATIONS)
    6. Raise UnsupportedConversionError for everything else
    """
    # --- tabpfn SequentialFeatureTransformer (UserList of unnamed steps) ---
    # This is the real cpu_preprocessor type in tabpfn >= 0.1.x pickles.
    # It stores steps as a plain indexed list (no name strings), so we
    # auto-generate step names "step_0", "step_1", ... to match the
    # vldm SequentialPipeline convention.
    # A shared ctx dict flows across steps so each copier can read
    # n_features_out written by the previous step.
    fqn_top = _qualname(obj)
    if _is_top_pipeline(fqn_top, profile):
        # Materialise the ordered step objects for either pipeline shape (C3).
        # v64 keeps the SVD as a separate AddSVDFeaturesStep at its original
        # position (after categorical); it is translated in place by _add_svd_factory
        # into a standalone QuantileSVDReshaper, so the column-count contract for the
        # intervening categorical step is preserved (no fold — see _add_svd_factory).
        steps = _iter_pipeline_steps(obj, profile)
        shared_ctx: dict = {}
        translated_steps = []
        for idx, step in enumerate(steps):
            step_name = f"step_{idx}"
            translated_steps.append(
                (
                    step_name,
                    _translate_tabpfn_step(step, f"{op_path}.{step_name}", shared_ctx, profile),
                )
            )
        return _vldm.SequentialPipeline(steps=translated_steps)

    # --- sklearn composites ---
    if isinstance(obj, sklearn.pipeline.Pipeline):
        return _vldm.SequentialPipeline(
            steps=[
                (name, translate_sklearn_obj(step, f"{op_path}.{name}", profile))
                for name, step in obj.steps
            ]
        )

    if isinstance(obj, sklearn.compose.ColumnTransformer):
        # Normalise obj.remainder: tabpfn's get_ordinal_encoder() sets remainder to a
        # FunctionTransformer() (identity passthrough) rather than the string
        # "passthrough". The codec only recognises string sentinels, so normalise the
        # FunctionTransformer() to "passthrough" before building the vldm wrapper.
        def _normalise_remainder(rem):
            if rem in ("passthrough", "drop"):
                return rem
            if isinstance(rem, sklearn.preprocessing.FunctionTransformer) and rem.func is None:
                return "passthrough"
            # Any other estimator-as-remainder would leak a raw sklearn object into the
            # translated pipeline, violating the no-tabpfn-leak invariant.
            raise UnsupportedConversionError(
                "ColumnTransformer.remainder",
                op_path,
                "non-identity estimator remainder not supported",
            )

        rebuilt = []
        for name, trans, cols in obj.transformers_:
            if trans in ("passthrough", "drop"):
                rebuilt.append((name, trans, list(cols)))
            else:
                rebuilt.append(
                    (
                        name,
                        translate_sklearn_obj(trans, f"{op_path}.{name}", profile),
                        list(cols),
                    )
                )
        ct = _vldm.ColumnTransformer(
            transformers=rebuilt, remainder=_normalise_remainder(obj.remainder)
        )
        # Set n_features_in_ from source so vldm.ColumnTransformer._from_state_dict
        # can derive the remaining sklearn-internal state (transformers_, _columns,
        # _remainder, sparse_output_) from `transformers` + `remainder`. We must NOT
        # copy transformers_ from the source — that contains the original
        # sklearn/tabpfn classes and would re-leak them into the translated pipeline,
        # defeating the whole point of translation.
        if hasattr(obj, "n_features_in_"):
            ct.n_features_in_ = int(obj.n_features_in_)
        # Replay the manual sklearn-internal restoration logic so transform() works
        # without going through .fit().
        _vldm.ColumnTransformer._from_state_dict(ct, {"n_features_in_": ct.n_features_in_})
        return ct

    if isinstance(obj, sklearn.pipeline.FeatureUnion):
        rebuilt = [
            (name, translate_sklearn_obj(trans, f"{op_path}.{name}", profile))
            for name, trans in obj.transformer_list
        ]
        fu = _vldm.FeatureUnion(transformer_list=rebuilt)
        # FeatureUnion is simpler: transform delegates to each child's transform
        # then concatenates. Children are already vldm-wrapped (rebuilt above).
        # NOTE: n_features_in_ is a read-only property on sklearn's FeatureUnion
        # (derived from child transformers at call time); do NOT attempt to assign it.
        return fu

    if isinstance(obj, sklearn.preprocessing.FunctionTransformer):
        return _translate_function_transformer(obj, op_path)

    if isinstance(obj, sklearn.preprocessing.OrdinalEncoder):
        return _copy_ordinal_encoder(obj, op_path)

    # --- dispatch by FQN ---
    # fqn_top was already computed above for the SequentialFeatureTransformer check.
    fqn = fqn_top

    # tabpfn step classes — port of extract_model._build_estimator_pipeline_from_member logic
    if fqn in _TABPFN_STEP_DISPATCH:
        entry = _TABPFN_STEP_DISPATCH[fqn]
        if callable(entry):  # factory(src, op_path, ctx, profile) -> vldm op
            return entry(obj, op_path, {}, profile)
        vldm_cls, init_kwargs_fn, state_copier_fn = entry
        init_kwargs = init_kwargs_fn(obj, profile)
        new = vldm_cls(**init_kwargs)
        ctx: dict = {}
        state_copier_fn(obj, new, ctx, profile)
        return new

    # tabpfn power-transformer estimators — require _scaler flattening
    if fqn in _TABPFN_POWER_DISPATCH:
        target_cls = _TABPFN_POWER_DISPATCH[fqn]
        init = {k: getattr(obj, k) for k in target_cls._init_param_keys if hasattr(obj, k)}
        new = target_cls(**init)
        _copy_power_transformer(obj, new)
        return new

    # tabpfn SquashingScaler — requires robust_scaler_ / minmax_scaler_ flattening
    if fqn in _TABPFN_SQUASH_DISPATCH:
        target_cls = _TABPFN_SQUASH_DISPATCH[fqn]
        init = {k: getattr(obj, k) for k in target_cls._init_param_keys if hasattr(obj, k)}
        new = target_cls(**init)
        _copy_squashing_scaler(obj, new)
        return new

    # tabpfn custom estimators -> vldm equivalents via _state_keys/_init_param_keys
    if fqn in _TABPFN_TRANSLATIONS:
        return _instantiate_with_state(_TABPFN_TRANSLATIONS[fqn], obj, op_path)

    # sklearn leaf classes -> vldm wrappers
    if fqn in _SK_TRANSLATIONS:
        return _instantiate_with_state(_SK_TRANSLATIONS[fqn], obj, op_path)

    raise UnsupportedConversionError(
        fqn,
        op_path,
        (
            "add a wrapper in vldm/preprocessing/sklearn_wrappers/"
            if fqn.startswith("sklearn.")
            else "open a vldm issue with this op_path"
        ),
    )


def _categorical_indices(clf, profile=None) -> list[int]:
    """Categorical column indices from a fitted TabPFNClassifier.

    - v63 (tabpfn 6.3.x): ``inferred_categorical_indices_`` (possibly an empty
      list — legitimate "no categoricals").
    - v64 (tabpfn 6.4.x): the attribute was removed; categorical indices come
      from ``inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL)``
      (verified in-env, tabpfn 6.4.1). Imported lazily — extract-only, never in
      ``vldm/``.

    If the expected attribute is ABSENT we must NOT default to ``[]`` (that
    silently drops the input sanitizer); we fail loud.
    """
    if profile is not None and profile.family == "v64":
        schema = getattr(clf, "inferred_feature_schema_", None)
        if schema is None:
            raise UnsupportedConversionError(
                "inferred_feature_schema_",
                "input_sanitizer",
                "v64 classifier missing inferred_feature_schema_; unsupported tabpfn version?",
            )
        from tabpfn.preprocessing.datamodel import FeatureModality

        return [int(i) for i in schema.indices_for(FeatureModality.CATEGORICAL)]
    if hasattr(clf, "inferred_categorical_indices_"):
        # Explicit None-check (not `or []`): a truthiness test would raise
        # "ambiguous truth value" if the attr were ever a numpy array.
        val = clf.inferred_categorical_indices_
        return list(val) if val is not None else []
    raise UnsupportedConversionError(
        "inferred_categorical_indices_",
        "input_sanitizer",
        "cannot determine categorical indices from fitted classifier; unsupported tabpfn version?",
    )


def translate_input_sanitizer(clf, profile=None) -> _vldm.InputSanitizer:
    """Build an InputSanitizer from a fitted TabPFNClassifier.

    Identity when no categoricals were inferred; else translate the fitted
    ordinal_encoder_ ColumnTransformer to a vldm ColumnTransformer.
    """
    from sklearn_tabpfn_ext.input_sanitizer import InputSanitizer

    n_features_in = int(clf.n_features_in_)
    cat_ix = _categorical_indices(clf, profile)
    if not cat_ix:
        return InputSanitizer.identity(n_features_in)

    ord_enc = getattr(clf, "ordinal_encoder_", None)
    if ord_enc is None:
        raise UnsupportedConversionError(
            "ordinal_encoder_",
            "input_sanitizer",
            "classifier has inferred categoricals but no fitted ordinal_encoder_",
        )
    import sklearn.compose

    is_v64 = profile is not None and profile.family == "v64"
    if is_v64:
        # v64: get_ordinal_encoder() returns an OrderPreservingColumnTransformer
        # (a ColumnTransformer subclass) that restores original column positions
        # (audit §3.2). It translates like a vanilla ColumnTransformer; we tag the
        # produced vldm ColumnTransformer with order_preserving=True so the runtime
        # scatters transformed blocks back to their source positions. Verified
        # in-env FQN: tabpfn.preprocessing.steps.preprocessing_helpers.OrderPreservingColumnTransformer.
        if not isinstance(ord_enc, sklearn.compose.ColumnTransformer):
            raise UnsupportedConversionError(
                type(ord_enc).__name__,
                "input_sanitizer",
                f"{type(ord_enc).__name__} not a ColumnTransformer subclass; unsupported v64 encoder",
            )
    elif type(ord_enc) is not sklearn.compose.ColumnTransformer:
        # v63: only the vanilla sklearn ColumnTransformer is supported.
        raise UnsupportedConversionError(
            type(ord_enc).__name__,
            "input_sanitizer",
            f"{type(ord_enc).__name__} not supported; only the vanilla sklearn "
            "ColumnTransformer (tabpfn 6.3.x) is supported",
        )
    ct = translate_sklearn_obj(ord_enc, op_path="input_sanitizer", profile=profile)
    if is_v64:
        ct.order_preserving = True
        # Replay sklearn-internal restoration with the order_preserving flag set so
        # the runtime transform() scatters blocks back to original positions.
        _vldm.ColumnTransformer._from_state_dict(ct, {"n_features_in_": ct.n_features_in_})
    return InputSanitizer(
        n_features_in=n_features_in,
        inferred_categorical_indices=cat_ix,
        column_transformer=ct,
    )


def translate_member(member, profile=None) -> _vldm.SequentialPipeline:
    """Translate a tabpfn TabPFNPreprocessedEnsembleMember.cpu_preprocessor.

    The ``cpu_preprocessor`` is a ``tabpfn.preprocessing.pipeline_interfaces.
    SequentialFeatureTransformer`` (a UserList subclass with unnamed steps)
    containing 5 tabpfn step objects.  We walk each step, producing a vldm
    ``SequentialPipeline`` (step names auto-generated as "step_0" … "step_4")
    with zero tabpfn references.

    This is the top-level entry point called from source-conversion tooling.
    """
    sk_pipeline = member.cpu_preprocessor
    return translate_sklearn_obj(sk_pipeline, op_path="cpu_preprocessor", profile=profile)
