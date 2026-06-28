"""Save / load vldm preprocessing artifacts (JSON + npz; no pickle)."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from sklearn_tabpfn_ext import __version__ as _vldm_version
from sklearn_tabpfn_ext import registry
from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.composite.column_transformer import ColumnTransformer
from sklearn_tabpfn_ext.composite.feature_union import FeatureUnion
from sklearn_tabpfn_ext.composite.sequential import SequentialPipeline
from sklearn_tabpfn_ext.exceptions import (
    ArtifactNotFoundError,
    ArtifactSchemaError,
    ArtifactSecurityError,
    OpInitError,
    OpStateError,
    UnknownOperatorError,
    VersionMismatchWarning,
)
from sklearn_tabpfn_ext.input_sanitizer import InputSanitizer
from sklearn_tabpfn_ext.schema import MetaJson, OpSpec, PipelineSpec, SourceMeta


def _serialise_op(op, name: str, op_path: str, state_acc: dict[str, np.ndarray]) -> OpSpec:
    """Recursive: turn a vldm operator into an OpSpec, accumulating arrays into state_acc.

    op_path is the dotted path from root using `name` segments; used as state.npz key prefix.

    For ops that declare ``_child_attrs`` (fitted attributes that are themselves
    vldm operators), each such child is serialised recursively and appended to
    ``OpSpec.children`` AFTER any composite-specific children.  On load,
    ``_build_op`` iterates ``_child_attrs`` and pops the corresponding children
    from the tail of ``spec.children`` (in order).
    """
    cls = type(op)
    # Write the registered canonical op_id (frozen wire-format string), NOT the
    # Python module path — so moving modules never changes written op_ids and
    # existing vldm.preprocessing.* artifacts stay loadable.
    op_id = getattr(cls, "_canonical_op_id", f"{cls.__module__}.{cls.__qualname__}")

    init_params = op._init_params_dict() if hasattr(op, "_init_params_dict") else {}
    state = op._to_state_dict() if hasattr(op, "_to_state_dict") else {}
    state_keys = list(state.keys())

    # Persist arrays under op_path/key.
    for key, arr in state.items():
        npz_key = f"{op_path}/{key}" if op_path else key
        state_acc[npz_key] = np.asarray(arr)

    # Composite descent.
    children: list[OpSpec] = []
    if isinstance(op, SequentialPipeline):
        # init_params encodes the steps mapping by name and child_ref.
        steps_meta = []
        for i, (step_name, step) in enumerate(op.steps):
            child_path = f"{op_path}.{step_name}" if op_path else step_name
            children.append(_serialise_op(step, step_name, child_path, state_acc))
            steps_meta.append([step_name, {"child_ref": i}])
        init_params = {"steps": steps_meta}
    elif isinstance(op, FeatureUnion):
        tlist_meta = []
        for i, (step_name, step) in enumerate(op.transformer_list):
            child_path = f"{op_path}.{step_name}" if op_path else step_name
            children.append(_serialise_op(step, step_name, child_path, state_acc))
            tlist_meta.append([step_name, {"child_ref": i}])
        init_params = {"transformer_list": tlist_meta}
    elif isinstance(op, ColumnTransformer):
        trans_meta = []
        next_idx = 0
        for trans_name, trans, cols in op.transformers_:
            cols_list = list(cols) if not isinstance(cols, list) else cols
            if trans == "passthrough":
                trans_meta.append(
                    {"name": trans_name, "columns": cols_list, "child_ref": "passthrough"}
                )
            elif trans == "drop":
                trans_meta.append({"name": trans_name, "columns": cols_list, "child_ref": "drop"})
            else:
                child_path = f"{op_path}.{trans_name}" if op_path else trans_name
                children.append(_serialise_op(trans, trans_name, child_path, state_acc))
                trans_meta.append({"name": trans_name, "columns": cols_list, "child_ref": next_idx})
                next_idx += 1
        # order_preserving is a vldm-specific ColumnTransformer kwarg (tabpfn 6.4.x
        # OrderPreservingColumnTransformer). It MUST be serialised here — this CT
        # special-case builds init_params explicitly and does NOT consult
        # _init_param_keys, so otherwise it is silently dropped and a 6.4.x
        # order-preserving encoder reloads as order_preserving=False (wrong column
        # order at serve time).
        init_params = {
            "transformers": trans_meta,
            "remainder": op.remainder,
            "order_preserving": getattr(op, "order_preserving", False),
        }

    # _child_attrs: fitted attributes that are themselves vldm operators.
    # Serialise each as a child OpSpec appended after any composite-specific children.
    # _build_op restores them in the same order after _from_state_dict.
    child_attrs: tuple[str, ...] = getattr(cls, "_child_attrs", ())
    written_child_attrs: list[str] = []
    for attr_name in child_attrs:
        child_op = getattr(op, attr_name, None)
        if child_op is None:
            continue
        if not isinstance(child_op, VldmEstimatorMixin):
            raise TypeError(
                f"{op_id}.{attr_name} is listed in _child_attrs but is not a "
                f"VldmEstimatorMixin instance (got {type(child_op).__qualname__}). "
                "Ensure the translator converts it to a vldm operator before saving."
            )
        child_name = attr_name.rstrip("_") if attr_name.endswith("_") else attr_name
        child_path = f"{op_path}.{child_name}" if op_path else child_name
        children.append(_serialise_op(child_op, child_name, child_path, state_acc))
        written_child_attrs.append(attr_name)

    # Pass serialised_child_attrs ONLY when load-relevant, so non-relevant ops
    # omit it under model_dump_json(exclude_unset=True) (spec §3/§6):
    #   (1) a child-attr was skipped (None) -> legacy tail-counting would misalign;
    #   (2) the op is a _forward_guard op -> deliberate version marker.
    opspec_kwargs: dict[str, Any] = dict(
        op_id=op_id,
        name=name,
        init_params=init_params,
        state_keys=state_keys,
        children=children,
    )
    skipped = len(written_child_attrs) != len(child_attrs)
    is_guard = bool(getattr(cls, "_forward_guard", False))
    if is_guard or (child_attrs and skipped):
        opspec_kwargs["serialised_child_attrs"] = tuple(written_child_attrs)
    return OpSpec(**opspec_kwargs)


def save(pipeline, out_dir: Path | str, source_meta: dict[str, Any] | None = None) -> None:
    """Write meta.json + pipeline.json + state.npz under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_acc: dict[str, np.ndarray] = {}
    root = _serialise_op(pipeline, "root", "", state_acc)
    spec = PipelineSpec(schema_version=1, root=root)

    n_in = getattr(pipeline, "n_features_in_", -1)
    n_out = -1
    # n_features_out is not always available; best-effort.
    if hasattr(pipeline, "_final_estimator") and hasattr(
        pipeline._final_estimator, "n_features_out_"
    ):
        n_out = int(pipeline._final_estimator.n_features_out_)

    meta = MetaJson(
        schema_version=1,
        vldm_version=_vldm_version,
        sklearn_version=sklearn.__version__,
        numpy_version=np.__version__,
        source=SourceMeta(
            **(
                source_meta
                or {
                    "kind": "tabpfn",
                    "tabpfn_version": None,
                    "extracted_at": datetime.datetime.now(datetime.UTC).isoformat(),
                    "estimator_index": 0,
                }
            )
        ),
        estimator_id=f"estimator_{(source_meta or {}).get('estimator_index', 0)}",
        n_features_in=int(n_in),
        n_features_out_after_pipeline=int(n_out),
    )

    (out_dir / "meta.json").write_text(meta.model_dump_json(indent=2))
    (out_dir / "pipeline.json").write_text(spec.model_dump_json(indent=2, exclude_unset=True))
    np.savez_compressed(out_dir / "state.npz", **state_acc)  # type: ignore[arg-type]  # numpy stub expects bool for keyword args; dict[str, ndarray] is correct at runtime


def load(estimator_dir: Path | str):
    """Read JSON+npz artifact; return a fitted SequentialPipeline.

    Never invokes pickle/joblib. np.load uses allow_pickle=False as a
    hard security gate: any object-dtype array in state.npz raises.
    """
    estimator_dir = Path(estimator_dir)
    if not estimator_dir.exists():
        raise ArtifactNotFoundError(str(estimator_dir))

    for f in ("meta.json", "pipeline.json", "state.npz"):
        if not (estimator_dir / f).exists():
            raise ArtifactNotFoundError(str(estimator_dir / f))

    try:
        meta = MetaJson.model_validate_json((estimator_dir / "meta.json").read_text())
    except Exception as e:
        raise ArtifactSchemaError(f"meta.json: {e}") from e

    try:
        spec = PipelineSpec.model_validate_json((estimator_dir / "pipeline.json").read_text())
    except Exception as e:
        raise ArtifactSchemaError(f"pipeline.json: {e}") from e

    _check_versions(meta)

    try:
        npz = np.load(estimator_dir / "state.npz", allow_pickle=False, mmap_mode="r")
    except ValueError as e:
        raise ArtifactSecurityError(
            f"refusing to load state.npz (allow_pickle=False rejected): {e}"
        ) from e

    # _build_op accesses npz arrays lazily; object-dtype arrays raise ValueError
    # on access when allow_pickle=False — re-raise as ArtifactSecurityError.
    try:
        return _build_op(spec.root, npz, op_path="")
    except ValueError as e:
        if "allow_pickle" in str(e) or "Object arrays" in str(e):
            raise ArtifactSecurityError(
                f"refusing to load state.npz (pickle-encoded array detected): {e}"
            ) from e
        raise


def _check_versions(meta: MetaJson) -> None:
    """Warn on sklearn major version drift; never raises (load proceeds)."""
    import warnings

    cur_skl = sklearn.__version__
    if cur_skl.split(".")[0] != meta.sklearn_version.split(".")[0]:
        warnings.warn(
            f"sklearn major version differs: artifact={meta.sklearn_version} "
            f"runtime={cur_skl}; fitted state may be incompatible",
            VersionMismatchWarning,
            stacklevel=2,
        )


INPUT_SANITIZER_FORMAT_VERSION = 1


def save_input_sanitizer(san, out_dir, source_tabpfn_version: str) -> None:
    """Persist an InputSanitizer to out_dir/ (meta.json + optional codec triple)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "format_version": INPUT_SANITIZER_FORMAT_VERSION,
        "source_tabpfn_version": source_tabpfn_version,
        "n_features_in": san.n_features_in,
        "inferred_categorical_indices": san.inferred_categorical_indices,
        "identity": san.is_identity,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if not san.is_identity:
        # kind="sklearn": this sidecar contains a standalone categorical encoder
        # (sklearn OrdinalEncoder / ColumnTransformer), not a TabPFN ensemble member.
        # estimator_index=-1: sentinel meaning "not an ensemble member"; 0 would falsely
        # imply this is the zeroth TabPFN estimator.
        save(
            san.column_transformer,
            out_dir / "encoder",
            source_meta={
                "kind": "sklearn",
                "tabpfn_version": source_tabpfn_version,
                "extracted_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "estimator_index": -1,
            },
        )


def load_input_sanitizer(in_dir):
    """Reconstruct an InputSanitizer from in_dir/."""
    in_dir = Path(in_dir)
    meta = json.loads((in_dir / "meta.json").read_text())
    if meta.get("format_version") != INPUT_SANITIZER_FORMAT_VERSION:
        raise ValueError(
            f"input_sanitizer format_version {meta.get('format_version')} != "
            f"{INPUT_SANITIZER_FORMAT_VERSION}; re-extract with this vldm"
        )
    if meta["identity"]:
        return InputSanitizer.identity(int(meta["n_features_in"]))
    ct = load(in_dir / "encoder")
    return InputSanitizer(
        n_features_in=int(meta["n_features_in"]),
        inferred_categorical_indices=list(meta["inferred_categorical_indices"]),
        column_transformer=ct,
    )


def _build_op(spec: OpSpec, npz: Any, op_path: str) -> Any:
    """Recursive: instantiate a vldm operator tree from spec + npz."""
    cls: type[VldmEstimatorMixin] = registry.get(spec.op_id)  # type: ignore[assignment]  # registry.get returns `type`; all registered classes are VldmEstimatorMixin subclasses by construction

    # ------------------------------------------------------------------
    # Resolve child_ref dicts: {"child_ref": <int>} → built child object.
    # This handles SequentialPipeline steps and FeatureUnion transformer_list.
    # ColumnTransformer transformers entries have additional keys and are
    # handled separately below.
    # ------------------------------------------------------------------
    children_built: dict[int, Any] = {}

    def _get_child(ref: int | str) -> Any:
        """Build (and cache) child at index ref from spec.children."""
        if ref in ("passthrough", "drop"):
            return ref
        # After the str guards above, ref is guaranteed to be int.
        assert isinstance(ref, int), f"unexpected non-int child ref: {ref!r}"
        if ref not in children_built:
            child_spec = spec.children[ref]
            child_path = f"{op_path}.{child_spec.name}" if op_path else child_spec.name
            children_built[ref] = _build_op(child_spec, npz, child_path)
        return children_built[ref]

    def _resolve(value: Any) -> Any:
        """Recursively replace {"child_ref": <int>} with the built child."""
        if isinstance(value, dict) and set(value.keys()) == {"child_ref"}:
            return _get_child(value["child_ref"])
        if isinstance(value, list):
            return [_resolve(v) for v in value]
        if isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        return value

    init_params = _resolve(spec.init_params)

    # ------------------------------------------------------------------
    # Reshape composite init_params into the form sklearn expects.
    # ------------------------------------------------------------------

    # SequentialPipeline: steps → list[(name, child)]
    if "steps" in init_params and all(isinstance(s, list) for s in init_params["steps"]):
        init_params["steps"] = [tuple(s) for s in init_params["steps"]]

    # FeatureUnion: transformer_list → list[(name, child)]
    if "transformer_list" in init_params and all(
        isinstance(s, list) for s in init_params["transformer_list"]
    ):
        init_params["transformer_list"] = [tuple(s) for s in init_params["transformer_list"]]

    # ColumnTransformer: transformers entries are dicts {name, columns, child_ref}.
    # child_ref here is still an integer (not a {"child_ref":...} singleton dict),
    # so _resolve left it as-is.  We build children here and filter out the
    # remainder placeholder (whose class is not in our registry) — _from_state_dict
    # reconstructs transformers_ (including the remainder) from the fitted state.
    if "transformers" in init_params and isinstance(init_params["transformers"], list):
        first = init_params["transformers"][0] if init_params["transformers"] else None
        if isinstance(first, dict) and "child_ref" in first and "name" in first:
            rebuilt = []
            for entry in init_params["transformers"]:
                ref = entry["child_ref"]
                name_entry = entry["name"]
                cols = entry["columns"]
                # Skip ONLY entries whose child class is unregistered: the
                # serialiser saves post-fit transformers_, which contains the
                # sklearn-generated passthrough FunctionTransformer for
                # remainder="passthrough"; that class isn't in our registry and
                # ColumnTransformer._from_state_dict recreates it from the
                # remainder kwarg + leftover columns. Any OTHER exception
                # (corrupted state, missing init param, etc.) MUST propagate —
                # silently dropping a registered child would produce a
                # subtly-wrong pipeline.
                try:
                    child = _get_child(ref)
                except UnknownOperatorError:
                    continue
                rebuilt.append((name_entry, child, cols))
            remainder_str = init_params.get("remainder", "drop")
            # Carry order_preserving through the reshape (defaults False for
            # pre-existing artifacts that never serialised it = v63 behaviour).
            order_preserving = init_params.get("order_preserving", False)
            init_params = {
                "transformers": rebuilt,
                "remainder": remainder_str,
                "order_preserving": order_preserving,
            }

    # ------------------------------------------------------------------
    # Apply _nan_init_param_keys: JSON null (None) -> np.nan for known NaN params.
    # JSON does not support NaN; it serialises as null and deserialises as None.
    # Subclasses declare which init params may legitimately be NaN so we can
    # convert them back before construction.
    # ------------------------------------------------------------------
    nan_keys: tuple[str, ...] = getattr(cls, "_nan_init_param_keys", ())
    for nan_key in nan_keys:
        if nan_key in init_params and init_params[nan_key] is None:
            init_params[nan_key] = float("nan")

    try:
        obj = cls(**init_params)
    except TypeError as e:
        raise OpInitError(op_path or "<root>", f"{cls.__qualname__}({init_params!r}): {e}") from e

    # ------------------------------------------------------------------
    # Restore fitted state.
    # ------------------------------------------------------------------
    state: dict[str, Any] = {}
    for key in spec.state_keys:
        npz_key = f"{op_path}/{key}" if op_path else key
        if npz_key not in npz.files:
            raise OpStateError(op_path or "<root>", key, "missing in state.npz")
        state[key] = np.asarray(npz[npz_key])

    cls._from_state_dict(obj, state)

    # ------------------------------------------------------------------
    # Restore _child_attrs: fitted attributes that are vldm operators.
    # These were appended to spec.children AFTER composite-specific children
    # by _serialise_op, in the same order as cls._child_attrs.
    # We pick them from the tail of spec.children (matching that order).
    # ------------------------------------------------------------------
    child_attrs: tuple[str, ...] = getattr(cls, "_child_attrs", ())
    if child_attrs:
        if "serialised_child_attrs" in spec.model_fields_set:
            # Explicit form (new): restore exactly the recorded child-attrs, in
            # order, zipped to the tail children. Unlisted child-attrs -> None.
            recorded = spec.serialised_child_attrs
            for name in recorded:
                if name not in child_attrs:
                    raise ArtifactSchemaError(
                        f"{op_path or '<root>'}: serialised_child_attrs lists "
                        f"{name!r}, not in {cls.__qualname__}._child_attrs {child_attrs}"
                    )
            n_recorded = len(recorded)
            if len(spec.children) < n_recorded:
                raise ArtifactSchemaError(
                    f"{op_path or '<root>'}: serialised_child_attrs lists {n_recorded} "
                    f"child-attr(s) but only {len(spec.children)} child spec(s) present"
                )
            tail = spec.children[len(spec.children) - n_recorded :] if n_recorded else []
            built_by_name: dict[str, Any] = {}
            for name, child_spec in zip(recorded, tail, strict=True):
                child_name = name.rstrip("_") if name.endswith("_") else name
                child_path = f"{op_path}.{child_name}" if op_path else child_name
                built_by_name[name] = _build_op(child_spec, npz, child_path)
            for attr_name in child_attrs:
                setattr(obj, attr_name, built_by_name.get(attr_name))
        else:
            # Legacy form (no field): tail-counting; all child-attrs were present.
            if len(spec.children) < len(child_attrs):
                raise ArtifactSchemaError(
                    f"{op_path or '<root>'}: {cls.__qualname__} declares "
                    f"{len(child_attrs)} child-attr(s) but only {len(spec.children)} "
                    f"child spec(s) present"
                )
            n_composite = len(spec.children) - len(child_attrs)
            for i, attr_name in enumerate(child_attrs):
                child_spec = spec.children[n_composite + i]
                child_name = attr_name.rstrip("_") if attr_name.endswith("_") else attr_name
                child_path = f"{op_path}.{child_name}" if op_path else child_name
                built_child = _build_op(child_spec, npz, child_path)
                setattr(obj, attr_name, built_child)

    return obj
