"""Pydantic v2 schemas for vldm preprocessing artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceMeta(_StrictModel):
    kind: Literal["tabpfn", "sklearn"] = "tabpfn"
    tabpfn_version: str | None = None
    extracted_at: str
    estimator_index: int
    # Phase A+ extension (2026-05-08): sha256 hex of the source pkl. Used by
    # vldm-sigma-server's parity-test fixtures to verify the artifact and
    # the fixture were derived from the same input pickle.
    tabpfn_pickle_sha256: str | None = None
    # Phase A+ extension (2026-05-09): record clf.inference_precision from
    # the source pkl. vldm runtime does NOT consume these fields directly
    # (EngineConfig.dtype controls runtime), but recording serves three
    # purposes: (1) audit trail for cross-path debugging, (2) explicit
    # warning if the source pkl carries non-deterministic "auto" /
    # "autocast", (3) forward compat if vldm consumes inference_precision
    # later. See
    # docs/superpowers/specs/2026-05-09-extract-inference-precision-check-design.md
    # `original` is the canonical-string normalization of the raw pkl
    # value; `resolved` maps "auto"/"autocast" → "float32" and passes
    # explicit dtypes through.
    inference_precision_original: str | None = None
    inference_precision_resolved: str | None = None
    # Phase 0.0.4 extension (#38, offline-spec stage 2): provenance recorded at
    # EXTRACT time (the conda tabpfn env where extract_model.py runs — in the
    # standard single-env workflow this is also where the model was fitted).
    # Makes the artifact self-describing for deploy-side version-mismatch
    # detection + reproducibility. All Optional so stage-1 artifacts (pre-#38)
    # still validate. NB: the version fields duplicate MetaJson's top-level
    # sklearn_version/numpy_version (same extract env) — kept here too so the
    # source.* block is self-contained.
    sklearn_version: str | None = None
    numpy_version: str | None = None
    torch_version: str | None = None
    python_version: str | None = None
    n_estimators: int | None = None
    n_features: int | None = None
    n_train_rows: int | None = None
    training_hyperparams: dict[str, Any] | None = None

    @field_validator("tabpfn_pickle_sha256")
    @classmethod
    def _tabpfn_pickle_sha256_is_valid_hex_digest(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) != 64 or any(c not in "0123456789abcdefABCDEF" for c in v):
            raise ValueError(
                "tabpfn_pickle_sha256 must be a 64-character hexadecimal sha256 digest"
            )
        return v


class MetaJson(_StrictModel):
    schema_version: Literal[1] = 1
    vldm_version: str
    sklearn_version: str
    numpy_version: str
    source: SourceMeta
    estimator_id: str
    n_features_in: int
    n_features_out_after_pipeline: int


class OpSpec(_StrictModel):
    op_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    init_params: dict[str, Any]
    state_keys: list[str]
    # Pydantic v2 self-reference: keep the string-literal forward ref
    # to avoid relying on `from __future__ import annotations` + a
    # later `model_rebuild()` call. Ruff's UP-rule may try to un-quote
    # this on auto-fix; keep the string and resist if so.
    children: list["OpSpec"] = Field(default_factory=list)  # noqa: UP037
    # Optional: names of _child_attrs actually serialised (in order). Written
    # only when load-relevant (a child-attr was skipped, or the op is a
    # _forward_guard op). Absent ⇒ legacy tail-counting restore. See codec.py
    # and spec §3/§6.
    serialised_child_attrs: tuple[str, ...] = ()

    @field_validator("op_id")
    @classmethod
    def _op_id_dotted(cls, v: str) -> str:
        if "." not in v:
            raise ValueError("op_id must be a dotted fully-qualified class name")
        return v


class PipelineSpec(_StrictModel):
    schema_version: Literal[1] = 1
    root: OpSpec
