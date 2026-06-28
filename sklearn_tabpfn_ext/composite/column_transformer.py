"""vldm ColumnTransformer wrapping sklearn.compose.ColumnTransformer."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer as SkColumnTransformer

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.composite.column_transformer.ColumnTransformer")
class ColumnTransformer(VldmEstimatorMixin, SkColumnTransformer):
    _state_keys = ("n_features_in_",)
    _init_param_keys = (
        "remainder",
        "order_preserving",
    )  # transformers handled as children by codec

    def __init__(
        self,
        transformers,
        *,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True,
        force_int_remainder_cols=True,
        order_preserving: bool = False,
    ):
        super().__init__(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out,
            force_int_remainder_cols=force_int_remainder_cols,
        )
        self.order_preserving = order_preserving

    def transform(self, X):
        out = super().transform(X)
        if not self.order_preserving:
            return out
        import numpy as np

        X = np.asarray(X)
        n_cols = X.shape[1]
        # Order-preserving is only well-defined when the transform is NET
        # width-preserving: every input column appears in the output exactly
        # once (one-to-one encoder blocks + passthrough remainder, no drop).
        # A width-changing block OR a dropped column makes out.shape[1] != n_cols.
        if out.shape[1] != n_cols:
            raise ValueError(
                f"order_preserving requires a net width-preserving transform "
                f"(one-to-one blocks, no drop); got output width {out.shape[1]} "
                f"!= {n_cols} input columns"
            )
        # transformers_ tuples are (name, transformer, cols). sklearn omits a
        # block whose TRANSFORMER is "drop" (2nd element). Scatter each emitted
        # block (incl. the appended remainder) back to its original columns,
        # consuming `out` left-to-right.
        result = np.empty((X.shape[0], n_cols), dtype=out.dtype)
        offset = 0
        for _name, trans, cols in self.transformers_:
            if trans == "drop":  # dropped block emits nothing into `out`
                continue
            # cols selects this block's columns; its output IS present in `out`.
            # Normalise to a concrete index list (slice / ndarray / tuple) rather
            # than skipping non-lists — skipping would leave the block's output in
            # `out` while not advancing `offset`, desyncing every later slice
            # (silent column corruption).
            if isinstance(cols, slice):
                cols = list(range(*cols.indices(n_cols)))
            elif not isinstance(cols, list):
                cols = list(cols)
            width = len(cols)
            result[:, cols] = out[:, offset : offset + width]
            offset += width
        return result

    def _to_state_dict(self):
        import numpy as np

        return {"n_features_in_": np.asarray(self.n_features_in_)}

    @classmethod
    def _from_state_dict(cls, obj, state):
        # 1. n_features_in_: required by _check_n_features in transform().
        if "n_features_in_" in state:
            obj.n_features_in_ = int(state["n_features_in_"])

        # 2. Derive the "leftover" columns not covered by any transformer.
        #    Phase A always uses explicit integer-list columns from the translator.
        used: set[int] = set()
        for _, _, cols in obj.transformers:
            if isinstance(cols, list):
                used.update(cols)
        leftover = [i for i in range(obj.n_features_in_) if i not in used]

        # 3. _columns: per-transformer column specs (unfitted path in _iter).
        #    Used by _iter(fitted=False) which _get_empty_routing calls on every
        #    transform() invocation (sklearn 1.6.1 routing shim).
        obj._columns = [cols for _, _, cols in obj.transformers]

        # 4. _remainder: (name, transformer_or_string, leftover_indices).
        #    Also consumed by _iter(fitted=False) to chain the remainder entry.
        if obj.remainder in ("drop", "passthrough"):
            obj._remainder = ("remainder", obj.remainder, leftover)
        else:
            # estimator instance passed as remainder - not expected in Phase A
            # but handled gracefully to avoid a hard crash.
            obj._remainder = ("remainder", obj.remainder, leftover)

        # 5. transformers_: the post-fit list used by _iter(fitted=True) in the
        #    hot path of transform().  Children are already fitted (provided by
        #    codec at construction time).  For "passthrough" remainder sklearn
        #    stores a FunctionTransformer (not the string), because _transform_one
        #    calls .transform() on it directly; the string substitution only
        #    happens inside _fit_transform_one which we never call here.
        from sklearn.preprocessing import FunctionTransformer

        transformers_ = list(obj.transformers)  # user-provided (children already fitted)
        if leftover:
            if obj.remainder == "passthrough":
                passthrough_ft = FunctionTransformer(
                    accept_sparse=True,
                    check_inverse=False,
                    feature_names_out="one-to-one",
                )
                transformers_.append(("remainder", passthrough_ft, leftover))
            elif obj.remainder == "drop":
                # "drop" is filtered by skip_drop=True in _iter; still append so
                # _get_empty_routing can see the name in its routing Bunch.
                transformers_.append(("remainder", "drop", leftover))
            else:
                transformers_.append(("remainder", obj.remainder, leftover))
        obj.transformers_ = transformers_

        # 6. sparse_output_: read by _hstack on every transform() call.
        #    Phase A operates on dense numpy arrays exclusively; set False.
        obj.sparse_output_ = False
