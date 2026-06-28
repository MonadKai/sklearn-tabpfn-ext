"""SequentialPipeline — vldm-native replacement for sklearn.pipeline.Pipeline.

Behaves identically to sklearn Pipeline at fit/transform time, but is its own
class so we can register it under a vldm op_id and provide
_to_state_dict / _from_state_dict hooks (no-op: it has no fitted state of
its own; children carry state).
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline as SkPipeline

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.composite.sequential.SequentialPipeline")
class SequentialPipeline(VldmEstimatorMixin, SkPipeline):
    _state_keys = ()
    _init_param_keys = ()  # steps handled specially in codec (children)

    def _to_state_dict(self) -> dict[str, np.ndarray]:
        return {}

    @classmethod
    def _from_state_dict(cls, obj, state):
        # No own state; children restored separately by codec.
        return
