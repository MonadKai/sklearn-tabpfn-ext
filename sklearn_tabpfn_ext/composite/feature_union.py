"""vldm FeatureUnion wrapping sklearn.pipeline.FeatureUnion."""

from __future__ import annotations

from sklearn.pipeline import FeatureUnion as SkFeatureUnion

from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.registry import register


@register("vldm.preprocessing.composite.feature_union.FeatureUnion")
class FeatureUnion(VldmEstimatorMixin, SkFeatureUnion):
    _state_keys = ()
    _init_param_keys = ()  # transformer_list as children

    def _to_state_dict(self):
        return {}

    @classmethod
    def _from_state_dict(cls, obj, state):
        return
