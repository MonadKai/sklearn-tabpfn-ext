"""sklearn-class wrappers for vldm preprocessing artifacts.

Each wrapper subclasses the sklearn class, adds VldmEstimatorMixin and a
register decorator, and declares _state_keys/_init_param_keys. transform
behavior is inherited unchanged.
"""

from __future__ import annotations

from sklearn_tabpfn_ext.sklearn_wrappers.simple_imputer import SimpleImputer

# Import every wrapper module so the @register decorators run.
from sklearn_tabpfn_ext.sklearn_wrappers.standard_scaler import StandardScaler
from sklearn_tabpfn_ext.sklearn_wrappers.truncated_svd import TruncatedSVD

__all__ = [
    "SimpleImputer",
    "StandardScaler",
    "TruncatedSVD",
]
