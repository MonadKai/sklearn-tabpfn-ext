"""sklearn-tabpfn-ext — serializable sklearn-compatible TabPFN preprocessing operators.

Imports every concrete operator at package load time so that the @register
decorators populate OPERATOR_REGISTRY exactly once.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Foundation modules (trigger registration of every class)
from sklearn_tabpfn_ext import (  # noqa: F401
    codec,
    registry,
    schema,
    sklearn_wrappers,
)

# New tabpfn-equivalent operators
from sklearn_tabpfn_ext.adaptive_quantile import AdaptiveQuantileTransformer
from sklearn_tabpfn_ext.base import VldmEstimatorMixin
from sklearn_tabpfn_ext.categorical import (
    CategoricalOrdinalEncoder,
    IdentityCategoricalEncoder,
)

# Composite ops
from sklearn_tabpfn_ext.composite import (
    ColumnTransformer,
    FeatureUnion,
    FunctionTransformer,
    SequentialPipeline,
)

# Core operators
from sklearn_tabpfn_ext.constant_filter import ConstantFeatureFilter
from sklearn_tabpfn_ext.differentiable_znorm import DifferentiableZNorm
from sklearn_tabpfn_ext.fingerprint import FingerprintFeatureAdder
from sklearn_tabpfn_ext.input_sanitizer import InputSanitizer
from sklearn_tabpfn_ext.kdi_with_nan import KDITransformerWithNaN
from sklearn_tabpfn_ext.ordinal_encoder import OrdinalEncoder

# Pipeline layer
from sklearn_tabpfn_ext.pipeline import (
    EstimatorPipeline,
    PreprocessingPipeline,
    build_estimator_pipeline,
)
from sklearn_tabpfn_ext.polynomial_features import PolynomialFeaturesAdder
from sklearn_tabpfn_ext.quantile_svd import QuantileSVDReshaper
from sklearn_tabpfn_ext.safe_power import SafePowerTransformer
from sklearn_tabpfn_ext.shuffle import FeatureShuffler
from sklearn_tabpfn_ext.squashing_scaler import SquashingScaler

__all__ = [
    "AdaptiveQuantileTransformer",
    "CategoricalOrdinalEncoder",
    "ColumnTransformer",
    "ConstantFeatureFilter",
    "DifferentiableZNorm",
    "EstimatorPipeline",
    "FeatureShuffler",
    "FeatureUnion",
    "FingerprintFeatureAdder",
    "FunctionTransformer",
    "IdentityCategoricalEncoder",
    "InputSanitizer",
    "KDITransformerWithNaN",
    "OrdinalEncoder",
    "PolynomialFeaturesAdder",
    "PreprocessingPipeline",
    "QuantileSVDReshaper",
    "SafePowerTransformer",
    "SequentialPipeline",
    "SquashingScaler",
    "VldmEstimatorMixin",
    "build_estimator_pipeline",
]
