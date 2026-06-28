import numpy as np

from sklearn_tabpfn_ext.input_sanitizer import InputSanitizer
from sklearn_tabpfn_ext.pipeline import PreprocessingPipeline


def test_pipeline_applies_sanitizer_first_then_identity_is_noop():
    class _Echo:
        def transform(self, X):
            return np.asarray(X)

    X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    # identity sanitizer: output equals the no-sanitizer path
    pp_id = PreprocessingPipeline(
        ensemble_pipelines=[_Echo()], input_sanitizer=InputSanitizer.identity(3)
    )
    out_id = pp_id.transform(X)[0]
    assert np.array_equal(out_id, X.astype(np.float32))

    # a non-identity sanitizer that swaps columns 0 and 2 must be applied first.
    # Duck-typed: PreprocessingPipeline.transform only calls `.transform`.
    class _Swap:
        def transform(self, X):
            X = np.asarray(X, dtype=np.float64).copy()
            X[:, [0, 2]] = X[:, [2, 0]]
            return X

    pp_sw = PreprocessingPipeline(ensemble_pipelines=[_Echo()], input_sanitizer=_Swap())
    out_sw = pp_sw.transform(X)[0]
    assert np.array_equal(out_sw, np.array([[3.0, 2.0, 1.0]], dtype=np.float32))


def test_pipeline_default_no_sanitizer_unchanged():
    # Backward compat: constructing without input_sanitizer keeps pre-fix behaviour.
    class _Echo:
        def transform(self, X):
            return np.asarray(X)

    X = np.array([[5.0, 6.0]], dtype=np.float32)
    pp = PreprocessingPipeline(ensemble_pipelines=[_Echo()])
    assert np.array_equal(pp.transform(X)[0], X.astype(np.float32))
