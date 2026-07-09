def test_package_imports():
    import sklearn_tabpfn_ext

    assert sklearn_tabpfn_ext.__version__ == "0.1.0"


def test_tabpfn_public_api_imports_without_tabpfn_or_torch():
    import sys

    from sklearn_tabpfn_ext.tabpfn import (
        TranslationProfile,
        categorical_indices,
        profile_for,
        translate_categorical_step,
        translate_input_sanitizer,
        translate_member,
        translate_sklearn_obj,
    )

    assert TranslationProfile("6.3.2", "v63") == profile_for("6.3.2")
    assert callable(translate_input_sanitizer)
    assert callable(translate_member)
    assert callable(translate_sklearn_obj)
    assert "tabpfn" not in sys.modules
    assert "torch" not in sys.modules
    assert callable(categorical_indices)
    assert callable(translate_categorical_step)
