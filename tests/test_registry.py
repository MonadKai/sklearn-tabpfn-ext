"""Registry contract tests."""

from __future__ import annotations

import pytest

from sklearn_tabpfn_ext import registry as reg
from sklearn_tabpfn_ext.exceptions import UnknownOperatorError


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Each test starts with empty registries."""
    monkeypatch.setattr(reg, "OPERATOR_REGISTRY", {})
    monkeypatch.setattr(reg, "OP_ALIASES", {})


def test_register_and_get():
    @reg.register("vldm.test.A")
    class A:
        pass

    assert reg.get("vldm.test.A") is A


def test_register_duplicate_raises():
    @reg.register("vldm.test.A")
    class _A1:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @reg.register("vldm.test.A")
        class _A2:
            pass


def test_get_unknown_raises():
    with pytest.raises(UnknownOperatorError) as ei:
        reg.get("vldm.does.NotExist")
    assert "vldm.does.NotExist" in str(ei.value)


def test_alias_redirect():
    @reg.register("vldm.test.NewName")
    class New:
        pass

    reg.OP_ALIASES["vldm.test.OldName"] = "vldm.test.NewName"
    assert reg.get("vldm.test.OldName") is New


def test_get_unknown_lists_available():
    @reg.register("vldm.test.A")
    class _A:
        pass

    @reg.register("vldm.test.B")
    class _B:
        pass

    with pytest.raises(UnknownOperatorError) as ei:
        reg.get("vldm.test.C")
    msg = str(ei.value)
    assert "vldm.test.A" in msg
    assert "vldm.test.B" in msg
