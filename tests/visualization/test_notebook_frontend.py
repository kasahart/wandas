"""Contracts for detecting a notebook front end that can render rich output."""

from __future__ import annotations

import sys
import types

import pytest

from wandas.visualization.notebook import in_notebook_frontend


class _TerminalShell:
    pass


class _KernelShell:
    pass


def _install_ipython(monkeypatch: pytest.MonkeyPatch, shell: object | None, *, terminal: bool) -> None:
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    if terminal:
        module = types.ModuleType("IPython.terminal.interactiveshell")
        module.TerminalInteractiveShell = _TerminalShell  # ty: ignore[unresolved-attribute]
        monkeypatch.setitem(sys.modules, "IPython.terminal.interactiveshell", module)
    else:
        monkeypatch.delitem(sys.modules, "IPython.terminal.interactiveshell", raising=False)


@pytest.fixture(autouse=True)
def _no_marimo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "marimo", raising=False)


def test_plain_script_without_ipython_is_not_a_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "IPython", raising=False)

    assert in_notebook_frontend() is False


def test_ipython_installed_without_a_shell_is_not_a_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_ipython(monkeypatch, None, terminal=False)

    assert in_notebook_frontend() is False


def test_terminal_ipython_is_not_a_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_ipython(monkeypatch, _TerminalShell(), terminal=True)

    assert in_notebook_frontend() is False


def test_ipython_kernel_is_a_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_ipython(monkeypatch, _KernelShell(), terminal=True)

    assert in_notebook_frontend() is True


@pytest.mark.parametrize("running", [True, False])
def test_marimo_follows_running_in_notebook(monkeypatch: pytest.MonkeyPatch, running: bool) -> None:
    marimo = types.ModuleType("marimo")
    marimo.running_in_notebook = lambda: running  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "marimo", marimo)
    monkeypatch.delitem(sys.modules, "IPython", raising=False)

    assert in_notebook_frontend() is running
