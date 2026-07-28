from __future__ import annotations

import inspect
import re
from collections import Counter
from pathlib import Path

import pytest

_UNSUPPORTED_IMPORT = re.compile(r"""pytest\.importorskip\(\s*["'](torch|tensorflow)["']""")
_SKIP_REASONS = {
    "torch": "PyTorch does not provide a supported Pyodide wheel; excluded from Pyodide CI",
    "tensorflow": "TensorFlow does not provide a supported Pyodide wheel; excluded from Pyodide CI",
}
_allowed_skips: dict[str, str] = {}
_unexpected_skips: list[str] = []
_results: Counter[tuple[str, str]] = Counter()
_collected: Counter[str] = Counter()


def _suite(nodeid: str) -> str:
    if "/core/" in nodeid:
        return "core"
    if "/pyodide/test_wav_smoke.py" in nodeid:
        return "wav_smoke"
    return "other"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "pyodide_unsupported(dependency): test requires a dependency unavailable in Pyodide",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    _allowed_skips.clear()
    _unexpected_skips.clear()
    _results.clear()
    _collected.clear()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        test_object = getattr(item, "obj", None)
        if test_object is None:
            continue
        try:
            source = inspect.getsource(test_object)
        except (OSError, TypeError):
            continue
        match = _UNSUPPORTED_IMPORT.search(source)
        if match is None:
            continue
        dependency = match.group(1)
        reason = _SKIP_REASONS[dependency]
        item.add_marker(pytest.mark.pyodide_unsupported(dependency))
        item.add_marker(pytest.mark.skip(reason=reason))
        _allowed_skips[item.nodeid] = dependency


def pytest_collection_finish(session: pytest.Session) -> None:
    for item in session.items:
        _collected[_suite(item.nodeid)] += 1

    core_collected = _collected["core"]
    wav_collected = _collected["wav_smoke"]
    print(f"PYODIDE_CORE_COLLECTED={core_collected}")
    print(f"PYODIDE_WAV_SMOKE_COLLECTED={wav_collected}")
    if core_collected == 0:
        raise pytest.UsageError("Pyodide core collection produced zero tests")
    if wav_collected == 0:
        raise pytest.UsageError("Pyodide WAV smoke collection produced zero tests")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call" and report.passed:
        _results[(_suite(report.nodeid), "passed")] += 1
    elif report.failed:
        _results[(_suite(report.nodeid), "failed")] += 1
    elif report.skipped:
        _results[(_suite(report.nodeid), "skipped")] += 1
        if report.nodeid not in _allowed_skips:
            _unexpected_skips.append(report.nodeid)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    for suite in ("core", "wav_smoke"):
        label = suite.upper()
        print(f"PYODIDE_{label}_PASSED={_results[(suite, 'passed')]}")
        print(f"PYODIDE_{label}_FAILED={_results[(suite, 'failed')]}")
        print(f"PYODIDE_{label}_SKIPPED={_results[(suite, 'skipped')]}")

    if _allowed_skips:
        print("Intentional Pyodide skips:")
        for nodeid, dependency in sorted(_allowed_skips.items()):
            print(f"  {Path(nodeid).name}: {_SKIP_REASONS[dependency]}")
    if _unexpected_skips:
        terminalreporter.write_sep("=", "unexpected Pyodide skips", red=True)
        for nodeid in _unexpected_skips:
            terminalreporter.write_line(nodeid, red=True)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _unexpected_skips:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
