from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from tests.pyodide import pytest_plugin


def test_collection_time_skip_fails_the_pyodide_session() -> None:
    session_state = SimpleNamespace(exitstatus=pytest.ExitCode.OK)
    session = cast(pytest.Session, session_state)
    report = cast(
        pytest.CollectReport,
        SimpleNamespace(
            skipped=True,
            nodeid="/work/tests/core/test_module_level_skip.py",
        ),
    )

    pytest_plugin.pytest_sessionstart(session)
    try:
        pytest_plugin.pytest_collectreport(report)
        pytest_plugin.pytest_sessionfinish(session, pytest.ExitCode.OK)
        assert session_state.exitstatus == pytest.ExitCode.TESTS_FAILED
    finally:
        pytest_plugin.pytest_sessionstart(session)
