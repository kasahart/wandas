from __future__ import annotations

import importlib.metadata
import os
import platform
import sys

import pytest


def main() -> int:
    os.chdir("/work")
    print(f"Wandas wheel version: {importlib.metadata.version('wandas')}")
    locked_dependencies = ("cattrs", "dask", "locket", "partd")
    locked_versions = ", ".join(f"{name}={importlib.metadata.version(name)}" for name in locked_dependencies)
    print(f"Locked PyPI dependencies: {locked_versions}")
    print(f"Python: {platform.python_version()} ({sys.platform})")
    print("Running Pyodide core tests and WAV smoke tests")
    return int(
        pytest.main(
            [
                "-p",
                "tests.pyodide.pytest_plugin",
                "--strict-config",
                "--strict-markers",
                "--tb=long",
                "-ra",
                "-s",
                "-vv",
                "/work/tests/core",
                "/work/tests/pyodide/test_wav_smoke.py",
            ]
        )
    )
