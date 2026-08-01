"""Contract tests for CI path routing and validation lanes."""

from pathlib import Path
from typing import Any

import yaml

from scripts.ci_route import CHECKS, classify_paths

REPO_ROOT = Path(__file__).resolve().parents[2]


def _workflow(name: str) -> dict[str, Any]:
    path = REPO_ROOT / ".github" / "workflows" / name
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(data, dict)
    return data


def _decision(**selected: bool) -> dict[str, bool]:
    return {**{check: False for check in CHECKS}, "unknown": False, **selected}


def test_documentation_only_paths_skip_native_validation() -> None:
    assert classify_paths(["docs/src/contributing.md"]) == _decision(docs=True)
    assert classify_paths(["README.md"]) == _decision(docs=True)
    assert classify_paths(["learning-path/03_working_with_frames.py"]) == _decision(lint=True, docs=True)


def test_product_and_configuration_paths_select_required_checks() -> None:
    assert classify_paths(["wandas/frames/base.py"]) == _decision(
        native=True,
        lint=True,
        docs=True,
        wheel=True,
        pyodide=True,
    )
    assert classify_paths(["pyproject.toml"]) == _decision(
        native=True,
        lint=True,
        docs=True,
        wheel=True,
        pyodide=True,
    )
    assert classify_paths(["tests/pyodide/test_wav_smoke.py"]) == _decision(
        native=True,
        lint=True,
        pyodide=True,
    )


def test_unknown_or_empty_paths_select_everything() -> None:
    expected = {**{check: True for check in CHECKS}, "unknown": True}

    assert classify_paths([".vscode/settings.json"]) == expected
    assert classify_paths([]) == expected


def test_fast_lane_has_three_representative_native_environments() -> None:
    workflow = _workflow("ci.yml")
    matrix = workflow["jobs"]["native-test"]["strategy"]["matrix"]["include"]

    assert {(item["os"], item["python-version"], item["coverage"]) for item in matrix} == {
        ("ubuntu-latest", "3.10", "false"),
        ("ubuntu-latest", "3.14", "true"),
        ("windows-latest", "3.14", "false"),
    }
    assert "--no-cov" in workflow["jobs"]["native-test"]["steps"][3]["run"]
    assert workflow["jobs"]["ci-gate"]["if"] == "always()"
    assert workflow["jobs"]["ci-gate"]["name"] == "CI Gate"


def test_full_lane_preserves_the_ten_environment_compatibility_matrix() -> None:
    workflow = _workflow("full-compatibility.yml")
    matrix = workflow["jobs"]["native-test"]["strategy"]["matrix"]["include"]

    assert len(matrix) == 10
    assert {(item["os"], item["python-version"]) for item in matrix} == {
        (os_name, python_version)
        for os_name in ("ubuntu-latest", "windows-latest")
        for python_version in ("3.10", "3.11", "3.12", "3.13", "3.14")
    }
    assert sum(item["coverage"] == "true" for item in matrix) == 1
    assert workflow["on"]["schedule"]
    assert "workflow_dispatch" in workflow["on"]
    assert "workflow_call" in workflow["on"]
    assert workflow["jobs"]["full-gate"]["if"] == "always()"


def test_release_publish_waits_for_full_compatibility() -> None:
    workflow = _workflow("cd.yml")
    full_job = workflow["jobs"]["full-compatibility"]
    publish_job = workflow["jobs"]["publish-to-pypi"]

    assert full_job["uses"] == "./.github/workflows/full-compatibility.yml"
    assert full_job["with"]["ref"] == "${{ github.sha }}"
    assert "full-compatibility" in publish_job["needs"]
