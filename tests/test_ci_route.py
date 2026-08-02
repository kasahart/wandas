"""Contract tests for CI path routing and validation lanes."""

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci_route import CHECKS, classify_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
BASH_GATE_ONLY = pytest.mark.skipif(os.name == "nt", reason="CI Gate runs Bash on ubuntu-latest")
REQUIRED_OUTPUT_NAMES = ("NATIVE_REQUIRED", "LINT_REQUIRED", "DOCS_REQUIRED", "WHEEL_REQUIRED", "PYODIDE_REQUIRED")
RESULT_NAMES = ("NATIVE_RESULT", "LINT_RESULT", "DOCS_RESULT", "WHEEL_RESULT", "PYODIDE_RESULT")
FULL_VALIDATION_JOBS = ("lint", "docs", "core-install-smoke", "pyodide", "native-test")


def _workflow(name: str) -> dict[str, Any]:
    path = REPO_ROOT / ".github" / "workflows" / name
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(data, dict)
    return data


def _decision(**selected: bool) -> dict[str, bool]:
    return {**{check: False for check in CHECKS}, "unknown": False, **selected}


def _gate_script() -> str:
    workflow = _workflow("ci.yml")
    steps = workflow["jobs"]["ci-gate"]["steps"]
    assert len(steps) == 1
    script = steps[0]["run"]
    assert isinstance(script, str)
    return script


def _full_gate_script() -> str:
    workflow = _workflow("full-compatibility.yml")
    steps = workflow["jobs"]["full-gate"]["steps"]
    assert len(steps) == 1
    script = steps[0]["run"]
    assert isinstance(script, str)
    return script


def _run_gate(
    *,
    route_result: str = "success",
    required: dict[str, str] | None = None,
    results: dict[str, str] | None = None,
    omit_required: str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = {
        "ROUTE_RESULT": route_result,
        **dict.fromkeys(REQUIRED_OUTPUT_NAMES, "false"),
        **dict.fromkeys(RESULT_NAMES, "skipped"),
    }
    environment.update(required or {})
    environment.update(results or {})
    if omit_required is not None:
        environment.pop(omit_required)
    return subprocess.run(
        ["bash"],
        input=_gate_script(),
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, **environment},
    )


def _run_full_gate(
    *,
    resolve_result: str = "success",
    resolved_sha: str = "a" * 40,
) -> subprocess.CompletedProcess[str]:
    environment = {
        "RESOLVE_RESULT": resolve_result,
        "RESOLVED_SHA": resolved_sha,
        **dict.fromkeys(RESULT_NAMES, "success"),
    }
    return subprocess.run(
        ["bash"],
        input=_full_gate_script(),
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, **environment},
    )


def test_documentation_only_paths_skip_native_validation() -> None:
    assert classify_paths(["docs/src/contributing.md"]) == _decision(docs=True)
    assert classify_paths(["README.md"]) == _decision(docs=True)
    assert classify_paths(["README.ja.md"]) == _decision(docs=True)
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
    assert classify_paths(["uv.lock"]) == _decision(
        native=True,
        lint=True,
        docs=True,
        wheel=True,
        pyodide=True,
    )
    assert classify_paths(["tests/io/test_wav_io.py"]) == _decision(native=True, lint=True)
    assert classify_paths(["tests/pyodide/test_wav_smoke.py"]) == _decision(
        native=True,
        lint=True,
        pyodide=True,
    )
    assert classify_paths(["tests/docs/test_readme_examples.py"]) == _decision(native=True, lint=True)


@pytest.mark.parametrize(
    "path",
    [
        "docs/src/how-to/pyodide-browser.md",
        "examples/pyodide/index.html",
        "scripts/test_pyodide.sh",
        "scripts/run_pyodide_tests.mjs",
    ],
)
def test_pyodide_guide_example_and_harness_select_pyodide(path: str) -> None:
    assert classify_paths([path])["pyodide"] is True


@pytest.mark.parametrize(
    "path",
    [
        "tests/core/test_base_frame.py",
        "tests/frame_helpers.py",
        "tests/__init__.py",
        "tests/pyodide/test_wav_smoke.py",
    ],
)
def test_pyodide_harness_inputs_select_pyodide(path: str) -> None:
    assert classify_paths([path])["pyodide"] is True


def test_unrelated_test_changes_do_not_select_pyodide() -> None:
    assert classify_paths(["tests/io/test_wav_io.py"]) == _decision(native=True, lint=True)


def test_path_routing_uses_the_union_of_all_changed_paths() -> None:
    assert classify_paths(["tests/core/test_base_frame.py", "docs/src/contributing.md"]) == _decision(
        native=True,
        lint=True,
        docs=True,
        pyodide=True,
    )


def test_workflow_changes_select_every_check() -> None:
    assert classify_paths([".github/workflows/ci.yml"]) == _decision(
        native=True,
        lint=True,
        docs=True,
        wheel=True,
        pyodide=True,
    )


def test_routing_script_changes_select_every_check() -> None:
    assert classify_paths(["scripts/ci_route.py"]) == _decision(
        native=True,
        lint=True,
        docs=True,
        wheel=True,
        pyodide=True,
    )


def test_unknown_or_empty_paths_select_everything() -> None:
    expected = {**{check: True for check in CHECKS}, "unknown": True}

    assert classify_paths([".vscode/settings.json"]) == expected
    assert classify_paths(["scripts/new_tool.sh"]) == expected
    assert classify_paths(["__ci_route_unknown__"]) == expected
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


def test_documentation_job_runs_contract_tests_without_coverage() -> None:
    workflow = _workflow("ci.yml")
    docs_job = workflow["jobs"]["docs"]
    steps = docs_job["steps"]

    sync_commands = [step["run"] for step in steps if isinstance(step, dict) and "uv sync" in step.get("run", "")]
    contract_commands = [
        step["run"] for step in steps if isinstance(step, dict) and "tests/docs" in step.get("run", "")
    ]

    assert len(sync_commands) == 1
    assert "--group docs --group test" in sync_commands[0]
    assert contract_commands == ["uv run --no-sync pytest tests/docs -q --no-cov"]


def test_full_lane_preserves_the_ten_environment_compatibility_matrix() -> None:
    workflow = _workflow("full-compatibility.yml")
    matrix = workflow["jobs"]["native-test"]["strategy"]["matrix"]["include"]
    resolver = workflow["jobs"]["resolve-ref"]
    resolver_step = next(step for step in resolver["steps"] if step.get("id") == "resolve")
    resolver_checkout = next(step for step in resolver["steps"] if step.get("uses") == "actions/checkout@v4")

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
    assert resolver["outputs"]["resolved_sha"] == "${{ steps.resolve.outputs.resolved_sha }}"
    assert resolver_checkout["with"]["ref"] == "${{ github.sha }}"
    assert resolver_checkout["with"]["fetch-depth"] == "0"
    assert resolver_step["env"]["REQUESTED_REF"] == "${{ inputs.ref || github.sha }}"
    assert "git ls-remote origin" in resolver_step["run"]
    assert "Resolved SHA:" in resolver_step["run"]
    assert workflow["jobs"]["full-gate"]["needs"] == ["resolve-ref", *FULL_VALIDATION_JOBS]
    for job_name in FULL_VALIDATION_JOBS:
        assert workflow["jobs"][job_name]["needs"] == "resolve-ref"
        checkout_steps = [
            step for step in workflow["jobs"][job_name]["steps"] if step.get("uses") == "actions/checkout@v4"
        ]
        assert len(checkout_steps) == 1
        checkout_ref = checkout_steps[0]["with"]["ref"]
        assert checkout_ref == "${{ needs['resolve-ref'].outputs.resolved_sha }}"
        assert "inputs.ref" not in checkout_ref
        assert "github.ref" not in checkout_ref


def test_release_publish_waits_for_full_compatibility() -> None:
    workflow = _workflow("cd.yml")
    full_job = workflow["jobs"]["full-compatibility"]
    build_job = workflow["jobs"]["build"]
    installation_job = workflow["jobs"]["test-installation"]
    publish_job = workflow["jobs"]["publish-to-pypi"]

    assert full_job["uses"] == "./.github/workflows/full-compatibility.yml"
    assert full_job["with"]["ref"] == "${{ github.sha }}"
    assert build_job["needs"] == "full-compatibility"
    assert "full-compatibility" in publish_job["needs"]
    build_checkout = [step for step in build_job["steps"] if step.get("uses") == "actions/checkout@v4"]
    assert len(build_checkout) == 1
    assert build_checkout[0]["with"]["ref"] == "${{ github.sha }}"
    installation_checkout = [step for step in installation_job["steps"] if step.get("uses") == "actions/checkout@v4"]
    assert len(installation_checkout) == 1
    assert installation_checkout[0]["with"]["ref"] == "${{ github.sha }}"
    assert installation_job["needs"] == "build"


@pytest.mark.parametrize(
    ("resolve_result", "resolved_sha"),
    [("failure", ""), ("success", ""), ("success", "not-a-sha")],
)
@BASH_GATE_ONLY
def test_full_gate_rejects_resolver_failure_or_invalid_sha(resolve_result: str, resolved_sha: str) -> None:
    result = _run_full_gate(resolve_result=resolve_result, resolved_sha=resolved_sha)

    assert result.returncode != 0
    assert "resolver" in result.stderr.lower()


@BASH_GATE_ONLY
def test_ci_gate_rejects_route_failure() -> None:
    result = _run_gate(route_result="failure")

    assert result.returncode != 0
    assert "routing job" in result.stderr


@pytest.mark.parametrize("output_name", REQUIRED_OUTPUT_NAMES)
@BASH_GATE_ONLY
def test_ci_gate_rejects_missing_required_outputs(output_name: str) -> None:
    result = _run_gate(omit_required=output_name)

    assert result.returncode != 0
    assert "Invalid CI routing outputs" in result.stderr


@pytest.mark.parametrize("output_name", REQUIRED_OUTPUT_NAMES)
@pytest.mark.parametrize("invalid_value", ["", "TRUE", "unexpected"])
@BASH_GATE_ONLY
def test_ci_gate_rejects_missing_or_invalid_required_outputs(output_name: str, invalid_value: str) -> None:
    result = _run_gate(required={output_name: invalid_value})

    assert result.returncode != 0
    assert "Invalid CI routing outputs" in result.stderr


@pytest.mark.parametrize(
    ("required_name", "result_name"),
    list(zip(REQUIRED_OUTPUT_NAMES, RESULT_NAMES, strict=True)),
)
@pytest.mark.parametrize("job_result", ["failure", "cancelled", "skipped"])
@BASH_GATE_ONLY
def test_ci_gate_rejects_non_success_for_required_jobs(
    required_name: str,
    result_name: str,
    job_result: str,
) -> None:
    result = _run_gate(required={required_name: "true"}, results={result_name: job_result})

    assert result.returncode != 0
    assert "Selected CI checks did not succeed" in result.stderr


@BASH_GATE_ONLY
def test_ci_gate_accepts_intentionally_skipped_non_required_jobs() -> None:
    result = _run_gate()

    assert result.returncode == 0
    assert "intentionally skipped checks are accepted" in result.stdout
