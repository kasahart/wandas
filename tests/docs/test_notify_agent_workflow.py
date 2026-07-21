"""Contract tests for cross-repository release notification."""

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "notify-agent.yml"
BASH_WORKFLOW_ONLY = pytest.mark.skipif(
    os.name == "nt",
    reason="notify-agent runs with Bash on ubuntu-latest, not Windows bash.exe/WSL",
)


def _workflow() -> dict[str, Any]:
    data = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(data, dict)
    return data


def _steps_by_name() -> dict[str, dict[str, Any]]:
    steps = _workflow()["jobs"]["notify"]["steps"]
    assert isinstance(steps, list)
    return {str(step["name"]): step for step in steps}


def _run_script(
    tmp_path: Path,
    script: str,
    *,
    event_name: str = "push",
    input_tag: str = "",
    push_tag: str = "",
    existing_tag: str = "",
    token: str = "",
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        """#!/usr/bin/env bash
if [[ "$*" == *"refs/tags/${EXISTING_TAG}"* && -n "${EXISTING_TAG}" ]]; then
  exit 0
fi
exit 2
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    output_path = tmp_path / "github-output"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "EVENT_NAME": event_name,
        "INPUT_TAG": input_tag,
        "PUSH_TAG": push_tag,
        "EXISTING_TAG": existing_tag,
        "GITHUB_OUTPUT": str(output_path),
        "GITHUB_REPOSITORY": "kasahart/wandas",
        "WANDAS_AGENT_TOKEN": token,
    }
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    outputs = {}
    if output_path.exists():
        outputs = dict(line.split("=", maxsplit=1) for line in output_path.read_text(encoding="utf-8").splitlines())
    return result, outputs


def test_notify_agent_accepts_tag_push_and_explicit_replay() -> None:
    workflow = _workflow()
    triggers = workflow["on"]

    assert triggers["push"]["tags"] == ["v*.*.*"]
    replay = triggers["workflow_dispatch"]["inputs"]["tag"]
    assert replay == {
        "description": "Existing strict SemVer tag to notify or replay (vX.Y.Z)",
        "required": "true",
        "type": "string",
    }
    assert workflow["jobs"]["notify"]["if"] == (
        "github.event_name == 'workflow_dispatch' || github.event.deleted == false"
    )


def test_notify_agent_resolves_only_existing_strict_release_tags() -> None:
    resolve = _steps_by_name()["Resolve strict SemVer release tag"]
    script = resolve["run"]

    assert resolve["id"] == "release"
    assert resolve["env"] == {
        "EVENT_NAME": "${{ github.event_name }}",
        "INPUT_TAG": "${{ inputs.tag }}",
        "PUSH_TAG": "${{ github.ref_name }}",
    }
    assert "^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$" in script
    assert "git ls-remote --exit-code --tags" in script
    assert "::error title=Invalid release tag::" in script
    assert "::error title=Unknown release tag::" in script


def test_notify_agent_reports_missing_least_privilege_credential() -> None:
    credential = _steps_by_name()["Validate notification credential"]
    script = credential["run"]

    assert credential["if"] == "steps.release.outputs.valid == 'true'"
    assert credential["env"] == {"WANDAS_AGENT_TOKEN": "${{ secrets.WANDAS_AGENT_TOKEN }}"}
    assert "::error title=Missing WANDAS_AGENT_TOKEN::" in script
    assert "scoped only to kasahart/wandas-agent" in script
    assert "Contents: Read and write" in script


def test_notify_agent_dispatches_the_resolved_tag() -> None:
    dispatch = _steps_by_name()["Trigger wandas-agent submodule update"]

    assert dispatch["if"] == "steps.release.outputs.valid == 'true'"
    assert dispatch["uses"] == ("peter-evans/repository-dispatch@ff45666b9427631e3450c54a1bcbee4d9ff4d7c0")
    assert dispatch["with"] == {
        "token": "${{ secrets.WANDAS_AGENT_TOKEN }}",
        "repository": "kasahart/wandas-agent",
        "event-type": "wandas-updated",
        "client-payload": '{"tag": "${{ steps.release.outputs.tag }}"}',
    }


@BASH_WORKFLOW_ONLY
def test_notify_agent_shell_scripts_are_valid_bash() -> None:
    steps = _steps_by_name()

    for name in (
        "Resolve strict SemVer release tag",
        "Validate notification credential",
    ):
        result = subprocess.run(
            ["bash", "-n"],
            input=steps[name]["run"],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("event_name", "input_tag", "push_tag", "existing_tag", "expected_outputs"),
    [
        (
            "workflow_dispatch",
            "v0.6.0",
            "main",
            "v0.6.0",
            {"valid": "true", "tag": "v0.6.0"},
        ),
        ("push", "", "v1.2.3", "", {"valid": "true", "tag": "v1.2.3"}),
        ("push", "", "v1.2.3-rc.1", "", {"valid": "false"}),
    ],
)
@BASH_WORKFLOW_ONLY
def test_notify_agent_resolver_accepts_or_skips_expected_tags(
    tmp_path: Path,
    event_name: str,
    input_tag: str,
    push_tag: str,
    existing_tag: str,
    expected_outputs: dict[str, str],
) -> None:
    script = _steps_by_name()["Resolve strict SemVer release tag"]["run"]

    result, outputs = _run_script(
        tmp_path,
        script,
        event_name=event_name,
        input_tag=input_tag,
        push_tag=push_tag,
        existing_tag=existing_tag,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs == expected_outputs


@pytest.mark.parametrize("tag", ["v01.2.3", "v1.02.3", "v1.2.03", "v1.2"])
@BASH_WORKFLOW_ONLY
def test_notify_agent_manual_replay_rejects_non_strict_tags(tmp_path: Path, tag: str) -> None:
    script = _steps_by_name()["Resolve strict SemVer release tag"]["run"]

    result, outputs = _run_script(
        tmp_path,
        script,
        event_name="workflow_dispatch",
        input_tag=tag,
        existing_tag=tag,
    )

    assert result.returncode == 1
    assert outputs == {}
    assert "::error title=Invalid release tag::" in result.stdout


@BASH_WORKFLOW_ONLY
def test_notify_agent_manual_replay_rejects_unknown_tag(tmp_path: Path) -> None:
    script = _steps_by_name()["Resolve strict SemVer release tag"]["run"]

    result, outputs = _run_script(
        tmp_path,
        script,
        event_name="workflow_dispatch",
        input_tag="v0.6.1",
        existing_tag="v0.6.0",
    )

    assert result.returncode == 1
    assert outputs == {}
    assert "::error title=Unknown release tag::" in result.stdout


@pytest.mark.parametrize(("token", "expected_status"), [("", 1), ("secret-value", 0)])
@BASH_WORKFLOW_ONLY
def test_notify_agent_credential_preflight(tmp_path: Path, token: str, expected_status: int) -> None:
    script = _steps_by_name()["Validate notification credential"]["run"]

    result, outputs = _run_script(tmp_path, script, token=token)

    assert result.returncode == expected_status
    assert outputs == {}
    if token:
        assert token not in result.stdout + result.stderr
    else:
        assert "::error title=Missing WANDAS_AGENT_TOKEN::" in result.stdout
