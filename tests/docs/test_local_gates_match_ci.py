"""Keep the documented local setup and gate commands in step with CI."""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
RULE_DOC = "docs/src/contributing.md"

DOCUMENTED_FILES = (
    "AGENTS.md",
    "docs/src/contributing.md",
    "docs/src/contributing/frame-operation-extensions.md",
    ".agents/skills/wandas-learning-material-authoring/SKILL.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/workflows/copilot-setup-steps.yml",
    ".github/workflows/claude.yml",
    ".github/workflows/claude-code-review.yml",
    ".pre-commit-config.yaml",
    ".vscode/tasks.json",
    ".devcontainer/devcontainer.json",
)

_SYNC = re.compile(r"uv sync[^\n`\"]*")
_GATE = re.compile(r"(?:uv run (?:--\S+ )*|uvx )(ruff format|ruff check|ty check)\b([^\n`\"]*)")
_VALUE_OPTIONS = {"--config"}


def _ci_steps() -> list[str]:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    return [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if isinstance(step, dict) and "run" in step
    ]


def _ci_extras() -> set[str]:
    extras = {extra for run in _ci_steps() if "uv sync" in run for extra in re.findall(r"--extra (\S+)", run)}
    assert extras, "ci.yml has no `uv sync --extra ...` step; update this test to the new CI setup."
    return extras


def _targets(arguments: str) -> list[str]:
    targets: list[str] = []
    skip_value = False
    for token in arguments.split():
        if skip_value:
            skip_value = False
        elif token in _VALUE_OPTIONS:
            skip_value = True
        elif not token.startswith("-") and token != "&&":
            targets.append(token)
    return targets


def _ci_lint_targets() -> list[str]:
    target_lists = {
        tuple(_targets(match.group(2)))
        for run in _ci_steps()
        for match in _GATE.finditer(run)
        if "--no-sync" in run and _targets(match.group(2))
    }
    assert len(target_lists) == 1, f"ci.yml lint steps disagree on their targets: {sorted(target_lists)}"
    return list(target_lists.pop())


@pytest.mark.parametrize("relative_path", DOCUMENTED_FILES)
def test_documented_setup_installs_the_ci_extras(relative_path: str) -> None:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    required = _ci_extras()

    for command in _SYNC.findall(text):
        if "--all-extras" in command.split():
            continue
        missing = sorted(required - set(re.findall(r"--extra (\S+)", command)))
        assert not missing, (
            f"{relative_path} documents `{command.strip()}`, which leaves out the CI extras {missing}.\n"
            f"Rule: the documented local setup installs the same extras as the CI test sync "
            f"(see {RULE_DOC}, Development Environment Setup).\n"
            f"Fix: add {' '.join(f'--extra {name}' for name in missing)} to that command, "
            f"or update {RULE_DOC} and this file together if CI changed."
        )


@pytest.mark.parametrize("relative_path", DOCUMENTED_FILES)
def test_documented_lint_and_type_targets_match_ci(relative_path: str) -> None:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    expected = _ci_lint_targets()

    for match in _GATE.finditer(text):
        targets = _targets(match.group(2))
        if not targets:
            continue  # bare mentions and pre-commit hooks that receive filenames
        assert targets == expected, (
            f"{relative_path} documents `{match.group(0).strip()}` with targets {targets}, "
            f"but the CI lint job checks {expected}.\n"
            f"Rule: documented ruff format, ruff check and ty check commands use the CI lint targets "
            f"(see {RULE_DOC}, Code Quality Checks, and .github/workflows/ci.yml).\n"
            f"Fix: change the targets in {relative_path} to `{' '.join(expected)}`."
        )
