from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GITHUB_DIR = REPO_ROOT / ".github"


def _read(relative_path: str) -> str:
    return (GITHUB_DIR / relative_path).read_text(encoding="utf-8")


def test_testing_workflow_instructions_require_coverage_review() -> None:
    """Testing workflow guidance should make coverage regressions explicit."""
    text = _read("instructions/testing-workflow.instructions.md")

    assert "Treat `Run pytest` coverage output as part of the definition of done" in text
    assert "--cov=wandas --cov-report=term-missing" in text
    assert "do not hand off silent coverage regressions" in text


def test_test_grand_policy_mentions_coverage_aware_testing() -> None:
    """Grand test policy should describe how to react to coverage gaps."""
    text = _read("instructions/test-grand-policy.instructions.md")

    assert "## Coverage-Aware Testing" in text
    assert "Treat coverage regressions on changed code as a warning" in text
    assert "validation and error branches" in text


def test_agent_docs_reference_coverage_guidance() -> None:
    """Agent-facing docs should point implementers to coverage-aware validation."""
    implementer_text = _read("agents/wandas-implementer.agent.md")
    copilot_text = _read("copilot-instructions.md")

    assert "--cov=wandas --cov-report=term-missing" in implementer_text
    assert "silent coverage regressions on touched code" in implementer_text
    assert "Coverage Analysis" in implementer_text
    assert "coverage-aware testing guidance for touched code paths" in copilot_text


def test_agent_docs_keep_planner_first_workflow() -> None:
    """Agent docs should consistently describe planner-first workflow."""
    copilot_text = _read("copilot-instructions.md")
    implementer_text = _read("agents/wandas-implementer.agent.md")
    reviewer_text = _read("agents/wandas-reviewer.agent.md")
    publisher_text = _read("agents/wandas-publisher.agent.md")
    maintenance_text = _read("instructions/agent-maintenance.instructions.md")

    assert "default workflow is `wandas-planner` -> `wandas-implementer` -> `wandas-reviewer`" in copilot_text
    assert "Prefer `wandas-planner` first for new substantive work." in implementer_text
    assert "hand off to the planner with the next task." in reviewer_text
    assert "hand off to the planner for follow-up work if additional tasks remain." in publisher_text
    assert (
        "**Who**: Use the full `wandas-planner` -> `wandas-implementer` -> `wandas-reviewer` flow" in maintenance_text
    )


def test_all_agent_frontmatter_is_valid_yaml() -> None:
    """Every custom agent file should have parseable YAML frontmatter."""
    for path in sorted((GITHUB_DIR / "agents").glob("*.agent.md")):
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        _, frontmatter, _ = text.split("---", 2)
        data = yaml.safe_load(frontmatter)
        assert isinstance(data, dict), path.name
        assert data["name"].startswith("wandas-"), path.name
