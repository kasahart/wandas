from pathlib import Path

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


def test_agent_docs_define_fallback_when_planner_is_unavailable() -> None:
    """Agent docs should stay usable when the planner is not exposed in a runtime."""
    copilot_text = _read("copilot-instructions.md")
    implementer_text = _read("agents/wandas-implementer.agent.md")
    reviewer_text = _read("agents/wandas-reviewer.agent.md")
    publisher_text = _read("agents/wandas-publisher.agent.md")
    maintenance_text = _read("instructions/agent-maintenance.instructions.md")

    assert "If `wandas-planner` is not exposed in the current runtime" in copilot_text
    assert "planner is unavailable in the current runtime" in implementer_text
    assert "agent: wandas-planner" not in reviewer_text
    assert "agent: wandas-planner" not in publisher_text
    assert "planner-capable runtime or user guidance is required" in reviewer_text
    assert "planner-capable runtime is available" in publisher_text
    assert "Runtime Availability Rule" in maintenance_text
