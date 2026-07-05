from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GITHUB_DIR = REPO_ROOT / ".github"
AGENTS_SKILLS_DIR = REPO_ROOT / ".agents" / "skills"

REPO_SKILL_REQUIRED_CHECKS = {
    "wandas-pr-readiness": [
        "PR title/body full-scope check",
        "Validation evidence",
        "SHA alignment",
        "Finite post-push monitoring",
    ],
    "wandas-issue-triage": [
        "`Closes`",
        "`Related`",
        "Follow-up issue",
    ],
    "wandas-workspace-hygiene": [
        "dirty checkout",
        "ignored/generated artifacts",
        "worktree",
    ],
}


def _read_repo(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


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


def test_pr_lifecycle_harness_guidance_is_linked() -> None:
    """Publisher-facing docs should require PR completion and cleanup gates."""
    agents_text = _read_repo("AGENTS.md")
    copilot_text = _read("copilot-instructions.md")
    publisher_text = _read("agents/wandas-publisher.agent.md")
    maintenance_text = _read("instructions/agent-maintenance.instructions.md")
    lifecycle_text = _read("instructions/pr-lifecycle-harness.instructions.md")
    agents_text_lower = agents_text.lower()

    assert "cross-agent source of truth" in agents_text
    assert "If the current checkout has changes related to the task" in agents_text
    assert "If the relationship is unclear, ask before editing." in agents_text
    assert "PR title/body full-scope check" in agents_text
    assert "validation evidence" in agents_text_lower
    assert "Closes" in agents_text
    assert "Related" in agents_text
    assert "follow-up issue" in agents_text_lower
    assert "generated/ignored artifact cleanup" in agents_text_lower
    assert "local `HEAD`, `origin/<branch>`, and the PR head SHA" in agents_text
    assert "finite post-push monitoring" in agents_text_lower
    assert "no unresolved review threads" in agents_text_lower
    assert "no pending requested reviews" in agents_text_lower
    assert "repeated review feedback" in agents_text_lower
    assert "Codex" in agents_text
    assert ".github/agents/*.agent.md" in agents_text
    assert "runtime procedures" in agents_text
    for instruction_path in sorted((GITHUB_DIR / "instructions").glob("*.instructions.md")):
        assert f"`.github/instructions/{instruction_path.name}`" in agents_text
    assert "PR lifecycle harness" in copilot_text
    assert "pr-lifecycle-harness.instructions.md" in publisher_text
    assert "Copilot-specific adapter" in publisher_text
    assert "not the repository-canonical checklist" in publisher_text
    assert "pr-lifecycle-harness.instructions.md" in maintenance_text
    assert "## PR Completion Gate" in lifecycle_text
    assert "AGENTS.md contains the cross-agent canonical PR lifecycle checklist" in lifecycle_text
    assert "local `HEAD`, `origin/<branch>`, and the PR head SHA" in lifecycle_text
    assert "There are no unresolved actionable review threads." in lifecycle_text
    assert "There are no pending requested reviews." in lifecycle_text
    assert "## Issue Triage Gate" in lifecycle_text
    assert "## Workspace Hygiene Gate" in lifecycle_text
    assert "## Finite Monitoring Gate" in lifecycle_text


def test_repo_skills_are_thin_agents_guidance_adapters() -> None:
    """Repo skills should be validated, discoverable adapters to AGENTS.md."""
    agents_text = _read_repo("AGENTS.md")

    assert "`.agents/skills/`" in agents_text
    assert "`.claude/skills`" in agents_text
    assert "legacy/removed" in agents_text

    for skill_name, required_checks in REPO_SKILL_REQUIRED_CHECKS.items():
        skill_path = AGENTS_SKILLS_DIR / skill_name / "SKILL.md"
        text = skill_path.read_text(encoding="utf-8")

        assert text.startswith("---\n"), skill_path
        _, frontmatter, body = text.split("---", 2)
        data = yaml.safe_load(frontmatter)

        assert isinstance(data, dict), skill_path
        assert data["name"] == skill_name
        assert isinstance(data["description"], str)
        assert data["description"].startswith("Use when ")
        assert "AGENTS.md" in body
        assert "## When to use" in body
        assert "## Required checks" in body
        assert "## Output to report" in body
        assert "## What not to do" in body
        for required_check in required_checks:
            assert required_check in body


def test_pr_template_prompts_for_followup_and_hygiene() -> None:
    """PR authors should record follow-up issues and generated-file cleanup."""
    template_text = (GITHUB_DIR / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")

    assert "Follow-up issue" in template_text
    assert "No generated or ignored work files are left behind" in template_text
    assert "PR title and description describe the full current scope" in template_text
    assert "Closes #<issue-number> only when this PR fully satisfies the issue" in template_text
    assert "Related: #<issue-number> for parent, partial, or follow-up work" in template_text


def test_all_agent_frontmatter_is_valid_yaml() -> None:
    """Every custom agent file should have parseable YAML frontmatter."""
    for path in sorted((GITHUB_DIR / "agents").glob("*.agent.md")):
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        _, frontmatter, _ = text.split("---", 2)
        data = yaml.safe_load(frontmatter)
        assert isinstance(data, dict), path.name
        assert data["name"].startswith("wandas-"), path.name
