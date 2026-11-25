from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_HARNESS_DOC = "docs/src/contributing/agent-harness.md"


def test_agent_harness_stays_small_and_single_owned() -> None:
    claude = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8").strip()
    copilot = (REPO_ROOT / ".github" / "copilot-instructions.md").read_text(encoding="utf-8")

    assert claude == "@AGENTS.md", (
        f"CLAUDE.md contains {claude!r} instead of only '@AGENTS.md'.\n"
        f"Rule: CLAUDE.md only points to AGENTS.md (see {AGENT_HARNESS_DOC}).\n"
        "Fix: move the guidance into AGENTS.md or a developer guide and leave CLAUDE.md as '@AGENTS.md'."
    )
    assert "AGENTS.md" in copilot, (
        ".github/copilot-instructions.md does not mention AGENTS.md.\n"
        f"Rule: .github/copilot-instructions.md only points to AGENTS.md (see {AGENT_HARNESS_DOC}).\n"
        "Fix: make .github/copilot-instructions.md refer to AGENTS.md."
    )

    generic_agents = ("wandas-planner", "wandas-reviewer", "wandas-publisher")
    present = [
        f".github/agents/{name}.agent.md"
        for name in generic_agents
        if (REPO_ROOT / ".github" / "agents" / f"{name}.agent.md").exists()
    ]
    assert not present, (
        f"Generic agent files exist: {present}.\n"
        f"Rule: do not add generic planner, reviewer, or publisher agents (see {AGENT_HARNESS_DOC}).\n"
        "Fix: delete these files and keep the guidance in AGENTS.md or a developer guide."
    )


def test_agent_contract_links_resolve() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    expected = (
        "docs/src/contributing/frame-operation-extensions.md",
        "docs/src/contributing/io-contracts.md",
        "docs/src/explanation/public-api-stability.md",
        ".agents/skills/wandas-scalability-benchmark/SKILL.md",
        ".agents/skills/wandas-learning-material-authoring/SKILL.md",
    )

    missing = [path for path in expected if not (REPO_ROOT / path).is_file()]
    unlinked = [path for path in expected if path not in agents]
    assert not missing, (
        f"AGENTS.md routes to guidance files that do not exist: {missing}.\n"
        f"Rule: AGENTS.md links only to existing specialized guidance (see {AGENT_HARNESS_DOC}).\n"
        "Fix: restore the file, or update the link in AGENTS.md and the expected tuple in this test together."
    )
    assert not unlinked, (
        f"AGENTS.md no longer links these specialized guides: {unlinked}.\n"
        f"Rule: AGENTS.md routes each specialized guide by its repository path (see {AGENT_HARNESS_DOC}).\n"
        "Fix: add the link back to AGENTS.md, or remove the path from the expected tuple in this test on purpose."
    )
