from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_agent_harness_stays_small_and_single_owned() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    claude = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8").strip()
    copilot = (REPO_ROOT / ".github" / "copilot-instructions.md").read_text(encoding="utf-8")

    assert len(agents.splitlines()) <= 30
    assert claude == "@AGENTS.md"
    assert "AGENTS.md" in copilot

    generic_agents = ("wandas-planner", "wandas-reviewer", "wandas-publisher")
    assert all(not (REPO_ROOT / ".github" / "agents" / f"{name}.agent.md").exists() for name in generic_agents)


def test_agent_contract_links_resolve() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    expected = (
        "docs/src/contributing/frame-operation-extensions.md",
        "docs/src/contributing/io-contracts.md",
        ".agents/skills/wandas-scalability-benchmark/SKILL.md",
        ".agents/skills/wandas-learning-material-authoring/SKILL.md",
    )

    assert all((REPO_ROOT / path).is_file() for path in expected)
    assert all(path in agents for path in expected)
