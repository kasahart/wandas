import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GITHUB_DIR = REPO_ROOT / ".github"
SKILLS_DIR = REPO_ROOT / ".agents" / "skills"

CANONICAL_PATH = REPO_ROOT / "AGENTS.md"
CLAUDE_ADAPTER_PATH = REPO_ROOT / "CLAUDE.md"
COPILOT_ADAPTER_PATH = GITHUB_DIR / "copilot-instructions.md"
HARNESS_DOC_PATH = REPO_ROOT / "docs" / "src" / "contributing" / "agent-harness.md"

LOCAL_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = _read(path)
    assert text.startswith("---\n"), path
    _, raw_frontmatter, body = text.split("---", 2)
    data = yaml.safe_load(raw_frontmatter)
    assert isinstance(data, dict), path
    return data, body


def _local_link_targets(path: Path) -> list[Path]:
    targets: list[Path] = []
    for raw_target in LOCAL_LINK_RE.findall(_read(path)):
        target = raw_target.split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = unquote(target.split("#", 1)[0])
        targets.append((path.parent / target).resolve())
    return targets


def _long_prose_blocks(path: Path) -> set[str]:
    text = _read(path)
    if text.startswith("---\n"):
        _, _, text = text.split("---", 2)
    blocks = set()
    for block in re.split(r"\n\s*\n", text):
        normalized = " ".join(block.split())
        if len(normalized) >= 180:
            blocks.add(normalized)
    return blocks


def test_canonical_contract_keeps_cross_agent_invariants_and_routes() -> None:
    text = _read(CANONICAL_PATH)
    lower_text = text.lower()

    semantic_markers = (
        "`uv`",
        "git status --short",
        ".worktrees/",
        "unrelated user changes",
        "frame immutability",
        "metadata and lineage",
        "dask laziness",
        "operation_history",
        "wandas/frames",
        "wandas/processing",
        "uv run pytest",
        "uv run ruff check",
        "uv run ty check",
    )
    for marker in semantic_markers:
        assert marker in lower_text

    for vendor_marker in ("github copilot", "claude code", "codex", ".github/", ".claude/"):
        assert vendor_marker not in lower_text

    skill_targets = {
        path.relative_to(REPO_ROOT) for path in _local_link_targets(CANONICAL_PATH) if SKILLS_DIR in path.parents
    }
    expected_skill_targets = {path.relative_to(REPO_ROOT) for path in SKILLS_DIR.glob("*/SKILL.md")}
    assert skill_targets == expected_skill_targets
    assert HARNESS_DOC_PATH.resolve() in _local_link_targets(CANONICAL_PATH)


def test_each_tool_has_a_route_to_the_canonical_contract() -> None:
    assert CANONICAL_PATH.is_file()  # Codex loads this repository-root file.

    claude_lines = [line.strip() for line in _read(CLAUDE_ADAPTER_PATH).splitlines() if line.strip()]
    assert claude_lines[0] == "@AGENTS.md"
    assert ".agents/skills" in _read(CLAUDE_ADAPTER_PATH)
    assert ".claude/skills" in _read(CLAUDE_ADAPTER_PATH)

    assert CANONICAL_PATH.resolve() in _local_link_targets(COPILOT_ADAPTER_PATH)
    copilot_text = _read(COPILOT_ADAPTER_PATH)
    normalized_copilot_text = " ".join(copilot_text.split())
    assert ".agents/skills" in copilot_text
    assert "may be handled directly" in copilot_text
    assert "not a required" in normalized_copilot_text

    assert "Do not copy its body" in _read(CLAUDE_ADAPTER_PATH)


def test_repo_skills_are_discoverable_and_have_valid_metadata() -> None:
    skill_paths = sorted(SKILLS_DIR.glob("*/SKILL.md"))
    assert skill_paths

    names: set[str] = set()
    for path in skill_paths:
        data, body = _frontmatter(path)
        assert data["name"] == path.parent.name
        assert isinstance(data["description"], str)
        assert data["description"].strip()
        assert body.strip()
        names.add(str(data["name"]))

        metadata_path = path.parent / "agents" / "openai.yaml"
        if metadata_path.exists():
            metadata = yaml.safe_load(_read(metadata_path))
            assert isinstance(metadata, dict), metadata_path
            assert isinstance(metadata.get("interface"), dict), metadata_path

    assert len(names) == len(skill_paths)

    frame_skill = SKILLS_DIR / "wandas-frame-operation-extension" / "SKILL.md"
    frame_guide = REPO_ROOT / "docs" / "src" / "contributing" / "frame-operation-extensions.md"
    assert frame_guide.resolve() in _local_link_targets(frame_skill)


def test_copilot_frontmatter_and_capability_boundaries_are_valid() -> None:
    instruction_paths = sorted((GITHUB_DIR / "instructions").glob("*.instructions.md"))
    assert instruction_paths
    for path in instruction_paths:
        data, body = _frontmatter(path)
        assert isinstance(data.get("description"), str), path
        assert isinstance(data.get("applyTo"), str), path
        assert body.strip(), path

    agent_paths = sorted((GITHUB_DIR / "agents").glob("*.agent.md"))
    agent_data = {path.stem.removesuffix(".agent"): _frontmatter(path) for path in agent_paths}
    assert set(agent_data) == {"wandas-planner", "wandas-publisher", "wandas-reviewer"}

    planner, planner_body = agent_data["wandas-planner"]
    reviewer, reviewer_body = agent_data["wandas-reviewer"]
    publisher, publisher_body = agent_data["wandas-publisher"]

    assert "edit" not in planner["tools"]
    assert "execute" not in planner["tools"]
    assert "edit" not in reviewer["tools"]
    assert publisher["disable-model-invocation"] is True
    assert "execute" in publisher["tools"]
    assert "explicit" in publisher_body.lower()

    for body in (planner_body, reviewer_body, publisher_body):
        assert "AGENTS.md" in body
    assert "wandas-implementer" not in _read(COPILOT_ADAPTER_PATH)
    assert not (GITHUB_DIR / "agents" / "wandas-implementer.agent.md").exists()


def test_harness_local_links_resolve() -> None:
    harness_paths = [
        CANONICAL_PATH,
        CLAUDE_ADAPTER_PATH,
        COPILOT_ADAPTER_PATH,
        HARNESS_DOC_PATH,
        REPO_ROOT / "docs" / "src" / "contributing.md",
        REPO_ROOT / "docs" / "src" / "contributing" / "frame-operation-extensions.md",
        REPO_ROOT / "docs" / "src" / "contributing" / "io-contracts.md",
        *sorted((GITHUB_DIR / "agents").glob("*.agent.md")),
        *sorted((GITHUB_DIR / "instructions").glob("*.instructions.md")),
        *sorted(SKILLS_DIR.glob("*/SKILL.md")),
    ]

    broken = [
        (path.relative_to(REPO_ROOT), target)
        for path in harness_paths
        for target in _local_link_targets(path)
        if not target.exists()
    ]
    assert not broken


def test_adapters_do_not_duplicate_long_repository_rules() -> None:
    adapter_paths = [
        CLAUDE_ADAPTER_PATH,
        COPILOT_ADAPTER_PATH,
        *sorted((GITHUB_DIR / "agents").glob("*.agent.md")),
        *sorted(
            path
            for path in (GITHUB_DIR / "instructions").glob("*.instructions.md")
            if not path.name.startswith("test-")
        ),
    ]
    owners: defaultdict[str, list[Path]] = defaultdict(list)
    for path in [CANONICAL_PATH, *adapter_paths]:
        for block in _long_prose_blocks(path):
            owners[block].append(path.relative_to(REPO_ROOT))

    duplicates = {block: paths for block, paths in owners.items() if len(paths) > 1}
    assert not duplicates

    canonical_blocks = _long_prose_blocks(CANONICAL_PATH)
    for path in adapter_paths:
        assert canonical_blocks.isdisjoint(_long_prose_blocks(path)), path


def test_transitional_test_policies_remain_scoped_and_documented() -> None:
    policy_paths = sorted((GITHUB_DIR / "instructions").glob("test-*.instructions.md"))
    assert policy_paths
    for path in policy_paths:
        data, _ = _frontmatter(path)
        assert str(data["applyTo"]).startswith("tests/"), path

    harness_text = _read(HARNESS_DOC_PATH)
    assert ".github/instructions/test-*.instructions.md" in harness_text
    assert "follow-up" in harness_text.lower()


def test_pr_template_captures_reviewable_completion_evidence() -> None:
    text = _read(GITHUB_DIR / "PULL_REQUEST_TEMPLATE.md")
    for section in ("## Description", "## Changes", "## PR Readiness", "## Testing"):
        assert section in text
    for concern in ("Closes", "Related", "Follow-up issue", "generated or ignored", "skipped checks"):
        assert concern.lower() in text.lower()
