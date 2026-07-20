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
CHANGE_GUIDE_PATH = REPO_ROOT / "docs" / "src" / "contributing" / "change-coherence.md"
CHANGE_SKILL_DIR = SKILLS_DIR / "wandas-change-coherence"
CHANGE_SKILL_PATH = CHANGE_SKILL_DIR / "SKILL.md"
CHANGE_VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_change_coherence.py"
PR_READINESS_SKILL_PATH = SKILLS_DIR / "wandas-pr-readiness" / "SKILL.md"
PR_TEMPLATE_PATH = GITHUB_DIR / "PULL_REQUEST_TEMPLATE.md"
TEST_SKILL_DIR = SKILLS_DIR / "wandas-test-authoring"
TEST_SKILL_PATH = TEST_SKILL_DIR / "SKILL.md"
TEST_REFERENCE_DIR = TEST_SKILL_DIR / "references"
SCALABILITY_SKILL_PATH = SKILLS_DIR / "wandas-scalability-benchmark" / "SKILL.md"
SCALABILITY_ADAPTER_PATH = GITHUB_DIR / "instructions" / "scalability-benchmark.instructions.md"

TEST_ADAPTER_ROUTES = {
    "test-grand-policy.instructions.md": ("tests/**", "grand-policy.md"),
    "test-frames-policy.instructions.md": ("tests/frames/**", "frames.md"),
    "test-processing-policy.instructions.md": ("tests/processing/**", "processing.md"),
    "test-io-policy.instructions.md": ("tests/io/**", "io.md"),
    "test-visualization-policy.instructions.md": ("tests/visualization/**", "visualization.md"),
}

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


def test_test_authoring_skill_has_complete_references_and_codex_metadata() -> None:
    data, body = _frontmatter(TEST_SKILL_PATH)
    assert set(data) == {"name", "description"}
    assert data["name"] == "wandas-test-authoring"
    assert "tests" in str(data["description"]).lower()
    assert body.strip()

    reference_paths = {path.name for path in TEST_REFERENCE_DIR.glob("*.md")}
    assert reference_paths == {
        "grand-policy.md",
        "frames.md",
        "processing.md",
        "io.md",
        "visualization.md",
    }
    skill_targets = set(_local_link_targets(TEST_SKILL_PATH))
    assert {TEST_REFERENCE_DIR / name for name in reference_paths} <= skill_targets

    metadata_path = TEST_SKILL_DIR / "agents" / "openai.yaml"
    metadata = yaml.safe_load(_read(metadata_path))
    assert set(metadata) == {"interface"}
    interface = metadata["interface"]
    assert set(interface) == {"display_name", "short_description", "default_prompt"}
    assert all(isinstance(value, str) and value for value in interface.values())
    assert "$wandas-test-authoring" in interface["default_prompt"]


def test_scalability_benchmark_has_one_skill_owned_route() -> None:
    data, body = _frontmatter(SCALABILITY_SKILL_PATH)
    assert set(data) == {"name", "description"}
    assert data["name"] == "wandas-scalability-benchmark"
    assert "RecipePlan extraction or recipe node counts" in data["description"]
    assert "schema version 2" in body
    assert "--chunk-samples 48000 480000" in body
    assert body.count("uv run --locked") == 2
    assert "uv run --frozen" not in body
    assert "base and candidate commit SHAs" in body
    assert "raw JSON" in body

    metadata = yaml.safe_load(_read(SCALABILITY_SKILL_PATH.parent / "agents" / "openai.yaml"))
    assert set(metadata) == {"interface"}
    assert set(metadata["interface"]) == {"display_name", "short_description", "default_prompt"}
    assert "$wandas-scalability-benchmark" in metadata["interface"]["default_prompt"]

    adapter_data, adapter_body = _frontmatter(SCALABILITY_ADAPTER_PATH)
    assert set(adapter_data) == {"description", "applyTo"}
    apply_to = adapter_data["applyTo"].split(",")
    assert len(apply_to) == len(set(apply_to))
    assert all((REPO_ROOT / path).is_file() for path in apply_to)
    assert {
        "scripts/scalability_benchmark.py",
        "wandas/pipeline/__init__.py",
        "wandas/pipeline/builtins.py",
        "wandas/pipeline/compiler.py",
        "wandas/pipeline/decorators.py",
        "wandas/pipeline/model.py",
        "wandas/pipeline/registry.py",
        "wandas/processing/__init__.py",
        "wandas/processing/semantic.py",
    } <= set(apply_to)
    assert SCALABILITY_SKILL_PATH.resolve() in _local_link_targets(SCALABILITY_ADAPTER_PATH)
    assert SCALABILITY_SKILL_PATH.resolve() in _local_link_targets(CANONICAL_PATH)
    canonical_text = " ".join(_read(CANONICAL_PATH).split())
    trigger_phrases = (
        "WDF save/load",
        "whole-Frame materialization",
        "Dask chunking or graph task counts",
        "RecipePlan extraction or recipe node counts",
        "AudioOperation.process",
        "benchmark semantics",
        "Dask/xarray/HDF5 dependencies",
    )
    assert all(phrase in data["description"] for phrase in trigger_phrases)
    assert all(phrase in canonical_text for phrase in trigger_phrases)

    skill_targets = set(_local_link_targets(SCALABILITY_SKILL_PATH))
    assert {
        CANONICAL_PATH.resolve(),
        (REPO_ROOT / "docs/src/explanation/scalability-contract.md").resolve(),
        (REPO_ROOT / "scripts/scalability_benchmark.py").resolve(),
        (REPO_ROOT / "tests/test_scalability_benchmark.py").resolve(),
    } <= skill_targets

    duplicated_surfaces = [
        GITHUB_DIR / "agents" / "wandas-planner.agent.md",
        GITHUB_DIR / "agents" / "wandas-reviewer.agent.md",
        GITHUB_DIR / "instructions" / "agent-maintenance.instructions.md",
    ]
    assert all("--chunk-samples" not in _read(path) for path in duplicated_surfaces)
    assert not (GITHUB_DIR / "agents" / "wandas-implementer.agent.md").exists()


def test_change_coherence_has_one_canonical_route_and_structural_guardrail() -> None:
    data, body = _frontmatter(CHANGE_SKILL_PATH)
    assert set(data) == {"name", "description"}
    assert data["name"] == "wandas-change-coherence"
    assert body.strip()

    skill_targets = set(_local_link_targets(CHANGE_SKILL_PATH))
    assert skill_targets == {CHANGE_GUIDE_PATH.resolve()}

    guide_data, guide_body = _frontmatter(CHANGE_GUIDE_PATH)
    assert guide_data == {
        "change_coherence": {
            "scenarios": {
                "small_clear": "direct",
                "related_finding": "sibling_search_before_fix",
                "unstable_contract": "contract_replan_required",
            }
        }
    }
    assert guide_body.strip()
    assert not CHANGE_VALIDATOR_PATH.exists()
    assert not list((CHANGE_SKILL_DIR / "references").glob("change-record*"))

    metadata = yaml.safe_load(_read(CHANGE_SKILL_DIR / "agents" / "openai.yaml"))
    assert set(metadata) == {"interface"}
    assert set(metadata["interface"]) == {"display_name", "short_description", "default_prompt"}
    assert "$wandas-change-coherence" in metadata["interface"]["default_prompt"]

    route_paths = [CANONICAL_PATH, PR_READINESS_SKILL_PATH, GITHUB_DIR / "agents" / "wandas-reviewer.agent.md"]
    assert all(CHANGE_SKILL_PATH.resolve() in _local_link_targets(path) for path in route_paths)
    assert CHANGE_GUIDE_PATH.resolve() in _local_link_targets(HARNESS_DOC_PATH)

    vendor_surfaces = [GITHUB_DIR / "agents" / "wandas-reviewer.agent.md", PR_TEMPLATE_PATH]
    for path in vendor_surfaces:
        targets = set(_local_link_targets(path))
        assert CHANGE_SKILL_PATH.resolve() in targets
        assert CHANGE_GUIDE_PATH.resolve() not in targets


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
        PR_TEMPLATE_PATH,
        HARNESS_DOC_PATH,
        CHANGE_GUIDE_PATH,
        REPO_ROOT / "docs" / "src" / "contributing.md",
        REPO_ROOT / "docs" / "src" / "contributing" / "frame-operation-extensions.md",
        REPO_ROOT / "docs" / "src" / "contributing" / "io-contracts.md",
        *sorted((GITHUB_DIR / "agents").glob("*.agent.md")),
        *sorted((GITHUB_DIR / "instructions").glob("*.instructions.md")),
        *sorted(SKILLS_DIR.glob("*/SKILL.md")),
        *sorted(TEST_REFERENCE_DIR.glob("*.md")),
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
        PR_TEMPLATE_PATH,
        *sorted((GITHUB_DIR / "agents").glob("*.agent.md")),
        *sorted((GITHUB_DIR / "instructions").glob("*.instructions.md")),
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


def test_test_policy_adapters_route_without_becoming_a_second_owner() -> None:
    adapter_dir = GITHUB_DIR / "instructions"
    adapter_paths = {path.name: path for path in adapter_dir.glob("test-*.instructions.md")}
    assert set(adapter_paths) == set(TEST_ADAPTER_ROUTES)

    canonical_blocks = {block for path in TEST_REFERENCE_DIR.glob("*.md") for block in _long_prose_blocks(path)}
    for name, (apply_to, reference_name) in TEST_ADAPTER_ROUTES.items():
        path = adapter_paths[name]
        data, body = _frontmatter(path)
        targets = set(_local_link_targets(path))
        assert data["applyTo"] == apply_to
        assert TEST_SKILL_PATH.resolve() in targets
        assert (TEST_REFERENCE_DIR / reference_name).resolve() in targets
        assert "Copilot path-based routing" in body
        assert len(body.split()) < 50
        assert not _long_prose_blocks(path)
        assert canonical_blocks.isdisjoint(_long_prose_blocks(path))


def test_each_agent_entry_reaches_the_test_authoring_skill() -> None:
    assert TEST_SKILL_PATH.resolve() in _local_link_targets(CANONICAL_PATH)

    claude_text = _read(CLAUDE_ADAPTER_PATH)
    assert claude_text.lstrip().startswith("@AGENTS.md")

    test_adapters = sorted((GITHUB_DIR / "instructions").glob("test-*.instructions.md"))
    assert test_adapters
    assert all(TEST_SKILL_PATH.resolve() in _local_link_targets(path) for path in test_adapters)

    harness_text = _read(HARNESS_DOC_PATH)
    assert "test-*.instructions.md" in harness_text
    assert "thin Copilot path adapters" in harness_text
    assert "Deferred migration" not in harness_text


def test_pr_template_captures_reviewable_completion_evidence() -> None:
    text = _read(PR_TEMPLATE_PATH)
    for section in ("## Description", "## Changes", "## PR Readiness", "## Testing"):
        assert section in text
    for concern in ("Closes", "Related", "Follow-up issue", "generated or ignored", "skipped checks"):
        assert concern.lower() in text.lower()
    assert CHANGE_SKILL_PATH.resolve() in _local_link_targets(PR_TEMPLATE_PATH)
    assert "current contract" in text
