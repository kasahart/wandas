import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_URL = "https://kasahart.github.io/wandas/"


def _collect_nav_paths(nav):
    """Recursively collect path strings from mkdocs `nav` structure."""
    paths = []
    if isinstance(nav, list):
        for item in nav:
            if isinstance(item, str):
                paths.append(item)
            elif isinstance(item, dict):
                for v in item.values():
                    paths.extend(_collect_nav_paths(v))
    elif isinstance(nav, dict):
        for v in nav.values():
            paths.extend(_collect_nav_paths(v))
    elif isinstance(nav, str):
        paths.append(nav)
    return paths


def test_mkdocs_nav_targets_exist():
    """Test that all markdown files referenced in mkdocs.yml nav exist."""
    mk_path = REPO_ROOT / "docs/mkdocs.yml"
    assert mk_path.exists(), "docs/mkdocs.yml must exist"
    raw = mk_path.read_text(encoding="utf-8", errors="replace")
    # Remove python-specific YAML tags like !!python/name:... before parsing
    sanitized = re.sub(r"!!python/name:[^\n]+", "", raw)
    paths = []
    try:
        data = yaml.safe_load(sanitized)
        nav = data.get("nav", [])
        paths = _collect_nav_paths(nav)
    except yaml.YAMLError:
        # Fallback: try extracting nav block via regex (legacy behavior)
        m = re.search(r"^\s*nav:\n((?:\s+.*\n)+)", raw, flags=re.M)
        if not m:
            pytest.skip("No nav block found in docs/mkdocs.yml")
        nav_block = m.group(1)
        paths = re.findall(r"([A-Za-z0-9_\-/]+\.md)", nav_block)

    base = REPO_ROOT / "docs/src"
    assert base.exists(), f"Docs source directory {base} must exist"

    missing = []
    for p in paths:
        # only check markdown targets (skip external URLs)
        if isinstance(p, str) and not p.startswith("http") and p.endswith(".md"):
            target = base / p
            if not target.exists():
                missing.append(str(target))

    if missing:
        pytest.fail(
            f"Missing navigation target files\n"
            f"  Files: {', '.join(missing)}\n"
            f"These markdown files are referenced\n"
            f"  in mkdocs.yml nav but don't exist in docs/src/.\n"
            f"Create the missing files or remove them from the navigation."
        )


def test_index_images_exist():
    """Test that all image files referenced in index.md exist."""
    index = REPO_ROOT / "docs/src/index.md"
    assert index.exists(), "docs/src/index.md must exist"
    text = index.read_text(encoding="utf-8", errors="replace")

    # find markdown image references ![alt](path)
    imgs = re.findall(r"!\[.*?\]\(([^)]+)\)", text)
    base = REPO_ROOT / "docs/src"
    missing = []
    for img in imgs:
        img = img.strip()
        # skip absolute URLs
        if img.startswith("http") or img.startswith("/"):
            continue
        candidate = base / img
        if not candidate.exists():
            missing.append(str(candidate))

    if missing:
        pytest.fail(
            f"Missing image files referenced from index.md\n"
            f"  Files: {', '.join(missing)}\n"
            f"These image files are referenced in docs/src/index.md but don't exist.\n"
            f"Add the missing image files to the appropriate location in docs/src/."
        )


def test_mkdocs_production_metadata_matches_project_site() -> None:
    raw = (REPO_ROOT / "docs/mkdocs.yml").read_text(encoding="utf-8")
    assert f"site_url: {SITE_URL}" in raw
    assert "edit_uri: edit/main/docs/src/" in raw
    assert "content.action.edit" in raw
    assert "G-MEASUREMENT-ID" not in raw
    assert "analytics:" not in raw
    assert "copyright: © 2025–2026 Wandas Team" in raw


def test_learning_path_navigation_targets_deployed_html() -> None:
    lessons = sorted((REPO_ROOT / "learning-path").glob("0[0-8]_*.py"))
    assert len(lessons) == 9

    for index, lesson in enumerate(lessons):
        text = lesson.read_text(encoding="utf-8")
        assert re.search(r"\]\([^)]*\.py(?:[#?][^)]*)?\)", text) is None
        if index:
            assert f"]({lessons[index - 1].stem}.html)" in text
        if index < len(lessons) - 1:
            assert f"]({lessons[index + 1].stem}.html)" in text


def test_docs_learning_links_use_deployment_root() -> None:
    paths = (
        REPO_ROOT / "docs/src/api/core.md",
        REPO_ROOT / "docs/src/api/frames.md",
        REPO_ROOT / "docs/src/tutorial/index.md",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert 'href="../learning-path/' not in text
        for target in re.findall(r'href="([^"]*learning-path/[^"]+)"', text):
            assert target.startswith("/wandas/learning-path/")
