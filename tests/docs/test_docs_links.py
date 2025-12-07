import re
from pathlib import Path

import pytest
import yaml


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
    mk_path = Path("docs/mkdocs.yml")
    assert mk_path.exists(), "docs/mkdocs.yml must exist"
    raw = mk_path.read_text()
    # Remove python-specific YAML tags like !!python/name:... before parsing
    sanitized = re.sub(r"!!python/name:[^\n]+", "", raw)
    paths = []
    try:
        data = yaml.safe_load(sanitized)
        nav = data.get("nav", [])
        paths = _collect_nav_paths(nav)
    except Exception:
        # Fallback: try extracting nav block via regex (legacy behavior)
        m = re.search(r"^\s*nav:\n((?:\s+.*\n)+)", raw, flags=re.M)
        if not m:
            pytest.skip("No nav block found in docs/mkdocs.yml")
        nav_block = m.group(1)
        paths = re.findall(r"([A-Za-z0-9_\-/]+\.md)", nav_block)

    base = Path("docs/src")
    assert base.exists(), f"Docs source directory {base} must exist"

    missing = []
    for p in paths:
        # only check markdown targets (skip external URLs)
        if isinstance(p, str) and not p.startswith("http") and p.endswith(".md"):
            target = base / p
            if not target.exists():
                missing.append(str(target))

    if missing:
        pytest.fail("Missing nav target files: " + ", ".join(missing))


def test_index_images_exist():
    index = Path("docs/src/index.md")
    assert index.exists(), "docs/src/index.md must exist"
    text = index.read_text()

    # find markdown image references ![alt](path)
    imgs = re.findall(r"!\[.*?\]\(([^)]+)\)", text)
    base = Path("docs/src")
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
            "Missing image files referenced from index.md: " + ", ".join(missing)
        )


def test_learning_path_notebooks_exist():
    # tutorial/index.md links to learning-path notebooks by absolute repo path
    lp = Path("learning-path")
    assert lp.exists(), "learning-path directory must exist in repository root"

    expected = [
        "00_why_wandas.ipynb",
        "01_getting_started.ipynb",
        "02_working_with_data.ipynb",
    ]
    missing = [str(lp / e) for e in expected if not (lp / e).exists()]
    if missing:
        pytest.fail("Missing learning-path notebooks: " + ", ".join(missing))
