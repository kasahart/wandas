from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README_EN = REPO_ROOT / "README.md"
README_JA = REPO_ROOT / "README.ja.md"
DOCS_HOME = REPO_ROOT / "docs" / "src" / "index.md"
EXPLANATION_INDEX = REPO_ROOT / "docs" / "src" / "explanation" / "index.md"
SCALABILITY = REPO_ROOT / "docs" / "src" / "explanation" / "scalability-contract.md"
LEARNING_INTRO = REPO_ROOT / "learning-path" / "00_why_wandas.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_entry_points_route_scalability_questions_to_the_canonical_page() -> None:
    """The contract remains discoverable while Home avoids a second explanation."""
    for path in (README_EN, README_JA):
        text = _read(path)
        assert "scalability-contract.md" in text
    assert "explanation/index.md" in _read(DOCS_HOME)
    assert "scalability-contract.md" in _read(EXPLANATION_INDEX)
    assert "https://kasahart.github.io/wandas/explanation/scalability-contract/" in _read(LEARNING_INTRO)

    contract = _read(SCALABILITY)
    for phrase in ("bounded", "lazy", "material", "NumPy", "tensor"):
        assert phrase.lower() in contract.lower()


def test_home_is_navigation_without_duplicate_scalability_examples() -> None:
    """Home should route readers instead of repeating README/tutorial examples."""
    home = _read(DOCS_HOME)
    assert "```python" not in home
    assert "low_pass_filter" not in home
    assert "assets/images" not in home
