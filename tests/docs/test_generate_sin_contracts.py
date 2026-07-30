from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_generate_sin_experimental_status_is_consistent() -> None:
    stability = _read("docs/src/explanation/public-api-stability.md")
    api_overview = _read("docs/src/api/index.md")
    utils_api = _read("docs/src/api/utils.md")
    docs_home = _read("docs/src/index.md")
    readme = _read("README.md")
    readme_ja = _read("README.ja.md")

    assert "Top-level `generate_sin` is a self-contained learning/example helper" in stability
    assert "`wd.generate_sin()` is an experimental top-level learning helper" in api_overview
    assert "This helper remains outside `wandas.__all__`" in utils_api
    assert "`generate_sin_lazy()`" in utils_api
    assert "the low-level implementation name" in " ".join(utils_api.split())
    assert "`wd.generate_sin()` is an experimental helper" in docs_home
    assert "It is an experimental learning helper outside the stable top-level API" in readme
    assert "実験的な学習ヘルパー" in readme_ja


def test_documented_default_and_learning_examples_are_executable_contracts() -> None:
    utils_api = _read("docs/src/api/utils.md")
    learning_intro = _read("learning-path/00_why_wandas.py")
    getting_started = _read("learning-path/01_getting_started.py")

    assert "default_tone = wd.generate_sin()" in utils_api
    assert "`wd.generate_sin()` は、例を自己完結させるための実験的な学習ヘルパー" in learning_intro
    assert "simple_tone = wd.generate_sin()" in getting_started
