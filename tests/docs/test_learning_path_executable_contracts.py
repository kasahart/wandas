import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEARNING_PATH = REPO_ROOT / "learning-path"
APPS = tuple(sorted(LEARNING_PATH.glob("0[0-8]_*.py")))


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_all_nine_learning_apps_pass_marimo_check() -> None:
    """CI should catch undefined names and invalid reactive dependencies."""
    assert [path.name[:2] for path in APPS] == [f"{index:02d}" for index in range(9)]
    completed = subprocess.run(
        [sys.executable, "-m", "marimo", "check", *(str(path) for path in APPS)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_learning_apps_match_python_and_exact_time_axis_contracts() -> None:
    sources = {path.name: _text(path) for path in APPS}

    assert all("Python 3.9" not in source for source in sources.values())
    assert "Python 3.10" in sources["01_getting_started.py"]
    assert "Python 3.10+" in _text(LEARNING_PATH / "README.md")
    assert all("np.linspace(" not in source for source in sources.values())


def test_learning_examples_avoid_compatibility_and_removed_api_spelling() -> None:
    combined = "\n".join(_text(path) for path in APPS)

    assert "wd.read_wav(" not in combined
    assert "ChannelFrame.resample()" not in combined
    assert "ml_results.istft()" not in combined


def test_query_examples_let_query_determine_the_selection() -> None:
    tutorial = _text(REPO_ROOT / "docs/src/tutorial/index.md")

    assert "cf.get_channel" not in tutorial
    assert "get_channel(0, query=" not in tutorial
    assert 'audio.get_channel(query=re.compile(r"acc"))' in tutorial
    assert "audio.get_channel(query=lambda ch: ch.unit == 'g')" in tutorial


def test_working_with_data_uses_offline_versioned_fixtures() -> None:
    lesson = _text(LEARNING_PATH / "02_working_with_data.py")

    assert "urllib" not in lesson
    assert "urlretrieve" not in lesson
    assert "refs/heads/main" not in lesson
    assert '__file__).resolve().parent / "sample_audio.wav"' in lesson
    assert '__file__).resolve().parent / "sensor_data.csv"' in lesson
    assert (LEARNING_PATH / "sample_audio.wav").is_file()
    assert (LEARNING_PATH / "sensor_data.csv").is_file()


def test_custom_callable_examples_state_recipe_boundary() -> None:
    introduction = _text(LEARNING_PATH / "00_why_wandas.py")
    custom_lesson = _text(LEARNING_PATH / "05_custom_functions.py")

    assert "lineage=frame.lineage" not in introduction
    assert "wd.SpectrogramFrame.from_numpy(" in introduction
    assert "lineageへ記録されず" in introduction
    assert "runtime-only" in introduction
    assert "RecipeExtractionError" in custom_lesson
    assert "@recipe_operation" in custom_lesson
    assert "runtime-only" in custom_lesson
