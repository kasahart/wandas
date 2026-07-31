import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import TypeGuard

import wandas as wd

REPO_ROOT = Path(__file__).resolve().parents[2]
LEARNING_PATH = REPO_ROOT / "learning-path"
APPS = tuple(sorted(LEARNING_PATH.glob("0[0-8]_*.py")))


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _target_names(target: ast.expr) -> set[str]:
    return {node.id for node in ast.walk(target) if isinstance(node, ast.Name)}


def _is_time_axis_name(name: str) -> bool:
    normalized = name.strip("_")
    return normalized == "t" or "time" in normalized.split("_")


def _is_np_linspace(node: ast.AST) -> TypeGuard[ast.Call]:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr == "linspace"
    )


def _linspace_excludes_endpoint(call: ast.Call) -> bool:
    endpoint = next((keyword.value for keyword in call.keywords if keyword.arg == "endpoint"), None)
    if endpoint is None and len(call.args) >= 4:
        endpoint = call.args[3]
    return isinstance(endpoint, ast.Constant) and endpoint.value is False


def _non_exact_linspace_calls(path: Path) -> list[int]:
    tree = ast.parse(_text(path), filename=str(path))
    findings: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = {name for target in node.targets for name in _target_names(target)}
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = _target_names(node.target)
            value = node.value
        else:
            continue
        if not any(_is_time_axis_name(name) for name in targets):
            continue
        findings.extend(
            call.lineno for call in ast.walk(value) if _is_np_linspace(call) and not _linspace_excludes_endpoint(call)
        )
    return findings


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
    assert {path.name: lines for path in APPS if (lines := _non_exact_linspace_calls(path))} == {}


def test_exact_axis_gate_only_checks_time_axis_assignments(tmp_path: Path) -> None:
    valid = tmp_path / "valid.py"
    valid.write_text(
        "import numpy as np\n"
        "frequencies = np.linspace(20, 20_000, 50)\n"
        "time = np.linspace(0, 1, 100, False)\n"
        "sensor_time = np.linspace(0, 1, 100, endpoint=False)\n",
        encoding="utf-8",
    )
    assert _non_exact_linspace_calls(valid) == []

    invalid = tmp_path / "invalid.py"
    invalid.write_text("import numpy as np\n_time = np.linspace(0, 1, 100)\n", encoding="utf-8")
    assert _non_exact_linspace_calls(invalid) == [2]


def test_learning_examples_avoid_compatibility_and_removed_api_spelling() -> None:
    combined = "\n".join(_text(path) for path in APPS)

    assert "wd.read_wav(" not in combined
    assert "ChannelFrame.resample()" not in combined
    assert "ml_results.istft()" not in combined


def test_query_examples_let_query_determine_the_selection() -> None:
    tutorial = _text(REPO_ROOT / "docs/src/tutorial/index.md")

    assert "cf.get_channel" not in tutorial
    assert "get_channel(0, query=" not in tutorial
    assert 'audio.rename_channels({0: "acc_x"})' in tutorial
    assert '.with_calibration({0: wd.ChannelCalibration(unit="g")})' in tutorial
    assert '.with_channel_extra(0, {"gain": 0.8})' in tutorial
    assert 'query_audio.get_channel(query=re.compile(r"acc"))' in tutorial
    assert "query_audio.get_channel(query=lambda ch: ch.unit == 'g')" in tutorial

    query_audio = (
        wd.generate_sin()
        .rename_channels({0: "acc_x"})
        .with_calibration({0: wd.ChannelCalibration(unit="g")})
        .with_channel_extra(0, {"gain": 0.8})
    )
    assert query_audio.get_channel(query=re.compile(r"acc")).labels == ["acc_x"]
    assert query_audio.get_channel(query=lambda channel: channel.unit == "g").labels == ["acc_x"]
    assert query_audio.get_channel(query={"unit": "g", "gain": 0.8}).labels == ["acc_x"]


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
    assert "前半区間の抽出など、sampling rateを変えずに形状が変わる処理" in custom_lesson
    assert "ダウンサンプリングなど形状が変わる処理" not in custom_lesson
