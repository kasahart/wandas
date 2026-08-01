import contextlib
import importlib.util
import io
import subprocess
import sys
import urllib.request
from pathlib import Path

import wandas as wd

REPO_ROOT = Path(__file__).resolve().parents[2]
LEARNING_PATH = REPO_ROOT / "learning-path"
APPS = tuple(sorted(LEARNING_PATH.glob("[0-9][0-9]_*.py")))
OFFLINE_APPS = tuple(path for path in APPS if path.name != "06_skill_validation.py")


def _load_app(path: Path):
    spec = importlib.util.spec_from_file_location(f"_learning_path_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_learning_apps_pass_marimo_check() -> None:
    """CI should catch undefined names and invalid reactive dependencies."""
    completed = subprocess.run(
        [sys.executable, "-m", "marimo", "check", "--strict", *(str(path) for path in APPS)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_learning_apps_execute_offline_with_checked_in_fixtures(tmp_path, monkeypatch) -> None:
    def reject_network(*_args, **_kwargs):
        raise AssertionError("learning apps must use checked-in fixtures, not URL downloads")

    monkeypatch.setattr(urllib.request, "urlopen", reject_network)
    monkeypatch.setattr(urllib.request, "urlretrieve", reject_network)
    monkeypatch.chdir(tmp_path)

    for path in OFFLINE_APPS:
        module = _load_app(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _outputs, definitions = module.app.run()
        assert definitions["wd"] is wd

        if path.name == "02_working_with_data.py":
            assert definitions["wav_path"] == LEARNING_PATH / "sample_audio.wav"
            assert definitions["csv_path"] == LEARNING_PATH / "sensor_data.csv"
