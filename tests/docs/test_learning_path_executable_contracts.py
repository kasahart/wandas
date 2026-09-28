import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import dask
import numpy as np
import pytest
from dask.callbacks import Callback

import wandas as wd
from wandas.pipeline import RecipePlan, RecipeSerializationError, default_recipe_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
LEARNING_PATH = REPO_ROOT / "learning-path"
APPS = tuple(sorted(LEARNING_PATH.glob("[0-9][0-9]_*.py")))


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


def test_learning_apps_run_when_urllib_downloads_are_blocked(tmp_path, monkeypatch) -> None:
    def reject_network(*_args, **_kwargs):
        raise AssertionError("learning apps must use checked-in fixtures, not URL downloads")

    monkeypatch.setattr(urllib.request, "urlopen", reject_network)
    monkeypatch.setattr(urllib.request, "urlretrieve", reject_network)
    monkeypatch.chdir(tmp_path)

    for path in APPS:
        module = _load_app(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _outputs, definitions = module.app.run()
        assert definitions["wd"] is wd

        if path.name == "02_working_with_data.py":
            assert definitions["wav_path"] == LEARNING_PATH / "sample_audio.wav"
            assert definitions["csv_path"] == LEARNING_PATH / "sensor_data.csv"


def test_custom_lesson_replays_its_extension_without_global_registration() -> None:
    module = _load_app(LEARNING_PATH / "05_custom_functions.py")
    _outputs, definitions = module.app.run()
    registry = definitions["extension_registry"]
    payload = json.loads(definitions["extension_json"])
    source = definitions["extension_source"]
    # Use unseen samples so the notebook's materialized results cannot satisfy
    # the execution probe through a cache hit.
    replacement = (
        type(definitions["extension_replacement"])
        .from_numpy(
            np.array([[12, 24, 36], [48, 60, 72]], dtype=np.int16),
            sampling_rate=8000,
            metadata={"recording": "test"},
        )
        .with_source_time_offset(0.5)
    )
    calls = []

    def record_task(key, _graph, _state):
        calls.append(key)

    with dask.config.set(scheduler="synchronous"), Callback(pretask=record_task):
        loaded = RecipePlan.from_dict(payload, registry=registry)
        replayed = loaded.apply({"signal": replacement}, registry=registry)
        assert calls == [], "Loading and replay must build a lazy graph"
        values = replayed.data
    np.testing.assert_allclose(values, [[6.0, 12.0, 18.0], [24.0, 30.0, 36.0]])
    assert calls
    assert replayed.data.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(source.data, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(replacement.data, [[12, 24, 36], [48, 60, 72]])
    assert replayed.metadata == replacement.metadata
    np.testing.assert_array_equal(replayed.source_time_offset, replacement.source_time_offset)
    assert replayed.operation_history[-1]["operation"] == "lesson05.audio.gain"
    with pytest.raises(KeyError):
        default_recipe_registry().require("lesson05.audio.gain", 1)
    with pytest.raises(RecipeSerializationError):
        RecipePlan.from_dict(payload)
    payload["nodes"][0]["version"] = 999
    with pytest.raises(RecipeSerializationError):
        RecipePlan.from_dict(payload, registry=registry)
