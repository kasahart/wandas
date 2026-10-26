from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import wandas as wd
from wandas.pipeline import RecipeExecutionError, RecipePlan, default_recipe_registry

_SR = 8000


def _write_wav(path: Path, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.full(400, value) + np.linspace(0.0, 0.1, 400)
    sf.write(path, samples, _SR, subtype="DOUBLE")
    return path


def _plan() -> RecipePlan:
    source = wd.from_numpy(np.ones((1, 400)), sampling_rate=_SR)
    return RecipePlan.from_frame(source.remove_dc() * 2.0, input_names=("signal",))


def test_apply_files_saves_each_result_and_records_failures(tmp_path: Path) -> None:
    good = _write_wav(tmp_path / "in" / "a.wav", 0.2)
    missing = tmp_path / "in" / "missing.wav"
    other = _write_wav(tmp_path / "in" / "b.wav", 0.4)
    plan = _plan()

    table = plan.apply_files([good, missing, other], tmp_path / "out")

    assert list(table.columns) == ["path", "status", "output", "error", "seconds"]
    assert table["path"].tolist() == [str(good), str(missing), str(other)]
    assert table["status"].tolist() == ["ok", "error", "ok"]
    assert table["output"].tolist() == [str(tmp_path / "out" / "a.wav.wdf"), "", str(tmp_path / "out" / "b.wav.wdf")]
    assert table.loc[1, "error"].startswith("FileNotFoundError: ")
    assert table.loc[0, "error"] == ""
    assert (table["seconds"] >= 0).all()
    for path, output in ((good, table.loc[0, "output"]), (other, table.loc[2, "output"])):
        expected = plan.apply(wd.read(path))
        np.testing.assert_allclose(wd.load(output).data, expected.data)


def test_apply_files_keeps_folders_below_root(tmp_path: Path) -> None:
    first = _write_wav(tmp_path / "in" / "L1" / "a.wav", 0.1)
    second = _write_wav(tmp_path / "in" / "L2" / "a.wav", 0.2)

    table = _plan().apply_files([first, second], tmp_path / "out", root=tmp_path / "in")

    assert table["output"].tolist() == [
        str(tmp_path / "out" / "L1" / "a.wav.wdf"),
        str(tmp_path / "out" / "L2" / "a.wav.wdf"),
    ]
    assert table["status"].tolist() == ["ok", "ok"]


def test_apply_files_rejects_outputs_that_would_collide(tmp_path: Path) -> None:
    first = _write_wav(tmp_path / "L1" / "a.wav", 0.1)
    second = _write_wav(tmp_path / "L2" / "a.wav", 0.2)

    with pytest.raises(ValueError, match="same output; pass root"):
        _plan().apply_files([first, second], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_apply_files_rejects_path_outside_root(tmp_path: Path) -> None:
    outside = _write_wav(tmp_path / "other" / "a.wav", 0.1)

    with pytest.raises(ValueError, match="not inside root"):
        _plan().apply_files([outside], tmp_path / "out", root=tmp_path / "in")


def test_apply_files_raise_stops_at_first_failure(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"
    later = _write_wav(tmp_path / "later.wav", 0.1)

    with pytest.raises(FileNotFoundError):
        _plan().apply_files([missing, later], tmp_path / "out", on_error="raise")
    assert not (tmp_path / "out" / "later.wav.wdf").exists()


def test_apply_files_existing_output_is_a_failure_unless_overwrite(tmp_path: Path) -> None:
    recording = _write_wav(tmp_path / "a.wav", 0.1)
    plan = _plan()
    plan.apply_files([recording], tmp_path / "out")

    again = plan.apply_files([recording], tmp_path / "out")
    replaced = plan.apply_files([recording], tmp_path / "out", overwrite=True)

    assert again.loc[0, "status"] == "error"
    assert again.loc[0, "error"].startswith("FileExistsError: ")
    assert replaced.loc[0, "status"] == "ok"


def test_apply_files_with_processes_matches_serial(tmp_path: Path) -> None:
    recordings = [_write_wav(tmp_path / "in" / f"{index}.wav", index / 10) for index in range(3)]
    missing = tmp_path / "in" / "missing.wav"
    plan = _plan()

    serial = plan.apply_files([*recordings, missing], tmp_path / "serial")
    parallel = plan.apply_files([*recordings, missing], tmp_path / "parallel", jobs=2)

    assert parallel["status"].tolist() == serial["status"].tolist() == ["ok", "ok", "ok", "error"]
    assert parallel["error"].tolist() == serial["error"].tolist()
    for serial_output, parallel_output in zip(serial["output"][:3], parallel["output"][:3]):
        np.testing.assert_array_equal(wd.load(parallel_output).data, wd.load(serial_output).data)


def test_apply_files_with_processes_raises_first_failure(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _plan().apply_files([tmp_path / "missing.wav"], tmp_path / "out", jobs=2, on_error="raise")


def test_apply_files_requires_one_input(tmp_path: Path) -> None:
    source = wd.from_numpy(np.ones((1, 400)), sampling_rate=_SR)
    plan = RecipePlan.from_frame(source + np.ones(400), input_names=("signal", "offset"))

    with pytest.raises(RecipeExecutionError, match="requires a Recipe with one input"):
        plan.apply_files([], tmp_path / "out")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"jobs": 0}, "jobs must be a positive integer"),
        ({"on_error": "skip"}, "on_error must be"),
        ({"jobs": 2, "registry": default_recipe_registry()}, "custom registry cannot be sent"),
    ],
)
def test_apply_files_rejects_invalid_options(tmp_path: Path, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _plan().apply_files([], tmp_path / "out", **kwargs)


def test_apply_files_with_no_paths_returns_empty_table(tmp_path: Path) -> None:
    table = _plan().apply_files([], tmp_path / "out")

    assert table.empty
    assert list(table.columns) == ["path", "status", "output", "error", "seconds"]


def test_apply_files_error_names_failed_operation_and_cause(tmp_path: Path) -> None:
    source = wd.from_numpy(np.ones((1, 400)), sampling_rate=_SR)
    plan = RecipePlan.from_frame(source.low_pass_filter(cutoff=1000), input_names=("signal",))
    low_rate = tmp_path / "low.wav"
    sf.write(low_rate, np.zeros(400), 1000, subtype="DOUBLE")

    table = plan.apply_files([low_rate], tmp_path / "out")

    assert table.loc[0, "status"] == "error"
    assert table.loc[0, "error"] == "wandas.audio.lowpass_filter: ValueError: Cutoff frequency out of valid range"
