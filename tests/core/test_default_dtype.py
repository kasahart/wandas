from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import wandas as wd
from tests.frame_helpers import channel_first_values
from wandas.io.readers import CSVFileReader


@pytest.fixture(autouse=True)
def _restore_default() -> Iterator[None]:
    previous = wd.get_default_dtype()
    yield
    wd.set_default_dtype(previous)


def _wav(path: Path, subtype: str = "PCM_24") -> Path:
    samples = np.random.default_rng(0).uniform(-0.9, 0.9, (2_000, 2))
    sf.write(path, samples, 8_000, subtype=subtype)
    return path


def test_default_is_float64() -> None:
    assert wd.get_default_dtype() == np.dtype("float64")


@pytest.mark.parametrize("dtype", ["float32", np.float32, np.dtype("float32")])
def test_set_default_dtype_accepts_float32_spellings(dtype: Any) -> None:
    wd.set_default_dtype(dtype)

    assert wd.get_default_dtype() == np.dtype("float32")


@pytest.mark.parametrize("dtype", [None, "float16", "int16", "complex64", "not-a-dtype"])
def test_set_default_dtype_rejects_other_dtypes(dtype: Any) -> None:
    with pytest.raises(ValueError, match="must be float32 or float64"):
        wd.set_default_dtype(dtype)
    assert wd.get_default_dtype() == np.dtype("float64")


def test_default_dtype_context_restores_previous_value_after_error() -> None:
    with pytest.raises(RuntimeError), wd.default_dtype("float32"):
        assert wd.get_default_dtype() == np.dtype("float32")
        raise RuntimeError("inside")

    assert wd.get_default_dtype() == np.dtype("float64")


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
def test_read_decodes_audio_as_float32_matching_soundfile(tmp_path: Path, subtype: str) -> None:
    path = _wav(tmp_path / f"{subtype}.wav", subtype)
    expected, _ = sf.read(path, dtype="float32", always_2d=True)

    frame = wd.read(path, dtype="float32")

    assert frame._data.dtype == np.dtype("float32")
    values = channel_first_values(frame)
    assert values.dtype == np.dtype("float32")
    np.testing.assert_array_equal(values, expected.T)


def test_read_uses_default_dtype_at_call_time(tmp_path: Path) -> None:
    path = _wav(tmp_path / "a.wav")

    with wd.default_dtype("float32"):
        inside = wd.read(path)
    after = wd.read(path)

    # The dtype is fixed when the frame is created, not when it is computed.
    assert channel_first_values(inside).dtype == np.dtype("float32")
    assert channel_first_values(after).dtype == np.dtype("float64")


def test_explicit_dtype_overrides_default(tmp_path: Path) -> None:
    path = _wav(tmp_path / "a.wav")
    wd.set_default_dtype("float32")

    assert channel_first_values(wd.read(path, dtype="float64")).dtype == np.dtype("float64")


def test_read_rejects_unsupported_dtype(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be float32 or float64"):
        wd.read(_wav(tmp_path / "a.wav"), dtype="int16")


def test_read_float32_from_bytes_and_channel_selection(tmp_path: Path) -> None:
    content = _wav(tmp_path / "a.wav").read_bytes()
    expected, _ = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)

    frame = wd.read(io.BytesIO(content), channel=1, start=0.05, end=0.1, dtype="float32")

    np.testing.assert_array_equal(channel_first_values(frame), expected[400:800, 1:].T)


def test_csv_is_converted_to_the_selected_dtype(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    pd.DataFrame({"time": [0.0, 0.5, 1.0], "value": [1.25, -3.5, 0.1]}).to_csv(path, index=False)

    frame = wd.read(path, dtype="float32")

    np.testing.assert_array_equal(channel_first_values(frame), np.array([[1.25, -3.5, 0.1]], dtype=np.float32))


def test_non_soundfile_readers_are_not_passed_dtype(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Only the SoundFile reader is asked for a dtype; other readers keep their contract.
    path = tmp_path / "a.csv"
    pd.DataFrame({"time": [0.0, 0.5, 1.0], "value": [1.25, -3.5, 0.1]}).to_csv(path, index=False)
    received: list[dict[str, Any]] = []
    original = CSVFileReader.get_data.__func__

    def recording_get_data(cls: type[CSVFileReader], *args: Any, **kwargs: Any) -> Any:
        received.append(kwargs)
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(CSVFileReader, "get_data", classmethod(recording_get_data))

    channel_first_values(wd.read(path, dtype="float32"))

    assert received
    assert all("dtype" not in kwargs for kwargs in received)


def test_dataset_loader_uses_default_dtype(tmp_path: Path) -> None:
    _wav(tmp_path / "a.wav")
    wd.set_default_dtype("float32")

    dataset = wd.from_folder(str(tmp_path))

    frame = dataset[0]
    assert frame is not None
    assert channel_first_values(frame).dtype == np.dtype("float32")


def test_dataset_keeps_the_dtype_in_effect_when_created(tmp_path: Path) -> None:
    _wav(tmp_path / "a.wav")
    _wav(tmp_path / "b.wav")

    with wd.default_dtype("float32"):
        dataset = wd.from_folder(str(tmp_path))
    derived = dataset.apply(lambda frame: frame)
    subset = dataset.sample(n=1, seed=0)

    for frame in (dataset[0], dataset[1], derived[0], subset[0]):
        assert frame is not None
        assert channel_first_values(frame).dtype == np.dtype("float32")


@pytest.mark.parametrize("loader", ["from_folder", "from_files", "from_table"])
def test_dataset_loaders_accept_dtype(tmp_path: Path, loader: str) -> None:
    path = _wav(tmp_path / "a.wav")
    if loader == "from_folder":
        dataset = wd.from_folder(str(tmp_path), dtype="float32")
    elif loader == "from_files":
        dataset = wd.from_files([path], dtype="float32")
    else:
        dataset = wd.from_table(pd.DataFrame({"audio": [str(path)]}), path_column="audio", dtype="float32")

    frame = dataset[0]
    assert frame is not None
    assert channel_first_values(frame).dtype == np.dtype("float32")


def test_dataset_rejects_unsupported_dtype(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="float32 or float64"):
        wd.from_folder(str(tmp_path), dtype="int16")


def test_generate_sin_follows_default_and_explicit_dtype() -> None:
    reference = channel_first_values(wd.generate_sin(freqs=[440.0, 1000.0], duration=0.01))

    with wd.default_dtype("float32"):
        by_default = channel_first_values(wd.generate_sin(freqs=[440.0, 1000.0], duration=0.01))
    explicit = channel_first_values(wd.generate_sin(freqs=[440.0, 1000.0], duration=0.01, dtype="float32"))

    assert reference.dtype == np.dtype("float64")
    assert by_default.dtype == explicit.dtype == np.dtype("float32")
    np.testing.assert_array_equal(by_default, reference.astype(np.float32))
