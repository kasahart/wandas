"""Platform paths, extended scalar metadata and signal-CSV source contracts."""

import os
from pathlib import Path, PureWindowsPath
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import wandas as wd
import wandas.frames._dataset_inputs as inputs
from wandas.io.readers import CSVFileReader, SoundFileReader


def test_drive_paths_are_not_uri_schemes(monkeypatch) -> None:
    # Exercise native Windows parsing on every platform without filesystem access.
    monkeypatch.setattr(inputs, "Path", PureWindowsPath)
    base = cast(Path, PureWindowsPath("D:/catalog"))
    assert str(inputs._local_path("C://data/a.wav", base, location="item")) == r"C:\data\a.wav"
    with pytest.raises(ValueError, match="local paths"):
        inputs._local_path("https://example.com/a.wav", base, location="item")


def test_other_drive_relative_path_uses_drive_working_directory(monkeypatch) -> None:
    monkeypatch.setattr(inputs, "Path", PureWindowsPath)
    with patch.object(inputs.os.path, "abspath", return_value="C:/previous") as absolute:
        path = inputs._local_path("C:audio.wav", cast(Path, PureWindowsPath("D:/catalog")), location="item")
    assert str(path) == r"C:\previous\audio.wav"
    absolute.assert_called_once_with("C:.")


@pytest.mark.skipif(os.name != "nt", reason="native Windows drive semantics")
@pytest.mark.parametrize("entry", ["files", "dataframe", "catalog"])
def test_public_inputs_accept_drive_qualified_double_slash(tmp_path, entry) -> None:
    source = "C://data/a.wav"
    if entry == "files":
        dataset = wd.from_files([source])
    elif entry == "dataframe":
        dataset = wd.from_table(pd.DataFrame({"audio": [source]}), path_column="audio")
    else:
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(f"audio\n{source}\n")
        dataset = wd.from_table(catalog, path_column="audio")
    assert str(dataset._lazy_frames[0].file_path) == r"C:\data\a.wav"


@pytest.mark.parametrize("entry", ["files", "dataframe", "catalog"])
@pytest.mark.parametrize("base_contains_parent", [False, True])
def test_symlink_parent_segments_keep_filesystem_meaning(tmp_path, monkeypatch, entry, base_contains_parent) -> None:
    catalog_dir, data_dir = tmp_path / "catalog", tmp_path / "data"
    catalog_dir.mkdir()
    (data_dir / "run").mkdir(parents=True)
    try:
        (catalog_dir / "link").symlink_to(data_dir / "run", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks unavailable: {exc}")
    sf.write(data_dir / "audio.wav", np.full(80, 0.25), 8000, subtype="FLOAT")
    sf.write(catalog_dir / "audio.wav", np.full(80, -0.5), 8000, subtype="FLOAT")
    monkeypatch.chdir(tmp_path)
    base = "catalog/link/.." if base_contains_parent else "catalog"
    relative = "audio.wav" if base_contains_parent else "link/../audio.wav"
    with (
        patch.object(Path, "resolve", side_effect=AssertionError("eager filesystem resolution")),
        patch.object(SoundFileReader, "get_file_info", wraps=SoundFileReader.get_file_info) as headers,
    ):
        if entry == "files":
            dataset = wd.from_files([relative], base_dir=base)
        elif entry == "dataframe":
            dataset = wd.from_table(pd.DataFrame({"audio": [relative]}), path_column="audio", base_dir=base)
        else:
            catalog = catalog_dir / "index.csv"
            catalog.write_text(f"audio\n{relative}\n")
            dataset = wd.from_table(catalog, path_column="audio", base_dir=base)
        assert headers.call_count == 0
        assert dataset._lazy_frames[0].file_path.is_absolute()
    monkeypatch.chdir(data_dir / "run")
    frame = dataset[0]
    assert frame is not None
    np.testing.assert_allclose(frame.to_numpy(), 0.25)


@pytest.mark.parametrize("scalar", [np.float16(0.125), np.float32(0.125), np.float64(0.125), np.longdouble("0.125")])
def test_numpy_real_metadata_becomes_python_float(scalar) -> None:
    table = pd.DataFrame({"audio": ["missing.wav"], "value": pd.Series([scalar], dtype=object)})
    dataset = wd.from_table(table, path_column="audio")
    value = dataset._lazy_frames[0].metadata["value"]
    assert type(value) is float and value == 0.125


def test_extended_float_missing_and_infinity_follow_scalar_contract() -> None:
    table = pd.DataFrame({"audio": ["missing.wav"], "value": pd.Series([np.longdouble("nan")], dtype=object)})
    assert wd.from_table(table, path_column="audio")._lazy_frames[0].metadata["value"] is None
    table.loc[0, "value"] = np.longdouble("inf")
    with pytest.raises(ValueError, match="row 0.value"):
        wd.from_table(table, path_column="audio")


def test_finite_extended_float_outside_python_range_requires_conversion() -> None:
    value = np.finfo(np.longdouble).max
    if value <= np.finfo(float).max:
        pytest.skip("longdouble range does not exceed Python float")
    table = pd.DataFrame({"audio": ["missing.wav"], "value": pd.Series([value], dtype=object)})
    with pytest.raises(ValueError, match="row 0.value"):
        wd.from_table(table, path_column="audio")


@pytest.mark.parametrize("entry", ["files", "dataframe", "catalog"])
def test_signal_csv_inspection_is_eager_but_sample_graph_remains_lazy(tmp_path, entry) -> None:
    signal = tmp_path / "signal.csv"
    signal.write_text("time,value\n0,0.1\n0.001,0.2\n0.002,0.3\n")
    with (
        patch.object(pd, "read_csv", wraps=pd.read_csv) as parses,
        patch.object(CSVFileReader, "get_data", wraps=CSVFileReader.get_data) as samples,
    ):
        if entry == "files":
            dataset = wd.from_files([signal])
        elif entry == "dataframe":
            dataset = wd.from_table(pd.DataFrame({"audio": [signal]}), path_column="audio")
        else:
            catalog = tmp_path / "catalog.csv"
            catalog.write_text("audio\nsignal.csv\n")
            dataset = wd.from_table(catalog, path_column="audio")
        assert parses.call_count == samples.call_count == 0
        frame = dataset[0]
        assert frame is not None and parses.call_count == 1 and samples.call_count == 0
        assert dataset[0] is frame and parses.call_count == 1
        np.testing.assert_allclose(frame.to_numpy(), [0.1, 0.2, 0.3])
        assert parses.call_count == 2 and samples.call_count == 1


@pytest.mark.parametrize("entry", ["files", "dataframe"])
def test_malformed_signal_csv_fails_and_caches_on_item_access(tmp_path, entry) -> None:
    signal = tmp_path / "signal.csv"
    signal.write_text('time,value\n"unterminated\n')
    dataset = (
        wd.from_files([signal])
        if entry == "files"
        else wd.from_table(pd.DataFrame({"audio": [signal]}), path_column="audio")
    )
    with patch.object(pd, "read_csv", wraps=pd.read_csv) as parses:
        assert dataset[0] is None and dataset[0] is None
        assert parses.call_count == 1
