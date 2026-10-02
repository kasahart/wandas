from __future__ import annotations

import builtins
import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from wandas import from_files, from_table
from wandas.io.readers import SoundFileReader


def loaded(dataset, index):
    frame = dataset[index]
    assert frame is not None
    return frame


@pytest.fixture
def roots(tmp_path):
    paths = []
    for folder, value in [("root-a", 0.25), ("root-b", -0.5)]:
        p = tmp_path / folder / "same.wav"
        p.parent.mkdir()
        sf.write(p, np.full(800, value), 8000, subtype="FLOAT")
        paths.append(p)
    return paths


def test_file_list_order_duplicates_and_no_scan(roots):
    with (
        patch.object(Path, "glob", side_effect=AssertionError("directory scan")),
        patch.object(SoundFileReader, "get_file_info", wraps=SoundFileReader.get_file_info) as header,
        patch.object(SoundFileReader, "get_data", wraps=SoundFileReader.get_data) as pcm,
    ):
        ds = from_files(iter([roots[1], roots[0], roots[1]]))
        assert len(ds) == 3 and header.call_count == pcm.call_count == 0
        assert ds.get_metadata()["loaded_count"] == 0
        frame = loaded(ds, 0)
        assert header.call_count == 1 and pcm.call_count == 0
        assert frame.metadata["_source_file"] == str(roots[1])
        np.testing.assert_allclose(frame.to_numpy(), -0.5)
        assert pcm.call_count == 1 and ds[0] is frame
        assert len(ds["same.wav"]) == 3  # preserve existing label semantics
        assert header.call_count == 3 and pcm.call_count == 1


def test_relative_path_anchor_survives_cwd_change(roots, monkeypatch, tmp_path):
    ds = from_files(["root-a/same.wav"], base_dir=tmp_path)
    monkeypatch.chdir(roots[1].parent)
    assert loaded(ds, 0).metadata["_source_file"] == str(roots[0])


def test_table_duplicate_sources_keep_row_metadata_and_ignore_df_index(roots):
    table = pd.DataFrame(
        {
            "audio": [roots[0], roots[0], roots[1]],
            "trial": [11, 22, 33],
            "group": ["a", "b", "a"],
        },
        index=[5, 5, 9],
    )
    ds = from_table(table, path_column="audio")
    table.loc[:, "trial"] = 99
    assert len(ds) == 3
    assert loaded(ds, 0).metadata["trial"] == 11 and loaded(ds, 1).metadata["trial"] == 22
    assert ds[0] is not ds[1]  # distinct row items, not automatic source deduplication
    with patch.object(SoundFileReader, "get_data", side_effect=AssertionError("PCM")):
        selected = ds.select(group="a")
        assert len(selected) == 2
        trimmed = selected.trim(0, 0.05)
        assert loaded(trimmed, 0).metadata["trial"] == 11


def test_csv_anchor_and_string_types(roots, tmp_path, monkeypatch):
    csv_path = tmp_path / "catalog.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["audio", "code", "value", "note"])
        writer.writerow(["root-a/same.wav", "001", "1.2", ""])
    monkeypatch.chdir(roots[1].parent)
    frame = loaded(from_table(csv_path, path_column="audio"), 0)
    assert frame.metadata["_source_file"] == str(roots[0])
    assert frame.metadata["code"] == "001"
    assert frame.metadata["value"] == "1.2" and frame.metadata["note"] == ""


def test_explicit_base_overrides_csv_parent(roots, tmp_path):
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text("audio,label\nsame.wav,a\n")
    frame = loaded(from_table(csv_path, path_column="audio", base_dir=roots[1].parent), 0)
    assert frame.metadata["_source_file"] == str(roots[1])


def test_typed_scalars_and_missing_metadata(roots):
    table = pd.DataFrame(
        {
            "audio": [str(roots[0]), str(roots[1])],
            "code": pd.Series([1, pd.NA], dtype="Int64"),
            "flag": [True, False],
            "score": [np.nan, 0.25],
            "note": [None, ""],
        }
    )
    ds = from_table(table, path_column="audio")
    first, second = loaded(ds, 0).metadata, loaded(ds, 1).metadata
    assert type(first["code"]) is int and first["code"] == 1
    assert type(first["flag"]) is bool and first["flag"] is True
    assert first["score"] is None and first["note"] is None
    assert second["code"] is None and second["note"] == ""
    assert type(second["score"]) is float


def test_mapping_rows_missing_meta_and_explicit_selection(roots):
    ds = from_table(
        pd.DataFrame(
            [
                {"audio": roots[0], "label": "a", "bad": object()},
                {"audio": roots[1], "bad": object()},
            ]
        ),
        path_column="audio",
        metadata_columns=["label"],
    )
    assert loaded(ds, 0).metadata["label"] == "a" and loaded(ds, 1).metadata["label"] is None
    assert "bad" not in loaded(ds, 0).metadata


@pytest.mark.parametrize("value", [None, "", "   ", np.nan, pd.NA, pd.NaT, 123])
def test_invalid_audio_cell_rejected_without_read(roots, value):
    with patch.object(SoundFileReader, "get_file_info", side_effect=AssertionError("header")):
        with pytest.raises((ValueError, TypeError), match="row 0.audio"):
            from_table(pd.DataFrame([{"audio": value}]), path_column="audio")


def test_invalid_table_column_contracts(roots, tmp_path):
    with pytest.raises(TypeError):
        from_table(pd.DataFrame([{"audio": roots[0]}]))  # ty: ignore[missing-argument]  # no column guessing
    with pytest.raises(ValueError, match="Unknown path_column"):
        from_table(pd.DataFrame([{"other": roots[0]}]), path_column="audio")
    with pytest.raises(ValueError, match="Duplicate"):
        from_table(
            pd.DataFrame([[roots[0], "a"]], columns=["audio", "audio"]),
            path_column="audio",
        )
    with pytest.raises(ValueError, match="Reserved"):
        from_table(pd.DataFrame([{"audio": roots[0], "_source_file": "fake"}]), path_column="audio")
    assert (
        len(
            from_table(
                pd.DataFrame([{"audio": roots[0], "_source_file": "fake"}]),
                path_column="audio",
                metadata_columns=[],
            )
        )
        == 1
    )
    with pytest.raises(ValueError, match="Unknown metadata"):
        from_table(pd.DataFrame([{"audio": roots[0]}]), path_column="audio", metadata_columns=["typo"])
    with pytest.raises(ValueError, match="cannot also"):
        from_table(pd.DataFrame([{"audio": roots[0]}]), path_column="audio", metadata_columns=["audio"])
    csv_path = tmp_path / "duplicate.csv"
    csv_path.write_text("audio,audio\na.wav,b.wav\n")
    with pytest.raises(ValueError, match="Duplicate"):
        from_table(csv_path, path_column="audio")


@pytest.mark.parametrize("text", ["audio,label\na.wav,x,extra\n", "audio,label\na.wav\n"])
def test_malformed_csv_rejected(tmp_path, text):
    p = tmp_path / "bad.csv"
    p.write_text(text)
    with pytest.raises(ValueError, match="CSV cell count"):
        from_table(p, path_column="audio")


@pytest.mark.parametrize("value", [object(), [1, 2], float("inf"), pd.Timestamp("2026-01-01")])
def test_unsupported_selected_metadata_needs_explicit_conversion(roots, value):
    with pytest.raises((TypeError, ValueError)):
        from_table(pd.DataFrame([{"audio": roots[0], "value": value}]), path_column="audio")


def test_missing_and_broken_audio_are_lazy_cached_failures(tmp_path):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"invalid")
    with patch.object(SoundFileReader, "get_file_info", wraps=SoundFileReader.get_file_info) as header:
        ds = from_files([tmp_path / "missing.wav", broken])
        assert header.call_count == 0 and len(ds) == 2
        assert ds[0] is None and ds[1] is None
        count = header.call_count
        assert ds[0] is None and ds[1] is None and header.call_count == count
        assert ds.get_metadata()["loaded_count"] == 2


def test_empty_and_invalid_file_lists():
    assert len(from_files([])) == 0
    with pytest.raises(TypeError):
        from_files("a.wav")
    with pytest.raises(ValueError, match="local paths"):
        from_files(["https://example.invalid/a.wav"])
    with pytest.raises(TypeError):
        from_files([b"RIFF"])  # ty: ignore[invalid-argument-type]
    empty = pd.DataFrame(columns=["audio", "label"])
    assert len(from_table(empty, path_column="audio")) == 0


def test_csv_and_file_list_do_not_import_pandas(roots, tmp_path):
    p = tmp_path / "catalog.csv"
    p.write_text("audio,label\nroot-a/same.wav,a\n")
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "pandas" or name.startswith("pandas."):
            raise AssertionError("unnecessary pandas import")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded):
        assert len(from_files(roots)) == 2
        assert len(from_table(p, path_column="audio")) == 1


def test_default_cwd_snapshot_and_stft_sample_laziness(roots, monkeypatch):
    monkeypatch.chdir(roots[0].parent)
    ds = from_files(["same.wav"])
    monkeypatch.chdir(roots[1].parent)
    with patch.object(SoundFileReader, "get_data", side_effect=AssertionError("PCM")):
        subset = ds.sample(n=1, seed=123)
        spectrum = ds.stft(n_fft=128, hop_length=32)
        assert loaded(subset, 0).metadata["_source_file"] == str(roots[0])
        assert loaded(spectrum, 0).metadata["_source_file"] == str(roots[0])


def test_metadata_column_names_not_guessed(roots):
    with pytest.raises(TypeError, match="sequence"):
        from_table(
            pd.DataFrame([{"audio": roots[0], "label": "a"}]),
            path_column="audio",
            metadata_columns="label",
        )
    ds = from_table(
        pd.DataFrame([{"audio": roots[0], "key": "user-key", "partition_0": "user-value"}]),
        path_column="audio",
    )
    assert loaded(ds, 0).metadata["key"] == "user-key"
    assert loaded(ds, 0).metadata["partition_0"] == "user-value"


def test_external_origin_propagates_through_transforms_and_subsets(roots):
    ds = from_table(pd.DataFrame({"audio": roots, "group": ["a", "b"]}), path_column="audio")
    for candidate in [ds, ds.select(group="a"), ds.sample(n=1), ds.trim(0, 0.05), ds.stft(n_fft=128)]:
        assert candidate.folder_path is None
        assert candidate.get_metadata()["folder_path"] is None
        assert candidate.get_metadata()["loaded_count"] == 0


def test_public_metadata_is_owned_after_subset_and_transform(roots):
    table = pd.DataFrame({"audio": [roots[0]], "trial": [1]})
    ds = from_table(table, path_column="audio")
    subset = ds.select(trial=1)
    table.loc[0, "trial"] = 99
    frame = loaded(ds, 0)
    exposed = frame.metadata
    exposed["trial"] = 88
    assert loaded(subset, 0).metadata["trial"] == 1
    assert loaded(ds.trim(0, 0.05), 0).metadata["trial"] == 1


def test_construct_select_sample_and_compute_boundaries(roots):
    with (
        patch.object(SoundFileReader, "get_file_info", wraps=SoundFileReader.get_file_info) as header,
        patch.object(SoundFileReader, "get_data", wraps=SoundFileReader.get_data) as pcm,
    ):
        ds = from_table(pd.DataFrame({"audio": roots, "trial": [1, 2]}), path_column="audio")
        selected = ds.select(trial=2)
        sampled = selected.sample(n=1)
        ds.get_metadata()
        assert header.call_count == pcm.call_count == 0
        frame = loaded(sampled, 0)
        assert frame.metadata["trial"] == 2
        assert header.call_count == 1 and pcm.call_count == 0
        np.testing.assert_allclose(frame.to_numpy(), -0.5)
        assert pcm.call_count == 1 and header.call_count == 1
        assert sampled[0] is frame


def test_deferred_failure_and_source_lifetime(roots):
    ds = from_files(roots)
    frame = loaded(ds, 0)
    roots[0].unlink()  # Header was inspected, PCM has not been read.
    with pytest.raises(sf.LibsndfileError):
        frame.to_numpy()
    assert ds[0] is frame  # A cached Frame is distinct from a cached load failure.
    np.testing.assert_allclose(loaded(ds, 1).to_numpy(), -0.5)


def test_failed_item_does_not_block_valid_item(roots, tmp_path, caplog):
    ds = from_files([tmp_path / "missing.wav", roots[1]])
    assert ds[0] is None
    assert "Failed to load" in caplog.text
    np.testing.assert_allclose(loaded(ds, 1).to_numpy(), -0.5)


def test_empty_csv_catalog_with_header(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("audio,label\n")
    assert len(from_table(p, path_column="audio")) == 0


def test_table_base_dir_does_not_relocate_csv_itself(roots, tmp_path, monkeypatch):
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text("audio,label\nsame.wav,b\n")
    monkeypatch.chdir(tmp_path)
    ds = from_table("catalog.csv", path_column="audio", base_dir=roots[1].parent)
    assert loaded(ds, 0).metadata["_source_file"] == str(roots[1])


def test_dataframe_cwd_is_fixed_at_construction(roots, monkeypatch):
    monkeypatch.chdir(roots[0].parent)
    ds = from_table(pd.DataFrame({"audio": ["same.wav"]}), path_column="audio")
    monkeypatch.chdir(roots[1].parent)
    assert loaded(ds, 0).metadata["_source_file"] == str(roots[0])


def test_classmethod_entry_points_and_top_level_exports(roots):
    import wandas as wd
    from wandas.utils.frame_dataset import ChannelFrameDataset

    assert "from_files" in wd.__all__ and "from_table" in wd.__all__
    assert type(ChannelFrameDataset.from_files(roots)) is ChannelFrameDataset
    assert (
        type(ChannelFrameDataset.from_table(pd.DataFrame({"audio": roots}), path_column="audio")) is ChannelFrameDataset
    )


def test_non_dataframe_table_rejected():
    with pytest.raises(TypeError, match="DataFrame"):
        from_table([{"audio": "a.wav"}], path_column="audio")  # ty: ignore[invalid-argument-type]


def test_non_string_column_names_and_null_path_rejected(roots):
    with pytest.raises(ValueError, match="Column names"):
        from_table(pd.DataFrame({"audio": roots, 3: [1, 2]}), path_column="audio")
    with pytest.raises(ValueError, match="null characters"):
        from_files(["a\0.wav"])


def test_csv_quoted_values_and_malformed_quotes(roots, tmp_path):
    path = tmp_path / "catalog.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["audio", "note"])
        writer.writerow([roots[0], "日本語, quoted\nvalue"])
    assert loaded(from_table(path, path_column="audio"), 0).metadata["note"] == "日本語, quoted\nvalue"
    path.write_text('audio,note\n"unterminated,a\n')
    with pytest.raises(ValueError, match="Malformed CSV catalog"):
        from_table(path, path_column="audio")


def test_folder_origin_required_without_explicit_items():
    from wandas.utils.frame_dataset import ChannelFrameDataset

    with pytest.raises(ValueError, match="folder_path is required"):
        ChannelFrameDataset(folder_path=None)


def test_explicit_items_eager_initialization(roots):
    from wandas.utils.frame_dataset import ChannelFrameDataset

    ds = ChannelFrameDataset(folder_path=None, lazy_loading=False, _items=[(roots[0], {"trial": 1})])
    assert ds.get_metadata()["loaded_count"] == 1
    assert loaded(ds, 0).metadata["trial"] == 1
    assert ds.folder_path is None


def test_table_requires_explicit_nonempty_path_column():
    with pytest.raises(ValueError, match="path_column must explicitly"):
        from_table(pd.DataFrame({"audio": ["a.wav"]}), path_column="")
