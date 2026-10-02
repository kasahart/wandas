"""Regression contracts for bounded Dataset construction and subset snapshots."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import wandas as wd
import wandas.frames._dataset_inputs as inputs
import wandas.utils.frame_dataset as datasets
from wandas.io.readers import SoundFileReader


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    for index in range(3):
        sf.write(tmp_path / f"{index}.wav", np.full(80, index / 4), 8000, subtype="FLOAT")
    return tmp_path


def test_folder_snapshots_reused_resolver_mapping_once_per_item(folder: Path) -> None:
    shared = {"nested": {"trial": -1}}

    def resolver(path: Path) -> Mapping[str, object]:
        shared["nested"]["trial"] = int(path.stem)
        return shared

    with (
        patch.object(datasets, "deepcopy", wraps=datasets.deepcopy) as copies,
        patch.object(SoundFileReader, "get_file_info", side_effect=AssertionError("audio header read")),
        patch.object(SoundFileReader, "get_data", side_effect=AssertionError("PCM read")),
    ):
        dataset = wd.from_folder(str(folder), file_extensions=[".wav"], metadata_resolver=resolver)
        assert copies.call_count == 3
        assert [item.metadata["nested"] for item in dataset._lazy_frames] == [{"trial": 0}, {"trial": 1}, {"trial": 2}]
    shared["nested"]["trial"] = 99
    frame = dataset[0]
    assert frame is not None and frame.metadata["nested"] == {"trial": 0}


def test_dataframe_checks_dependency_once_for_all_scalar_cells() -> None:
    table = pd.DataFrame({"audio": ["missing.wav"] * 1000, "trial": range(1000), "optional": [pd.NA] * 1000})
    with patch.object(inputs, "require_pandas", wraps=inputs.require_pandas) as dependency:
        dataset = wd.from_table(table, path_column="audio")
        assert dependency.call_count == 1
    table.loc[0, "trial"] = 99
    assert dataset._lazy_frames[0].metadata == {"trial": 0, "optional": None}


def test_one_item_subset_copies_only_selected_metadata() -> None:
    dataset = wd.from_table(pd.DataFrame({"audio": ["missing.wav"] * 1000, "trial": range(1000)}), path_column="audio")
    with (
        patch.object(datasets.FrameDataset, "_get_file_paths", side_effect=AssertionError("full path list")),
        patch.object(SoundFileReader, "get_file_info", side_effect=AssertionError("audio header read")),
        patch.object(datasets, "deepcopy", wraps=datasets.deepcopy) as copies,
    ):
        assert len(dataset.sample(n=1)) == 1
        assert copies.call_count == 1
        copies.reset_mock()
        selected = dataset.select(trial=999)
        assert len(selected) == 1 and copies.call_count == 2
        copies.reset_mock()
        assert len(selected.sample(n=1)) == 1 and copies.call_count == 1


def test_subset_order_duplicates_nested_isolation_and_original_cache(folder: Path) -> None:
    dataset = wd.from_folder(
        str(folder),
        file_extensions=[".wav"],
        metadata_resolver=lambda path: {"trial": int(path.stem), "nested": {"tags": [path.stem]}},
    )
    subset = datasets._SampledFrameDataset(dataset, [2, 0, 2])
    assert [item.file_path.name for item in subset._lazy_frames] == ["2.wav", "0.wav", "2.wav"]
    nested = cast(dict[str, list[str]], subset._lazy_frames[0].metadata["nested"])
    nested["tags"].append("subset")
    assert dataset._lazy_frames[2].metadata["nested"] == {"tags": ["2"]}
    assert subset._lazy_frames[2].metadata["nested"] == {"tags": ["2"]}
    with patch.object(SoundFileReader, "get_file_info", wraps=SoundFileReader.get_file_info) as headers:
        frame = subset[0]
        assert frame is not None and subset[2] is frame and dataset[2] is frame
        assert headers.call_count == 1
    metadata = frame.metadata
    metadata["nested"]["tags"].append("public")
    assert frame.metadata["nested"] == {"tags": ["2"]}
    chained = subset.select(trial=2).sample(n=2)
    assert len(chained) == 2 and chained[0] is frame and chained[1] is frame


def test_subset_preserves_source_failure_cache(tmp_path: Path) -> None:
    dataset = wd.from_files([tmp_path / "missing.wav"])
    first, second = dataset.sample(n=1), dataset.sample(n=1)
    with patch.object(dataset, "_load_file", wraps=dataset._load_file) as loads:
        assert first[0] is None and second[0] is None and dataset[0] is None
        assert loads.call_count == 1
