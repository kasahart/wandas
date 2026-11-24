from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from dask.array.core import Array as DaArray

from wandas import from_files, from_table
from wandas.utils.frame_dataset import ChannelFrameDataset

_SR = 1000


def _write_hammer(path: Path, hits: tuple[float, ...]) -> np.ndarray:
    """Force pulses at known times on ch0 and a decaying response on ch1."""
    data = np.zeros((4 * _SR, 2), dtype=np.float32)
    for hit in hits:
        start = int(hit * _SR)
        data[start : start + 5, 0] = 0.5
        data[start : start + 100, 1] = np.exp(-np.arange(100) / 20.0)
    sf.write(path, data, _SR, subtype="FLOAT")
    return data.T


@pytest.fixture
def runs(tmp_path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    hits = {"a.wav": (0.5, 1.5, 3.0), "b.wav": (1.0,), "c.wav": ()}
    data = {name: _write_hammer(tmp_path / name, times) for name, times in hits.items()}
    table = pd.DataFrame({"audio": list(hits), "point": ["A", "B", "C"]})
    return table.assign(audio=[str(tmp_path / name) for name in hits]), data


def _hit(dataset: ChannelFrameDataset, index: int):
    frame = dataset[index]
    assert frame is not None
    return frame


class TestDatasetTrigger:
    def test_hits_of_all_recordings_carry_source_and_event_metadata(self, runs) -> None:
        table, data = runs
        hits = from_table(table, path_column="audio").trigger(0.1, duration=0.2, pre=0.01)

        assert isinstance(hits, ChannelFrameDataset)
        assert len(hits) == 4
        expected = [("A", 0, 0.5), ("A", 1, 1.5), ("A", 2, 3.0), ("B", 0, 1.0)]
        for index, (point, event, time) in enumerate(expected):
            frame = _hit(hits, index)
            assert frame.metadata["point"] == point
            assert frame.metadata["trigger_index"] == event
            assert frame.metadata["trigger_time"] == pytest.approx(time)
            assert frame.metadata["_source_file"].endswith(f"{point.lower()}.wav")
            assert frame.n_samples == 200
            start = int(round((time - 0.01) * _SR))
            np.testing.assert_array_equal(frame.data, data[f"{point.lower()}.wav"][:, start : start + 200])

    def test_hits_are_lazy_and_selectable_by_metadata(self, runs) -> None:
        table, _ = runs
        hits = from_table(table, path_column="audio").trigger(0.1, duration=0.2, pre=0.01)

        assert isinstance(_hit(hits, 0)._data, DaArray)
        point_a = hits.select(point="A")
        assert [_hit(point_a, i).metadata["trigger_index"] for i in range(len(point_a))] == [0, 1, 2]
        second = hits.select(trigger_index=1)
        assert len(second) == 1
        assert _hit(second, 0).metadata["trigger_time"] == pytest.approx(1.5)

    def test_transforms_after_trigger_keep_hit_metadata(self, runs) -> None:
        table, _ = runs
        hits = from_table(table, path_column="audio").trigger(0.1, duration=0.2, pre=0.01)

        resampled = hits.resample(500)
        frame = _hit(resampled, 3)
        assert frame.sampling_rate == 500
        assert frame.metadata["point"] == "B"
        assert frame.metadata["trigger_index"] == 0

    def test_trigger_uses_labelled_channel_and_leaves_source_unchanged(self, runs) -> None:
        table, _ = runs
        source = from_table(table, path_column="audio")
        hits = source.trigger(0.5, duration=0.2, channel=1, slope="falling")

        assert len(hits) == 4
        assert source.get_metadata()["file_count"] == 3
        assert "trigger_index" not in _hit(source, 0).metadata

    def test_failed_recordings_are_skipped_with_a_warning(self, runs, tmp_path: Path) -> None:
        table, _ = runs
        dataset = from_files([table["audio"][1], tmp_path / "missing.wav"])

        with pytest.warns(UserWarning, match=r"Skipped 1 recording\(s\) that failed to load: indices \[1\]"):
            hits = dataset.trigger(0.1, duration=0.2, pre=0.01)

        assert len(hits) == 1
        assert _hit(hits, 0).metadata["trigger_time"] == pytest.approx(1.0)

    def test_trigger_errors_of_a_recording_propagate(self, runs) -> None:
        table, _ = runs
        with pytest.raises(KeyError):
            from_table(table, path_column="audio").trigger(0.1, duration=0.2, channel="force")

    def test_existing_trigger_metadata_keys_are_rejected(self, runs) -> None:
        table, _ = runs
        dataset = from_table(table.assign(trigger_time=[1, 2, 3]), path_column="audio")
        with pytest.raises(ValueError, match="trigger_time"):
            dataset.trigger(0.1, duration=0.2)

    def test_dataset_without_hits_is_empty(self, runs) -> None:
        table, _ = runs
        hits = from_table(table, path_column="audio").trigger(10.0, duration=0.2)
        assert len(hits) == 0
