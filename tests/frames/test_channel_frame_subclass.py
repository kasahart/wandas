"""A ChannelFrame subclass keeps its class through loading and channel updates."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame


class ProjectFrame(ChannelFrame):
    """A project extension that adds methods to ChannelFrame."""

    def doubled(self) -> ProjectFrame:
        return self * 2.0


@pytest.fixture
def samples() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(-0.5, 0.5, size=(2, 800))


def test_from_numpy_returns_subclass(samples: np.ndarray) -> None:
    frame = ProjectFrame.from_numpy(samples, sampling_rate=8_000)

    assert type(frame) is ProjectFrame
    np.testing.assert_allclose(frame.doubled().to_numpy(), samples * 2.0)


def test_read_wav_returns_subclass(samples: np.ndarray, tmp_path: Path) -> None:
    path = tmp_path / "signal.wav"
    ChannelFrame.from_numpy(samples, sampling_rate=8_000).to_wav(path)

    from_path = ProjectFrame.read_wav(path, labels=["a", "b"])
    buffer = io.BytesIO(path.read_bytes())
    from_bytes = ProjectFrame.read_wav(buffer)
    from_file = ProjectFrame.from_file(path, channel=1)

    assert type(from_path) is ProjectFrame
    assert type(from_bytes) is ProjectFrame
    assert type(from_file) is ProjectFrame
    assert [ch.label for ch in from_path.channels] == ["a", "b"]
    np.testing.assert_allclose(from_path.to_numpy(), ChannelFrame.read_wav(path).to_numpy())
    np.testing.assert_allclose(from_file.to_numpy(), from_path.to_numpy()[1])


def test_read_csv_returns_subclass(tmp_path: Path) -> None:
    path = tmp_path / "signal.csv"
    path.write_text("time,x\n0.0,1.0\n0.001,2.0\n0.002,3.0\n")

    frame = ProjectFrame.read_csv(str(path))

    assert type(frame) is ProjectFrame
    np.testing.assert_allclose(frame.to_numpy(), [1.0, 2.0, 3.0])


def test_base_loaders_still_return_channel_frame(samples: np.ndarray, tmp_path: Path) -> None:
    path = tmp_path / "signal.wav"
    ChannelFrame.from_numpy(samples, sampling_rate=8_000).to_wav(path)

    assert type(ChannelFrame.read_wav(path)) is ChannelFrame
    assert type(ChannelFrame.from_numpy(samples, sampling_rate=8_000)) is ChannelFrame


def test_operations_and_channel_updates_keep_subclass(samples: np.ndarray) -> None:
    frame = ProjectFrame.from_numpy(samples, sampling_rate=8_000, ch_labels=["a", "b"])

    filtered = frame.low_pass_filter(cutoff=1_000.0)
    added = frame.add_channel(samples[0], label="c")
    removed = added.remove_channel("c")
    stacked = frame.concat_frame(ProjectFrame.from_numpy(samples, sampling_rate=8_000, ch_labels=["c", "d"]))

    for result in (filtered, added, removed, stacked):
        assert type(result) is ProjectFrame
    assert [ch.label for ch in stacked.channels] == ["a", "b", "c", "d"]
    np.testing.assert_allclose(removed.to_numpy(), samples)
    np.testing.assert_allclose(frame.to_numpy(), samples)


class TaggedFrame(ChannelFrame):
    """A subclass with its own required constructor state."""

    def __init__(self, *args: Any, tag: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tag = tag

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        return {"tag": self.tag}


def test_channel_updates_keep_subclass_constructor_state(samples: np.ndarray) -> None:
    frame = TaggedFrame(data=da.from_array(samples, chunks=(1, -1)), sampling_rate=8_000, tag="custom")

    added = frame.add_channel(samples[0], label="extra")
    results = (frame * 2.0, added, added.remove_channel("extra"), frame.concat_frame(frame, suffix_on_dup="_2"))

    for result in results:
        assert type(result) is TaggedFrame
        assert result.tag == "custom"
