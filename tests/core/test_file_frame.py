from pathlib import Path
from typing import Any, Optional

import ipywidgets as widgets
import pytest

from wandas.core.channel_frame import ChannelFrame
from wandas.core.file_frame import FileFrame

# language: python


# Define a DummyChannelFrame to simulate ChannelFrame behavior for testing.
class DummyChannelFrame(ChannelFrame):
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier

    def describe(
        self,
        axis_config: Optional[dict[str, dict[str, Any]]] = None,
        cbar_config: Optional[dict[str, Any]] = None,
    ) -> widgets.VBox:
        # Return an HTML widget for simplicity.
        return widgets.VBox([widgets.HTML(value=f"Channel: {self.identifier}")])

    # For equality testing.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DummyChannelFrame):
            return False
        return self.identifier == other.identifier


@pytest.fixture  # type: ignore [misc, unused-ignore]
def dummy_read_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    # Monkey-patch ChannelFrame.read_wav so that it returns a DummyChannelFrame.
    def fake_read_wav(filename: str) -> DummyChannelFrame:
        # For testing, we simulate the channel frame with the filename (or part of it)
        return DummyChannelFrame(identifier=filename)

    monkeypatch.setattr(ChannelFrame, "read_wav", fake_read_wav)


def test_from_filelist_valid(dummy_read_wav: None) -> None:
    # Provide a list of valid .wav file names.
    files = ["test1.wav", "test2.wav", "sample.wav"]
    file_frame = FileFrame.from_filelist(files, label="Test FileFrame")
    # Assert that each channel_frame is the dummy one returned by fake_read_wav.
    assert len(file_frame.channel_frames) == len(files)
    for file, channel in zip(files, file_frame.channel_frames):
        assert channel == DummyChannelFrame(identifier=file)
    assert file_frame.label == "Test FileFrame"


def test_from_filelist_invalid_extension() -> None:
    # Provide a list with a non-supported file extension.
    files = ["test1.mp3"]
    with pytest.raises(ValueError) as excinfo:
        FileFrame.from_filelist(files)
    assert "Unsupported file format" in str(excinfo.value)


def test_from_dir(tmp_path: Path, dummy_read_wav: None) -> None:
    # Create temporary directory with some .wav files and others.
    d = tmp_path / "subdir"
    d.mkdir()
    wav_files = ["a.wav", "b.wav"]
    txt_files = ["c.txt", "d.doc"]
    all_files = wav_files + txt_files
    for filename in all_files:
        file_path = d / filename
        file_path.write_text("dummy data")
    # Call from_dir with suffix filter to only include .wav files.
    file_frame = FileFrame.from_dir(str(d), label="FromDir Test", suffix=".wav")
    # We expect two files.
    assert len(file_frame.channel_frames) == len(wav_files)
    # Check that channels are created with the correct identifiers.
    expected_ids = [str(d / f) for f in wav_files]
    for expected, channel in zip(expected_ids, file_frame.channel_frames):
        assert channel == DummyChannelFrame(identifier=expected)
    assert file_frame.label == "FromDir Test"


def test_describe() -> None:
    # Create a list of dummy channel frames.
    dummy_channels: list[ChannelFrame] = [
        DummyChannelFrame("ch1"),
        DummyChannelFrame("ch2"),
    ]
    # Create a FileFrame manually.
    file_frame = FileFrame(channel_frames=dummy_channels, label="Describe Test")
    # Call describe and verify it returns an ipywidgets.VBox with correct children.
    vbox = file_frame.describe()
    assert isinstance(vbox, widgets.VBox)
    # The children should match the output of each dummy channel's describe.
    expected_children = [ch.describe() for ch in dummy_channels]
    # Assert the number of children is as expected.
    assert len(vbox.children) == len(expected_children)
    # Compare individual widget content.
    for actual_vbox, expected_vbox in zip(vbox.children, expected_children):
        # Ensure both are VBox instances.
        assert isinstance(actual_vbox, widgets.VBox)
        assert isinstance(expected_vbox, widgets.VBox)
        # Compare the value of the HTML widget inside the VBox.
        assert actual_vbox.children[0].value == expected_vbox.children[0].value


def test_iteration_getitem_len() -> None:
    # Create several dummy channel frames.
    dummy_channels: list[ChannelFrame] = [
        DummyChannelFrame("ch1"),
        DummyChannelFrame("ch2"),
        DummyChannelFrame("ch3"),
    ]
    file_frame = FileFrame(channel_frames=dummy_channels, label="Iteration Test")
    # Test __len__
    assert len(file_frame) == 3
    # Test __iter__
    iterated = list(iter(file_frame))
    assert iterated == dummy_channels
    # Test __getitem__ with a valid index.
    assert file_frame[1] == DummyChannelFrame("ch2")
    # Test invalid index raises IndexError.
    with pytest.raises(IndexError):
        _ = file_frame[5]
    # Test __getitem__ with invalid key type.
    with pytest.raises(TypeError):
        _ = file_frame["ch1"]
