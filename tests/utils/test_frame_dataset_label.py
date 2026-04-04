"""Test suite for ChannelFrameDataset label-based access with duplicate filenames."""

import warnings
from pathlib import Path

import pytest

import wandas as wd
from wandas.utils.frame_dataset import ChannelFrameDataset


@pytest.fixture
def dup_label_dataset(tmp_path: Path) -> ChannelFrameDataset:
    """Create a dataset with duplicate filenames in different subdirectories.

    Structure:
        root/a/dup.wav  (440 Hz sine)
        root/b/dup.wav  (540 Hz sine)
    """
    root = tmp_path / "root"
    sub1 = root / "a"
    sub2 = root / "b"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    for i, sub in enumerate([sub1, sub2]):
        freq = 440 + i * 100  # 440 Hz and 540 Hz — distinct, deterministic
        sig = wd.generate_sin(freqs=[freq], duration=0.1, sampling_rate=8000)
        sig.to_wav(str(sub / "dup.wav"))

    return ChannelFrameDataset.from_folder(str(root), recursive=True)


class TestDatasetLabelAccess:
    """Test suite for label-based dataset access — duplicate filename handling."""

    def test_get_all_by_label_duplicate_returns_both(self, dup_label_dataset: ChannelFrameDataset) -> None:
        """get_all_by_label returns all frames matching the filename."""
        matches = dup_label_dataset.get_all_by_label("dup.wav")
        assert isinstance(matches, list)
        assert len(matches) == 2
        assert all(getattr(m, "sampling_rate", None) is not None for m in matches)

    def test_get_by_label_deprecated_emits_warning(self, dup_label_dataset: ChannelFrameDataset) -> None:
        """Deprecated get_by_label emits DeprecationWarning and returns first match."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            first = dup_label_dataset.get_by_label("dup.wav")
            assert any(issubclass(w.category, DeprecationWarning) for w in rec)

        assert first is not None
        assert getattr(first, "sampling_rate", None) is not None

    def test_getitem_str_key_returns_list(self, dup_label_dataset: ChannelFrameDataset) -> None:
        """String indexing returns list of all matching frames."""
        frames = dup_label_dataset["dup.wav"]
        assert isinstance(frames, list)
        assert len(frames) == 2
