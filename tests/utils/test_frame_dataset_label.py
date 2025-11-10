import warnings
from pathlib import Path

import wandas as wd
from wandas.utils.frame_dataset import ChannelFrameDataset


def _make_dup_files(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    sub1 = root / "a"
    sub2 = root / "b"
    sub1.mkdir()
    sub2.mkdir()

    # Create two files with the same filename in different subdirectories
    for i, sub in enumerate([sub1, sub2]):
        sig = wd.generate_sin(freqs=[440 + i * 100], duration=0.1, sampling_rate=8000)
        fname = sub / "dup.wav"
        sig.to_wav(str(fname))

    return root


def test_get_all_by_label_returns_multiple(tmp_path: Path) -> None:
    root = _make_dup_files(tmp_path)

    ds = ChannelFrameDataset.from_folder(str(root), recursive=True)

    matches = ds.get_all_by_label("dup.wav")
    assert isinstance(matches, list)
    assert len(matches) == 2
    # each match should look like a ChannelFrame (has sampling_rate)
    assert all(getattr(m, "sampling_rate", None) is not None for m in matches)


def test_get_by_label_deprecation_returns_first(tmp_path: Path) -> None:
    root = _make_dup_files(tmp_path)
    ds = ChannelFrameDataset.from_folder(str(root), recursive=True)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        first = ds.get_by_label("dup.wav")

        # DeprecationWarning should be emitted
        assert any(issubclass(w.category, DeprecationWarning) for w in rec)

    assert first is not None
    assert getattr(first, "sampling_rate", None) is not None


def test_getitem_str_returns_list(tmp_path: Path) -> None:
    root = _make_dup_files(tmp_path)
    ds = ChannelFrameDataset.from_folder(str(root), recursive=True)

    frames = ds["dup.wav"]
    assert isinstance(frames, list)
    assert len(frames) == 2
