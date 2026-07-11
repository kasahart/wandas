from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

import wandas as wd
from wandas.frames.channel import ChannelFrame
from wandas.utils.frame_dataset import ChannelFrameDataset, _SampledFrameDataset


@pytest.fixture
def metadata_audio_folder(tmp_path: Path) -> Path:
    signal = np.linspace(-0.5, 0.5, 256, dtype=np.float32)
    paths = [
        "fan/train/section_00_source.wav",
        "fan/test/section_01_target.wav",
        "pump/train/section_00_source.wav",
    ]
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, signal, 8_000)
    return tmp_path


def dcase_resolver(path: Path) -> Mapping[str, object]:
    machine, split, filename = path.parts
    section, number, domain = filename.removesuffix(".wav").split("_")
    return {
        "machine": machine,
        "split": split,
        "section": f"{section}_{number}",
        "domain": domain,
    }


def test_resolver_runs_once_per_relative_path_without_loading_headers(metadata_audio_folder: Path) -> None:
    calls: list[Path] = []

    def resolver(path: Path) -> Mapping[str, object]:
        calls.append(path)
        return dcase_resolver(path)

    with patch.object(ChannelFrame, "from_file") as from_file:
        dataset = wd.from_folder(
            str(metadata_audio_folder),
            recursive=True,
            file_extensions=[".wav"],
            metadata_resolver=resolver,
        )

    assert calls == [
        Path("fan/test/section_01_target.wav"),
        Path("fan/train/section_00_source.wav"),
        Path("pump/train/section_00_source.wav"),
    ]
    from_file.assert_not_called()
    assert all(not lazy_frame.is_loaded for lazy_frame in dataset._lazy_frames)


def test_resolver_result_is_deep_copied_and_attached_to_new_frame(metadata_audio_folder: Path) -> None:
    nested = {"conditions": {"speed": 1_000}}
    dataset = ChannelFrameDataset.from_folder(
        str(metadata_audio_folder),
        recursive=True,
        file_extensions=[".wav"],
        metadata_resolver=lambda _path: nested,
    )
    nested["conditions"]["speed"] = 2_000

    frame = dataset[0]

    assert frame is not None
    assert frame.metadata["conditions"] == {"speed": 1_000}
    assert frame.metadata["_source_file"].endswith("section_01_target.wav")
    assert frame.previous is None
    assert frame is not dataset._load_file(dataset._lazy_frames[0].file_path)


@pytest.mark.parametrize(
    ("resolver", "error", "message"),
    [
        (lambda _path: None, TypeError, "must return a mapping"),
        (lambda _path: {1: "bad"}, TypeError, "keys must be strings"),
        (lambda _path: {"_source_file": "bad"}, ValueError, "reserved key"),
    ],
)
def test_invalid_resolver_results_fail_fast_with_path(
    metadata_audio_folder: Path,
    resolver: Any,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message) as exc_info:
        ChannelFrameDataset.from_folder(
            str(metadata_audio_folder),
            recursive=True,
            file_extensions=[".wav"],
            metadata_resolver=resolver,
        )
    assert "fan/test/section_01_target.wav" in str(exc_info.value)


def test_resolver_exception_fails_fast_with_path(metadata_audio_folder: Path) -> None:
    def resolver(_path: Path) -> Mapping[str, object]:
        raise LookupError("missing row")

    with pytest.raises(RuntimeError, match="fan/test/section_01_target.wav.*missing row"):
        ChannelFrameDataset.from_folder(
            str(metadata_audio_folder),
            recursive=True,
            file_extensions=[".wav"],
            metadata_resolver=resolver,
        )


def test_select_is_exact_and_lazy_preserves_order_and_metadata(metadata_audio_folder: Path) -> None:
    dataset = ChannelFrameDataset.from_folder(
        str(metadata_audio_folder),
        recursive=True,
        file_extensions=[".wav"],
        metadata_resolver=dcase_resolver,
    )

    with patch.object(ChannelFrame, "from_file") as from_file:
        selected = dataset.select(machine="fan", split="train")
        empty = dataset.select(machine="fan", split="validation")
        copied = dataset.select()

    from_file.assert_not_called()
    assert [path.name for path in selected._get_file_paths()] == ["section_00_source.wav"]
    assert len(empty) == 0
    assert copied is not dataset
    assert copied._get_file_paths() == dataset._get_file_paths()
    assert selected.get_metadata()["is_sampled"] is False
    assert all(not lazy_frame.is_loaded for lazy_frame in selected._lazy_frames)


def test_select_rejects_unknown_metadata_key(metadata_audio_folder: Path) -> None:
    dataset = ChannelFrameDataset.from_folder(
        str(metadata_audio_folder),
        recursive=True,
        file_extensions=[".wav"],
        metadata_resolver=dcase_resolver,
    )
    with pytest.raises(KeyError, match="Unknown file metadata key.*machin"):
        dataset.select(machin="fan")


def test_select_none_requires_metadata_key_presence(metadata_audio_folder: Path) -> None:
    dataset = ChannelFrameDataset.from_folder(
        str(metadata_audio_folder),
        recursive=True,
        file_extensions=[".wav"],
        metadata_resolver=lambda path: {"optional": None} if path.parts[1] == "test" else {},
    )

    selected = dataset.select(optional=None)

    assert len(selected) == 1
    assert selected._get_file_paths()[0].name == "section_01_target.wav"


def test_metadata_attachment_preserves_transform_previous(metadata_audio_folder: Path) -> None:
    dataset = ChannelFrameDataset.from_folder(
        str(metadata_audio_folder),
        recursive=True,
        file_extensions=[".wav"],
        metadata_resolver=dcase_resolver,
    )
    source = dataset[0]

    transformed = dataset.normalize()[0]

    assert source is not None
    assert transformed is not None
    assert transformed.previous is source


def test_metadata_survives_select_sample_and_processing(metadata_audio_folder: Path) -> None:
    dataset = ChannelFrameDataset.from_folder(
        str(metadata_audio_folder),
        recursive=True,
        file_extensions=[".wav"],
        metadata_resolver=dcase_resolver,
    )
    sampled = dataset.select(split="train").sample(n=1, seed=3)
    assert isinstance(sampled, _SampledFrameDataset)

    frame = sampled.apply(lambda value: value.normalize())[0]
    spectrogram = dataset.select(machine="fan").stft(n_fft=64)[0]

    assert frame is not None
    assert frame.metadata["split"] == "train"
    assert spectrogram is not None
    assert spectrogram.metadata["machine"] == "fan"


def test_sample_then_select_preserves_subset_contract_without_loading(metadata_audio_folder: Path) -> None:
    dataset = ChannelFrameDataset.from_folder(
        str(metadata_audio_folder),
        recursive=True,
        file_extensions=[".wav"],
        metadata_resolver=dcase_resolver,
    )
    sampled = dataset.sample(n=2, seed=3)

    with patch.object(ChannelFrame, "from_file") as from_file:
        selected = sampled.select(split="train")

    from_file.assert_not_called()
    assert len(selected) == 1
    assert selected._get_file_paths()[0].name == "section_00_source.wav"
    assert selected.get_metadata()["is_sampled"] is False
    frame = selected[0]
    assert frame is not None
    assert frame.metadata["split"] == "train"


def test_csv_lookup_resolver(metadata_audio_folder: Path) -> None:
    lookup = {
        "fan/train/section_00_source.wav": {"load": "low", "rpm": 1_000},
    }
    dataset = wd.from_folder(
        str(metadata_audio_folder),
        recursive=True,
        file_extensions=[".wav"],
        metadata_resolver=lambda path: lookup.get(path.as_posix(), {}),
    )

    selected = dataset.select(load="low", rpm=1_000)

    assert len(selected) == 1
    assert selected._get_file_paths()[0].name == "section_00_source.wav"


@pytest.fixture
def partitioned_audio_folder(tmp_path: Path) -> Path:
    signal = np.linspace(-0.5, 0.5, 256, dtype=np.float32)
    paths = [
        "group_a/recording_001.wav",
        "group_a/batch_01/recording_002.wav",
        "group=group_b/batch=batch_02/recording_003.wav",
    ]
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, signal, 8_000)
    return tmp_path


def test_path_metadata_infers_plain_and_hive_partitions_without_loading(
    partitioned_audio_folder: Path,
) -> None:
    with patch.object(ChannelFrame, "from_file") as from_file:
        dataset = wd.from_folder(
            str(partitioned_audio_folder),
            recursive=True,
            file_extensions=[".wav"],
            path_metadata=True,
        )

    from_file.assert_not_called()
    assert [lazy_frame.metadata for lazy_frame in dataset._lazy_frames] == [
        {"group": "group_b", "batch": "batch_02"},
        {"partition_0": "group_a", "partition_1": "batch_01"},
        {"partition_0": "group_a"},
    ]
    assert all(not lazy_frame.is_loaded for lazy_frame in dataset._lazy_frames)


def test_path_metadata_select_supports_missing_keys_and_processing(
    partitioned_audio_folder: Path,
) -> None:
    dataset = ChannelFrameDataset.from_folder(
        str(partitioned_audio_folder),
        recursive=True,
        file_extensions=[".wav"],
        path_metadata=True,
    )

    deep = dataset.select(partition_1="batch_01")
    transformed = dataset.normalize().stft(n_fft=64).select(partition_0="group_a")

    assert [path.name for path in deep._get_file_paths()] == ["recording_002.wav"]
    assert [path.name for path in transformed._get_file_paths()] == [
        "recording_002.wav",
        "recording_001.wav",
    ]
    frame = transformed[0]
    assert frame is not None
    assert frame.metadata["partition_0"] == "group_a"
    assert frame.metadata["partition_1"] == "batch_01"


def test_path_metadata_false_preserves_existing_behavior(partitioned_audio_folder: Path) -> None:
    dataset = ChannelFrameDataset.from_folder(
        str(partitioned_audio_folder),
        recursive=True,
        file_extensions=[".wav"],
    )

    assert all(lazy_frame.metadata == {} for lazy_frame in dataset._lazy_frames)


def test_path_metadata_excludes_root_and_filename(tmp_path: Path) -> None:
    sf.write(tmp_path / "recording.wav", np.zeros(64, dtype=np.float32), 8_000)

    dataset = wd.from_folder(str(tmp_path), file_extensions=[".wav"], path_metadata=True)

    assert dataset._lazy_frames[0].metadata == {}


def test_path_metadata_rejects_metadata_resolver(partitioned_audio_folder: Path) -> None:
    with pytest.raises(ValueError, match="path_metadata=True cannot be combined with metadata_resolver"):
        wd.from_folder(
            str(partitioned_audio_folder),
            recursive=True,
            path_metadata=True,
            metadata_resolver=lambda _path: {},
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "group=first/group=second/recording.wav",
    ],
)
def test_path_metadata_rejects_duplicate_or_colliding_partition_keys(
    tmp_path: Path,
    relative_path: str,
) -> None:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    sf.write(path, np.zeros(64, dtype=np.float32), 8_000)

    with pytest.raises(ValueError, match=r"Duplicate path metadata key.*recording\.wav"):
        ChannelFrameDataset.from_folder(
            str(tmp_path),
            recursive=True,
            file_extensions=[".wav"],
            path_metadata=True,
        )


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("_source_file=spoof/recording.wav", "cannot set reserved key"),
        ("partition_0=spoof/recording.wav", "uses the generated partition namespace"),
        ("partition_12=spoof/recording.wav", "uses the generated partition namespace"),
    ],
)
def test_path_metadata_rejects_reserved_hive_keys(
    tmp_path: Path,
    relative_path: str,
    message: str,
) -> None:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    sf.write(path, np.zeros(64, dtype=np.float32), 8_000)

    with pytest.raises(ValueError, match=message) as exc_info:
        ChannelFrameDataset.from_folder(
            str(tmp_path),
            recursive=True,
            file_extensions=[".wav"],
            path_metadata=True,
        )
    assert "recording.wav" in str(exc_info.value)
