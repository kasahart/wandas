import logging
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from matplotlib.axes import Axes

# Import classes under test
from wandas.frames.channel import ChannelFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.utils.frame_dataset import (
    ChannelFrameDataset,
    FrameDataset,
    LazyFrame,  # Import new class
    SpectrogramFrameDataset,
    _SampledFrameDataset,
)
from wandas.utils.types import NDArrayReal

_da_from_array = da.from_array

# --- Test Fixtures ---


@pytest.fixture
def sample_wav_data() -> tuple[int, NDArrayReal]:
    """Generate deterministic WAV data — 2-channel sine waves at known frequencies."""
    sampling_rate = 16000
    duration = 1.0
    n_samples = int(sampling_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    # Channel 1: 440 Hz sine, Channel 2: 880 Hz sine — analytically predictable
    ch1 = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ch2 = (0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    data = np.column_stack([ch1, ch2])
    return sampling_rate, data


@pytest.fixture
def sample_csv_data() -> tuple[int, pd.DataFrame]:
    """Generate deterministic CSV data — 2-channel sensor signals at known frequencies."""
    sampling_rate = 100  # 100 Hz — typical for sensor data (accelerometer, etc.)
    duration = 1.0  # 1 second
    n_samples = int(sampling_rate * duration)
    time_col = np.linspace(0, duration, n_samples, endpoint=False)
    # Analytically predictable: 5 Hz sine and 10 Hz cosine
    ch1_data = np.sin(2 * np.pi * 5 * time_col)  # 5 Hz sine wave
    ch2_data = np.cos(2 * np.pi * 10 * time_col)  # 10 Hz cosine wave
    df = pd.DataFrame({"time": time_col, "SensorA": ch1_data, "SensorB": ch2_data})
    return sampling_rate, df


@pytest.fixture
def create_test_files(
    tmp_path: Path,
    sample_wav_data: tuple[int, NDArrayReal],
    sample_csv_data: tuple[int, pd.DataFrame],
) -> Path:
    """Create sample WAV and CSV files in a temporary directory."""
    wav_sr, wav_data = sample_wav_data
    csv_sr, csv_df = sample_csv_data

    # Create WAV files
    sf.write(tmp_path / "test1.wav", wav_data, wav_sr)
    sf.write(tmp_path / "test2.wav", wav_data * 0.5, wav_sr)  # Different data

    # Create CSV file
    csv_df.to_csv(tmp_path / "test3.csv", index=False)

    # Create a subdirectory for recursive test
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    sf.write(sub_dir / "sub_test.wav", wav_data * 0.8, wav_sr)

    # Create a non-target file
    (tmp_path / "other.txt").write_text("ignore me")

    # Create an empty subdirectory for empty-dataset tests
    empty_subdir = tmp_path / "empty_subdir"
    empty_subdir.mkdir()

    return tmp_path


# --- Test LazyFrame ---


class TestLazyFrame:
    """Test suite for LazyFrame — lazy loading wrapper with caching."""

    def test_init_default_state_unloaded(self) -> None:
        """New LazyFrame starts in unloaded state with no frame."""
        file_path = Path("/path/to/file.wav")
        lazy_frame: LazyFrame[ChannelFrame] = LazyFrame(file_path)

        assert lazy_frame.file_path == file_path
        assert lazy_frame.frame is None
        assert lazy_frame.is_loaded is False
        assert lazy_frame.load_attempted is False

    def test_ensure_loaded_first_call_invokes_loader(self) -> None:
        """First ensure_loaded call invokes loader and caches result."""
        file_path = Path("/path/to/file.wav")
        lazy_frame: LazyFrame[ChannelFrame] = LazyFrame(file_path)

        mock_frame = ChannelFrame.from_numpy(np.array([[0.1, 0.2], [0.3, 0.4]]), 44100)
        loader = MagicMock(return_value=mock_frame)

        result = lazy_frame.ensure_loaded(loader)

        assert result is mock_frame
        assert lazy_frame.frame is mock_frame
        assert lazy_frame.is_loaded is True
        assert lazy_frame.load_attempted is True
        loader.assert_called_once_with(file_path)

    def test_ensure_loaded_second_call_uses_cache(self) -> None:
        """Second ensure_loaded call returns cached frame without invoking loader."""
        file_path = Path("/path/to/file.wav")
        lazy_frame: LazyFrame[ChannelFrame] = LazyFrame(file_path)

        mock_frame = ChannelFrame.from_numpy(np.array([[0.1, 0.2], [0.3, 0.4]]), 44100)
        loader = MagicMock(return_value=mock_frame)

        lazy_frame.ensure_loaded(loader)
        loader.reset_mock()
        cached_result = lazy_frame.ensure_loaded(loader)

        assert cached_result is mock_frame
        loader.assert_not_called()

    def test_ensure_loaded_failure_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """Loader failure results in None frame, but marks as loaded."""
        file_path = Path("/path/to/file.wav")
        lazy_frame: LazyFrame[ChannelFrame] = LazyFrame(file_path)

        loader = MagicMock(side_effect=RuntimeError("Failed to load file"))
        result = lazy_frame.ensure_loaded(loader)

        assert result is None
        assert lazy_frame.frame is None
        assert lazy_frame.is_loaded is True  # Load was attempted
        assert lazy_frame.load_attempted is True
        assert any("Failed to load file" in record.message and record.levelname == "ERROR" for record in caplog.records)

    def test_reset_clears_loaded_state(self) -> None:
        """Reset restores LazyFrame to unloaded state."""
        file_path = Path("/path/to/file.wav")
        lazy_frame: LazyFrame[ChannelFrame] = LazyFrame(file_path)

        mock_frame = ChannelFrame.from_numpy(np.array([[0.1, 0.2], [0.3, 0.4]]), 44100)
        loader = MagicMock(return_value=mock_frame)
        lazy_frame.ensure_loaded(loader)

        lazy_frame.reset()

        assert lazy_frame.frame is None
        assert lazy_frame.is_loaded is False
        assert lazy_frame.load_attempted is False


# --- Test FrameDataset (Abstract Base Class) ---


class TestFrameDatasetABC:
    """Test suite for FrameDataset abstract base class — Layer 1: type checking."""

    def test_instantiate_abc_directly_raises_typeerror(self) -> None:
        """FrameDataset ABC cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            FrameDataset("/some/path")

    def test_incomplete_subclass_raises_typeerror(self, tmp_path: Path) -> None:
        """Subclass without _load_file implementation cannot be instantiated."""

        class IncompleteFrameDataset(FrameDataset[ChannelFrame]):
            pass

        with pytest.raises(TypeError, match="abstract class"):
            IncompleteFrameDataset(str(tmp_path))

    def test_complete_subclass_instantiates_successfully(self, tmp_path: Path) -> None:
        """Subclass with _load_file properly implemented can be instantiated."""

        class MinimalFrameDataset(FrameDataset[ChannelFrame]):
            def _load_file(self, file_path: Path) -> ChannelFrame | None:
                return ChannelFrame.from_numpy(np.zeros((2, 10)), 44100)

        dataset = MinimalFrameDataset(str(tmp_path), lazy_loading=True)
        assert dataset is not None


# --- Test ChannelFrameDataset ---


class TestChannelFrameDataset:
    """Test suite for ChannelFrameDataset — init, load, apply, sample, metadata."""

    def test_init_lazy_defers_loading(self, create_test_files: Path) -> None:
        """Lazy initialization defers all file loading."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), recursive=False, lazy_loading=True)

        assert dataset._lazy_loading is True
        assert len(dataset) == 3  # test1.wav, test2.wav, test3.csv
        assert len(dataset._lazy_frames) == 3
        assert all(not lf.is_loaded for lf in dataset._lazy_frames)
        assert all(lf.frame is None for lf in dataset._lazy_frames)
        assert dataset.folder_path == folder_path
        assert dataset.file_extensions == [".wav", ".mp3", ".flac", ".csv"]  # Default extensions
        assert dataset._recursive is False

    def test_init_eager_loads_all_frames(self, create_test_files: Path) -> None:
        """Eager initialization loads all frames immediately."""
        folder_path = create_test_files

        # Initialize with eager loading
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=False)

        assert dataset._lazy_loading is False
        assert len(dataset) == 3

        # Verify all frames are loaded (not None)
        assert all(lf.is_loaded for lf in dataset._lazy_frames)
        assert all(lf.frame is not None for lf in dataset._lazy_frames)

        # Verify all frames are instances of ChannelFrame
        assert all(isinstance(lf.frame, ChannelFrame) for lf in dataset._lazy_frames)

        # Verify frame labels match the expected file names
        file_stems = [p.stem for p in dataset._get_file_paths()]
        frame_labels = [lf.frame.label for lf in dataset._lazy_frames if lf.frame is not None]
        assert frame_labels == file_stems

    def test_init_recursive_finds_subdir_files(self, create_test_files: Path) -> None:
        """Recursive mode discovers files in subdirectories."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), recursive=True, lazy_loading=True)
        assert len(dataset) == 4  # Includes subdir/sub_test.wav
        assert dataset._recursive is True
        assert Path("subdir/sub_test.wav") in [p.relative_to(folder_path) for p in dataset._get_file_paths()]

    def test_init_custom_extensions_filters_files(self, create_test_files: Path) -> None:
        """Custom extensions filter to only matching files."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), file_extensions=[".csv"], lazy_loading=True)
        assert len(dataset) == 1
        assert dataset._lazy_frames[0].file_path.name == "test3.csv"
        assert dataset.file_extensions == [".csv"]

    def test_init_empty_folder_returns_zero_length(self, tmp_path: Path) -> None:
        """Empty folder produces zero-length dataset."""
        dataset = ChannelFrameDataset(str(tmp_path), lazy_loading=True)
        assert len(dataset) == 0
        assert len(dataset._lazy_frames) == 0

    def test_init_nonexistent_folder_raises_error(self) -> None:
        """Non-existent folder raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ChannelFrameDataset("non_existent_folder")

    def test_len_returns_file_count(self, create_test_files: Path) -> None:
        """__len__ returns the number of discovered files."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        assert len(dataset) == 3

    def test_getitem_lazy_triggers_load_and_caches(self, create_test_files: Path) -> None:
        """Lazy __getitem__ triggers load on first access and caches."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Access first item - should trigger load
        with patch.object(dataset, "_load_file", wraps=dataset._load_file) as mock_load:
            frame0 = dataset[0]
            assert isinstance(frame0, ChannelFrame)
            assert frame0.label == "test1"  # From filename stem
            assert dataset._lazy_frames[0].is_loaded is True
            assert dataset._lazy_frames[0].frame is frame0
            mock_load.assert_called_once_with(dataset._lazy_frames[0].file_path)

        # Access again - should use cache
        with patch.object(ChannelFrameDataset, "_load_file") as mock_load_again:
            frame0_cached = dataset[0]
            assert frame0_cached is frame0
            mock_load_again.assert_not_called()

    def test_getitem_out_of_range_raises_indexerror(self, create_test_files: Path) -> None:
        """Out-of-bounds index raises IndexError."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        with pytest.raises(IndexError):
            _ = dataset[3]
        with pytest.raises(IndexError):
            _ = dataset[-1]  # Negative indexing not supported by this logic

    def test_load_file_wav_returns_channel_frame(self, create_test_files: Path) -> None:
        """WAV file loading returns ChannelFrame with correct metadata."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        wav_path = dataset._lazy_frames[0].file_path  # test1.wav

        frame = dataset._load_file(wav_path)
        assert isinstance(frame, ChannelFrame)
        assert frame.label == "test1"
        assert frame.sampling_rate == 16000  # From sample_wav_data
        assert frame.n_channels == 2
        assert isinstance(frame.data, np.ndarray)

    def test_load_file_csv_returns_channel_frame(self, create_test_files: Path) -> None:
        """CSV file loading returns ChannelFrame with correct metadata."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        csv_path = dataset._lazy_frames[2].file_path  # test3.csv

        # Need to explicitly call _load_file as from_file handles CSV specifics
        # We simulate this by calling ChannelFrame.from_file directly here
        frame = ChannelFrame.from_file(csv_path, time_column="time")

        assert isinstance(frame, ChannelFrame)
        assert frame.label == "test3"
        assert frame.sampling_rate == 100  # From sample_csv_data
        assert frame.n_channels == 2  # SensorA, SensorB
        assert frame.labels == ["SensorA", "SensorB"]
        assert isinstance(frame.data, np.ndarray)

    def test_load_file_mismatched_sr_triggers_resampling(
        self, create_test_files: Path, sample_wav_data: tuple[int, NDArrayReal]
    ) -> None:
        """Loading file with mismatched sampling rate triggers resampling."""
        folder_path = create_test_files
        original_sr, _ = sample_wav_data  # Should be 16000
        target_sr = 8000
        dataset = ChannelFrameDataset(str(folder_path), sampling_rate=target_sr, lazy_loading=True)
        wav_path = dataset._lazy_frames[0].file_path

        # Mock ChannelFrame.resampling to check if it's called
        # We don't need the return value if we only check the call itself.
        with patch.object(ChannelFrame, "resampling", autospec=True) as mock_resample:
            # We still need to call _load_file to trigger the resampling call
            try:
                # _load_file might raise an error if the mocked resampling
                # doesn't return something usable by subsequent code within _load_file,
                # but for just checking the call, this might be sufficient.
                # Alternatively, set a minimal return_value if needed.
                mock_resample.return_value = MagicMock()  # Provide a basic mock return
                _ = dataset._load_file(wav_path)
            except Exception:
                # Ignore errors after the mock call if only checking the call
                pass

            # Check if resampling was actually called with the correct target SR
            mock_resample.assert_called_once()
            # Verify the keyword argument explicitly
            args, kwargs = mock_resample.call_args
            assert "target_sr" in kwargs
            assert kwargs["target_sr"] == target_sr
            # Optionally check the instance it was called on had the original SR
            assert isinstance(args[0], ChannelFrame)
            assert args[0].sampling_rate == original_sr

    def test_load_file_corrupted_returns_none_and_logs_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Loading corrupted file returns None and logs error."""
        folder_path = tmp_path
        # Create a corrupted/invalid file
        invalid_file = folder_path / "invalid.wav"
        invalid_file.write_text("this is not a wav file")

        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        result = dataset._ensure_loaded(0)
        assert result is None
        # Check logs for warning from _ensure_loaded
        assert any(
            "Failed to load or initialize file" in record.message and record.levelname == "ERROR"
            for record in caplog.records
        )
        # Ensure the frame is marked as None after failure
        assert dataset._lazy_frames[0].frame is None
        assert dataset._lazy_frames[0].is_loaded is True  # Load was attempted

        # Test _load_all_files catching errors
        caplog.clear()
        dataset_eager = ChannelFrameDataset(str(folder_path), lazy_loading=False)
        assert dataset_eager._lazy_frames[0].frame is None  # Should be None due to load failure
        # In current implementation, error logs are emitted at ERROR level
        assert any("Failed to load or initialize file" in record.message for record in caplog.records)

    def test_apply_lazy_defers_transform(self, create_test_files: Path) -> None:
        """Lazy apply defers transform until item access."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        def dummy_transform(frame: ChannelFrame) -> ChannelFrame:
            # Simple transform: reverse data
            return frame._create_new_instance(data=frame._data[:, ::-1])

        transformed_dataset = dataset.apply(dummy_transform)

        assert isinstance(transformed_dataset, ChannelFrameDataset)
        assert transformed_dataset._lazy_loading is True
        assert transformed_dataset._source_dataset is dataset
        assert transformed_dataset._transform is dummy_transform
        assert len(transformed_dataset) == len(dataset)
        assert all(not lf.is_loaded for lf in transformed_dataset._lazy_frames)  # Still lazy

        # Access item to trigger transformation
        original_frame0 = dataset[0]  # Load original
        transformed_frame0 = transformed_dataset[0]  # Load and transform

        assert isinstance(transformed_frame0, ChannelFrame)
        assert transformed_dataset._lazy_frames[0].is_loaded is True
        assert transformed_dataset._lazy_frames[0].frame is transformed_frame0
        # Verify transformation was applied (compare computed data)
        assert transformed_frame0 is not None
        assert original_frame0 is not None
        np.testing.assert_allclose(
            transformed_frame0.compute(),
            original_frame0.compute()[:, ::-1],
        )

        # Access again, should use cache
        with patch.object(transformed_dataset, "_transform", wraps=transformed_dataset._transform) as mock_transform:
            cached_transformed = transformed_dataset[0]
            assert cached_transformed is transformed_frame0
            mock_transform.assert_not_called()

    def test_apply_chaining_composes_transforms(self, create_test_files: Path) -> None:
        """Chained apply calls compose transforms correctly."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        def transform1(frame: ChannelFrame) -> ChannelFrame:
            return frame._create_new_instance(data=frame._data * 2)

        def transform2(frame: ChannelFrame) -> ChannelFrame:
            return frame._create_new_instance(data=frame._data + 1)

        chained_dataset = dataset.apply(transform1).apply(transform2)

        assert isinstance(chained_dataset, ChannelFrameDataset)
        assert chained_dataset._lazy_loading is True
        assert chained_dataset._transform is transform2
        assert chained_dataset._source_dataset is not None
        source_ds = chained_dataset._source_dataset
        assert source_ds._transform is transform1
        assert source_ds._source_dataset is dataset

        # Trigger computation
        original_frame0 = dataset[0]
        final_frame0 = chained_dataset[0]

        assert original_frame0 is not None
        assert final_frame0 is not None

        expected_data = (original_frame0.compute() * 2) + 1
        np.testing.assert_allclose(final_frame0.compute(), expected_data)

    def test_apply_failing_transform_returns_none_and_logs(
        self, create_test_files: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Failing transform returns None and logs warning."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        def failing_transform(frame: ChannelFrame) -> ChannelFrame:
            raise ValueError("Transform failed!")

        transformed_dataset = dataset.apply(failing_transform)

        # In current implementation, exception is caught and None is returned
        result = transformed_dataset[0]
        assert result is None

        # Check logs for warning
        assert any("Transform failed!" in record.message and record.levelname == "WARNING" for record in caplog.records)
        # Ensure frame is marked as is_loaded but with frame=None
        assert transformed_dataset._lazy_frames[0].frame is None
        assert transformed_dataset._lazy_frames[0].is_loaded is True

    def test_resample_and_trim_via_apply_produces_correct_output(self, create_test_files: Path) -> None:
        """Resample and trim via apply produce correct sampling rate and duration."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        target_sr = 8000

        # Resample
        resampled_ds = dataset.resample(target_sr)
        assert isinstance(resampled_ds, ChannelFrameDataset)
        assert resampled_ds._lazy_loading is True
        assert resampled_ds._source_dataset is dataset
        assert resampled_ds._transform is not None
        # Trigger and check SR
        resampled_frame0 = resampled_ds[0]
        assert resampled_frame0 is not None
        assert resampled_frame0.sampling_rate == target_sr

        # Trim (apply on resampled)
        trimmed_ds = resampled_ds.trim(start=0.1, end=0.5)
        assert isinstance(trimmed_ds, ChannelFrameDataset)
        assert trimmed_ds._lazy_loading is True
        assert trimmed_ds._source_dataset is resampled_ds
        # Trigger and check duration/samples
        trimmed_frame0 = trimmed_ds[0]
        assert trimmed_frame0 is not None
        expected_duration = 0.5 - 0.1
        expected_samples = int(expected_duration * target_sr)
        assert trimmed_frame0.duration == pytest.approx(expected_duration)
        assert trimmed_frame0.n_samples == expected_samples

    def test_stft_lazy_matches_direct_stft(self, create_test_files: Path) -> None:
        """Lazy STFT matches direct ChannelFrame.stft result."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        n_fft = 512
        hop_length = 128

        # create the SpectrogramFrameDataset lazily
        stft_dataset = dataset.stft(n_fft=n_fft, hop_length=hop_length)
        assert isinstance(stft_dataset, SpectrogramFrameDataset)
        assert stft_dataset._lazy_loading is True
        assert stft_dataset._source_dataset is dataset
        assert len(stft_dataset) == len(dataset)
        assert all(not lf.is_loaded for lf in stft_dataset._lazy_frames)

        # compute expected spectrogram from the already‐loaded ChannelFrame
        original_frame = dataset[0]
        assert original_frame is not None
        expected_spec = original_frame.stft(n_fft=n_fft, hop_length=hop_length)

        # now load the spectrogram from the dataset under test
        stft_frame = stft_dataset[0]
        assert isinstance(stft_frame, SpectrogramFrame)
        assert stft_frame.n_fft == n_fft
        assert stft_frame.hop_length == hop_length
        assert stft_frame.sampling_rate == original_frame.sampling_rate

        # compare shapes
        assert stft_frame.shape == expected_spec.shape

        # compare the actual complex STFT values
        np.testing.assert_allclose(stft_frame.compute(), expected_spec.compute(), atol=1e-6)

        # second access must hit the cache and not retrigger transform
        with patch.object(stft_dataset, "_transform", wraps=stft_dataset._transform) as mock_transform:
            cached = stft_dataset[0]
            assert cached is stft_frame
            mock_transform.assert_not_called()

    def test_sample_by_count_returns_correct_size(self, create_test_files: Path) -> None:
        """Sampling by n returns exactly n items from original dataset."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        n_sample = 2
        sampled = dataset.sample(n=n_sample, seed=42)

        assert isinstance(sampled, _SampledFrameDataset)
        assert len(sampled) == n_sample
        assert sampled._original_dataset is dataset
        original_paths = [p.name for p in dataset._get_file_paths()]
        sampled_paths = [lf.file_path.name for lf in sampled._lazy_frames]
        assert all(p in original_paths for p in sampled_paths)

    def test_sample_by_ratio_returns_proportional(self, create_test_files: Path) -> None:
        """Sampling by ratio returns floor(total * ratio) items."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        ratio = 0.5
        sampled = dataset.sample(ratio=ratio, seed=42)

        assert isinstance(sampled, _SampledFrameDataset)
        assert len(sampled) == int(len(dataset) * ratio)

    def test_sample_default_returns_at_least_one(self, create_test_files: Path) -> None:
        """Default sampling (10%) returns at least 1 item."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        sampled = dataset.sample(seed=42)
        assert len(sampled) == max(1, int(len(dataset) * 0.1))

    def test_sample_exceeding_total_caps_at_total(self, create_test_files: Path) -> None:
        """Requesting more samples than available caps at total count."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        n_total = len(dataset)
        sampled = dataset.sample(n=n_total + 1)
        assert len(sampled) == n_total

    def test_sample_empty_dataset_returns_empty(self, create_test_files: Path) -> None:
        """Sampling from empty dataset returns empty result."""
        folder_path = create_test_files
        empty_ds = ChannelFrameDataset(str(folder_path / "empty_subdir"), lazy_loading=True)
        sampled = empty_ds.sample(n=1)
        assert len(sampled) == 0

    def test_sample_preserves_lazy_loading(self, create_test_files: Path) -> None:
        """Sampled dataset preserves lazy loading and frame data matches original."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        sampled = dataset.sample(n=1, seed=1)
        assert isinstance(sampled, _SampledFrameDataset)

        assert sampled._lazy_loading is True
        assert all(not lf.is_loaded for lf in sampled._lazy_frames)

        original_index = sampled._original_indices[0]
        original_frame = dataset[original_index]
        assert original_frame is not None
        sampled_frame = sampled[0]
        assert isinstance(sampled_frame, ChannelFrame)
        assert sampled_frame.label == original_frame.label
        np.testing.assert_array_equal(sampled_frame.compute(), original_frame.compute())

    def test_sample_apply_transform_propagates(self, create_test_files: Path) -> None:
        """Apply on sampled dataset correctly chains transforms."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        sampled = dataset.sample(n=1, seed=1)
        assert isinstance(sampled, _SampledFrameDataset)

        def gain_transform(f: ChannelFrame) -> ChannelFrame:
            return f * 2

        transformed = sampled.apply(gain_transform)
        assert isinstance(transformed, _SampledFrameDataset)

        original_index = sampled._original_indices[0]
        original_frame = dataset[original_index]
        assert original_frame is not None
        final_frame = transformed[0]
        assert final_frame is not None
        expected_data = original_frame.compute() * 2
        np.testing.assert_allclose(final_frame.compute(), expected_data)

    def test_get_metadata_lazy_unloaded_state(self, create_test_files: Path) -> None:
        """Metadata reflects unloaded state for lazy dataset."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True, sampling_rate=8000)

        meta = dataset.get_metadata()
        assert meta["folder_path"] == str(folder_path)
        assert meta["file_count"] == 3
        assert meta["loaded_count"] == 0
        assert meta["target_sampling_rate"] == 8000
        assert meta["actual_sampling_rate"] == 8000  # Inherits target when unloaded
        assert meta["lazy_loading"] is True
        assert meta["has_transform"] is False
        assert meta["frame_type"] == "Unknown"  # Not loaded yet

    def test_get_metadata_after_loading_one_item(self, create_test_files: Path) -> None:
        """Metadata updates after loading a single item."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True, sampling_rate=8000)
        _ = dataset[0]

        meta = dataset.get_metadata()
        assert meta["loaded_count"] == 1
        assert meta["actual_sampling_rate"] == 8000
        assert meta["frame_type"] == "ChannelFrame"

    def test_get_metadata_with_transform(self, create_test_files: Path) -> None:
        """Transformed dataset metadata reflects transform state."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True, sampling_rate=8000)
        transformed = dataset.apply(lambda f: f * 2)

        meta = transformed.get_metadata()
        assert meta["has_transform"] is True
        assert meta["loaded_count"] == 0

        _ = transformed[0]
        meta_loaded = transformed.get_metadata()
        assert meta_loaded["loaded_count"] == 1
        assert meta_loaded["frame_type"] == "ChannelFrame"

    def test_get_metadata_eager_loading(self, create_test_files: Path) -> None:
        """Eager-loaded dataset metadata shows all frames loaded."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=False)

        meta = dataset.get_metadata()
        assert meta["lazy_loading"] is False
        assert meta["loaded_count"] == 3
        assert meta["frame_type"] == "ChannelFrame"

    def test_from_folder_creates_dataset_with_options(self, create_test_files: Path) -> None:
        """From_folder class method creates dataset with specified options."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset.from_folder(
            str(folder_path),
            sampling_rate=8000,
            file_extensions=[".wav"],
            recursive=True,
            lazy_loading=False,
        )
        assert isinstance(dataset, ChannelFrameDataset)
        assert dataset.sampling_rate == 8000
        assert dataset.file_extensions == [".wav"]
        assert dataset._recursive is True
        assert dataset._lazy_loading is False
        assert len(dataset) == 3  # Only wav files, including subdir
        assert all(isinstance(lf.frame, ChannelFrame) for lf in dataset._lazy_frames)

    def test_save_raises_not_implemented(self, create_test_files: Path, tmp_path: Path) -> None:
        """Save method raises NotImplementedError for both dataset types."""
        folder_path = create_test_files
        output_folder = tmp_path / "output"
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # save method is not implemented, verify NotImplementedError is raised
        with pytest.raises(NotImplementedError, match="The save method is not currently implemented."):
            dataset.save(str(output_folder), filename_prefix="processed_")

        # Verify SpectrogramFrameDataset save also raises
        stft_dataset = dataset.stft()
        with pytest.raises(NotImplementedError, match="The save method is not currently implemented."):
            stft_dataset.save(str(output_folder), filename_prefix="spec_")


# --- Test SpectrogramFrameDataset ---


class TestSpectrogramFrameDataset:
    """Test suite for SpectrogramFrameDataset — lazy STFT, apply, plot."""

    def test_init_lazy_via_stft(self, create_test_files: Path) -> None:
        """SpectrogramFrameDataset created via stft() is lazy with correct state."""
        folder_path = create_test_files
        channel_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        stft_ds = channel_ds.stft()  # Creates SpectrogramFrameDataset

        assert isinstance(stft_ds, SpectrogramFrameDataset)
        assert stft_ds._lazy_loading is True
        assert stft_ds._source_dataset is channel_ds
        assert stft_ds._transform is not None
        assert len(stft_ds) == len(channel_ds)
        assert all(not lf.is_loaded for lf in stft_ds._lazy_frames)

    def test_load_file_direct_raises_not_implemented(self, tmp_path: Path) -> None:
        """Direct _load_file raises NotImplementedError."""
        # Create a dummy file that might look like a spectrogram source
        dummy_spec_file = tmp_path / "spec.npy"
        np.save(dummy_spec_file, np.zeros((1, 10, 5)))  # Deterministic zeros instead of random

        # Initialize directly (not the typical way)
        spec_ds = SpectrogramFrameDataset(str(tmp_path), file_extensions=[".npy"])

        # Current implementation: _load_file raises exception,
        # but _ensure_loaded catches it.
        # Call _load_file directly to verify exception
        with pytest.raises(
            NotImplementedError,
            match="No method defined for directly loading SpectrogramFrames",
        ):
            spec_ds._load_file(dummy_spec_file)

        # Via __getitem__, exception is caught and None is returned
        result = spec_ds[0]
        assert result is None

    def test_apply_transform_produces_different_output(self, create_test_files: Path) -> None:
        """Apply on SpectrogramFrameDataset produces transformed output."""
        folder_path = create_test_files
        channel_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        stft_ds = channel_ds.stft()

        def spec_transform(spec_frame: SpectrogramFrame) -> SpectrogramFrame:
            # Simple transform function (amplitude_to_db may not be implemented)
            return spec_frame + 1

        transformed_spec_ds = stft_ds.apply(spec_transform)

        assert isinstance(transformed_spec_ds, SpectrogramFrameDataset)
        assert transformed_spec_ds._lazy_loading is True
        assert transformed_spec_ds._source_dataset is stft_ds
        assert transformed_spec_ds._transform is spec_transform

        # Trigger computation
        original_spec_frame = stft_ds[0]  # Computes STFT
        transformed_spec_frame = transformed_spec_ds[0]  # Computes STFT then transform

        assert original_spec_frame is not None
        assert transformed_spec_frame is not None
        # Verify transformation (check if data changed)
        assert not np.allclose(original_spec_frame.compute(), transformed_spec_frame.compute())

    @patch("matplotlib.pyplot.show")
    def test_plot_delegates_to_frame_plot(
        self,
        mock_show: MagicMock,
        create_test_files: Path,
    ) -> None:
        """Plot() delegates to SpectrogramFrame.plot."""
        folder_path = create_test_files
        channel_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        stft_ds = channel_ds.stft()

        with patch.object(SpectrogramFrame, "plot", return_value=MagicMock(spec=Axes)) as mock_frame_plot:
            stft_ds.plot(0)
            mock_frame_plot.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_transform_error_logs_warning(
        self,
        mock_show: MagicMock,
        create_test_files: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Plot() logs warning when transform fails."""
        folder_path = create_test_files
        channel_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        def failing_stft(frame: ChannelFrame) -> SpectrogramFrame:
            if frame.label == "test1":
                raise ValueError("STFT failed")
            return frame.stft()

        error_spec_ds = SpectrogramFrameDataset(
            folder_path=str(folder_path),
            source_dataset=channel_ds,
            transform=failing_stft,
        )

        with patch.object(SpectrogramFrame, "plot") as mock_plot_error:
            error_spec_ds.plot(0)
            mock_plot_error.assert_not_called()
            assert any("STFT failed" in rec.message for rec in caplog.records if rec.levelname == "WARNING")

    @patch("matplotlib.pyplot.show")
    def test_plot_no_plot_method_logs_warning(
        self,
        mock_show: MagicMock,
        create_test_files: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Plot() logs warning when frame has no plot method."""
        folder_path = create_test_files
        channel_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        stft_ds = channel_ds.stft()

        with patch.object(SpectrogramFrame, "plot", None):
            stft_ds.plot(0)
            assert any("Frame" in rec.message for rec in caplog.records if rec.levelname == "WARNING")

    @patch("matplotlib.pyplot.show")
    def test_plot_runtime_error_logs_error(
        self,
        mock_show: MagicMock,
        create_test_files: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Plot() logs error when plotting raises RuntimeError."""
        folder_path = create_test_files
        channel_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        stft_ds = channel_ds.stft()

        plot_error_msg = "The save method is not currently implemented."
        with patch.object(
            SpectrogramFrame, "plot", side_effect=RuntimeError(plot_error_msg)
        ) as mock_plot_runtime_error:
            stft_ds.plot(1)
            mock_plot_runtime_error.assert_called_once()
            assert any(plot_error_msg in rec.message for rec in caplog.records if rec.levelname == "ERROR")


# --- Test _SampledFrameDataset (Internal Class) ---


class TestSampledFrameDataset:
    """Test suite for _SampledFrameDataset — init, getitem, apply, load."""

    def test_init_creates_lazy_subset(self, create_test_files: Path) -> None:
        """Initialization creates lazy subset from specified indices."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Prepare sampling indices
        sampled_indices = [0, 2]  # Select test1.wav and test3.csv

        # Initialize _SampledFrameDataset
        sampled_ds = _SampledFrameDataset(dataset, sampled_indices)

        # Verify basic properties
        assert sampled_ds._original_dataset is dataset
        assert sampled_ds._original_indices == sampled_indices
        assert len(sampled_ds) == len(sampled_indices)

        # Verify LazyFrames
        assert len(sampled_ds._lazy_frames) == len(sampled_indices)
        assert all(not lf.is_loaded for lf in sampled_ds._lazy_frames)
        assert sampled_ds._lazy_frames[0].file_path.name == "test1.wav"
        assert sampled_ds._lazy_frames[1].file_path.name == "test3.csv"

    def test_getitem_loads_frame_matching_original(self, create_test_files: Path) -> None:
        """__getitem__ loads frame identical to original dataset's frame."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        sampled_indices = [0, 2]  # test1.wav, test3.csv
        sampled_ds = _SampledFrameDataset(dataset, sampled_indices)

        sampled_frame_0 = sampled_ds[0]
        original_frame_0 = dataset[sampled_indices[0]]

        assert sampled_frame_0 is not None
        assert original_frame_0 is not None
        assert isinstance(sampled_frame_0, ChannelFrame)
        assert sampled_frame_0.label == original_frame_0.label
        np.testing.assert_array_equal(
            sampled_frame_0.compute(),
            original_frame_0.compute(),
            err_msg="Sampled dataset frame does not match original",
        )

    def test_getitem_caches_loaded_frame(self, create_test_files: Path) -> None:
        """__getitem__ returns cached instance on second access."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        sampled_ds = _SampledFrameDataset(dataset, [0, 2])

        first_access = sampled_ds[0]
        assert sampled_ds._lazy_frames[0].is_loaded is True
        second_access = sampled_ds[0]
        assert second_access is first_access

    def test_getitem_multiple_indices_load_correctly(self, create_test_files: Path) -> None:
        """__getitem__ loads correct frames for all sampled indices."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        sampled_indices = [0, 2]
        sampled_ds = _SampledFrameDataset(dataset, sampled_indices)

        sampled_frame_1 = sampled_ds[1]
        original_frame_1 = dataset[sampled_indices[1]]

        assert sampled_frame_1 is not None
        assert original_frame_1 is not None
        assert isinstance(sampled_frame_1, ChannelFrame)
        assert sampled_frame_1.label == original_frame_1.label
        np.testing.assert_array_equal(sampled_frame_1.compute(), original_frame_1.compute())

    def test_getitem_out_of_range_raises_indexerror(self, create_test_files: Path) -> None:
        """__getitem__ with out-of-range index raises IndexError."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        sampled_ds = _SampledFrameDataset(dataset, [0, 2])

        with pytest.raises(IndexError):
            _ = sampled_ds[2]  # Sampled dataset has only 2 elements

    def test_getitem_original_load_failure_returns_none(self, create_test_files: Path) -> None:
        """__getitem__ returns None when original dataset fails to load."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        sampled_ds = _SampledFrameDataset(dataset, [0, 2])
        _ = sampled_ds[0]  # Load first to populate cache

        with patch.object(dataset, "_ensure_loaded", return_value=None):
            sampled_ds._lazy_frames[0].reset()
            result = sampled_ds[0]
            assert result is None

    def test_apply_chains_transform_correctly(self, create_test_files: Path) -> None:
        """Apply on sampled dataset chains transform correctly."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        sampled_ds = dataset.sample(n=1, seed=42)

        def transform_gain(f: ChannelFrame) -> ChannelFrame:
            assert f is not None, "Input frame to transform_gain is None"
            return f * 3

        transformed_sampled = sampled_ds.apply(transform_gain)

        # Basic verification
        assert isinstance(transformed_sampled, _SampledFrameDataset)
        assert transformed_sampled._original_dataset is not dataset
        assert transformed_sampled._original_dataset._transform is transform_gain
        assert transformed_sampled._original_dataset._source_dataset is dataset
        assert len(transformed_sampled) == 1
        assert all(not lf.is_loaded for lf in transformed_sampled._lazy_frames)

        # Get original frame
        original_index = getattr(sampled_ds, "_original_indices")[0]
        original_frame = dataset[original_index]

        # Get sampled frame
        sampled_frame = sampled_ds[0]
        assert original_frame is not None
        assert sampled_frame is not None
        np.testing.assert_allclose(
            sampled_frame.compute(),
            original_frame.compute(),
            err_msg="Data mismatch between dataset and sampled_ds",
        )

        # Get and verify transformed frame
        final_frame = transformed_sampled[0]
        assert final_frame is not None
        assert original_frame is not None
        expected_data = original_frame.compute() * 3
        np.testing.assert_allclose(
            final_frame.compute(),
            expected_data,
            err_msg="Data mismatch after applying transform",
        )

        # Verify cache is working
        cached_final_frame = transformed_sampled[0]
        assert cached_final_frame is final_frame

    def test_load_file_direct_raises_not_implemented(self, create_test_files: Path) -> None:
        """Direct _load_file raises NotImplementedError."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        sampled_indices = [0, 1]
        sampled_ds = _SampledFrameDataset(dataset, sampled_indices)

        with pytest.raises(
            NotImplementedError,
            match="_SampledFrameDataset does not load files directly.",
        ):
            sampled_ds._load_file(Path("dummy.wav"))

    def test_init_invalid_indices_raises_index_error(self, create_test_files: Path) -> None:
        """Out-of-range indices raise IndexError."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Out-of-range index case
        invalid_indices = [0, 10]  # 10 is out of range (only 3 files)

        with pytest.raises(
            IndexError,
            match="Indices are out of range for the original dataset.",
        ):
            _SampledFrameDataset(dataset, invalid_indices)


class TestLazyFrameEdgeCases:
    """Additional edge case tests for LazyFrame."""

    def test_ensure_loaded_already_loaded_without_load_attempt(self) -> None:
        """Ensure_loaded returns cached frame when is_loaded=True but load_attempted=False.

        This exercises the early return path in LazyFrame.ensure_loaded when the frame is
        already loaded but no load has been attempted yet.
        """
        file_path = Path("/path/to/file.wav")
        lazy_frame: LazyFrame[ChannelFrame] = LazyFrame(file_path)

        # Simulate external loading - is_loaded=True but load_attempted=False
        mock_frame = ChannelFrame.from_numpy(np.array([[0.1, 0.2], [0.3, 0.4]]), 44100)
        lazy_frame.frame = mock_frame
        lazy_frame.is_loaded = True
        # load_attempted remains False

        loader = MagicMock(return_value=mock_frame)

        # ensure_loaded should return cached frame without calling loader
        result = lazy_frame.ensure_loaded(loader)

        assert result is mock_frame
        assert lazy_frame.frame is mock_frame
        assert lazy_frame.is_loaded is True
        assert lazy_frame.load_attempted is False  # Should remain unchanged
        loader.assert_not_called()


class TestFrameDatasetGetByLabel:
    """Tests for get_by_label method (deprecated)."""

    def test_get_by_label_deprecation_warning(self, create_test_files: Path) -> None:
        """Get_by_label emits DeprecationWarning and returns first match."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Access the frame to ensure it's loaded
        _ = dataset[0]

        with pytest.warns(DeprecationWarning, match="get_by_label.*deprecated"):
            result = dataset.get_by_label("test1.wav")

        assert isinstance(result, ChannelFrame)
        assert result.label == "test1"

    def test_get_by_label_no_match(self, create_test_files: Path) -> None:
        """Get_by_label returns None when no match found."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # No warning should be emitted when no matches
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = dataset.get_by_label("nonexistent.wav")

        assert result is None
        assert len(w) == 0


class TestFrameDatasetGetItemStringKey:
    """Tests for __getitem__ with string key."""

    def test_getitem_string_key_single_match(self, create_test_files: Path) -> None:
        """__getitem__ with string key returns list of matches."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # String key should return a list via get_all_by_label
        frames = dataset["test1.wav"]
        assert isinstance(frames, list)
        assert len(frames) == 1
        assert isinstance(frames[0], ChannelFrame)
        assert frames[0].label == "test1"

    def test_getitem_string_key_multiple_matches(self, tmp_path: Path) -> None:
        """__getitem__ with string key returns multiple matches for duplicate filenames."""
        # Create a directory structure with duplicate filenames in different subfolders
        root = tmp_path / "multi_match_dataset"
        sub1 = root / "sub1"
        sub2 = root / "sub2"
        sub1.mkdir(parents=True, exist_ok=True)
        sub2.mkdir(parents=True, exist_ok=True)

        samplerate = 8000
        data = np.zeros(samplerate, dtype=np.float32)

        file_name = "duplicate.wav"
        sf.write(sub1 / file_name, data, samplerate)
        sf.write(sub2 / file_name, data, samplerate)

        dataset = ChannelFrameDataset(str(root), lazy_loading=True, recursive=True)

        # String key should return all matches with the same Path.name across subfolders
        frames = dataset[file_name]
        assert isinstance(frames, list)
        assert len(frames) == 2
        assert all(isinstance(f, ChannelFrame) for f in frames)


class TestFrameDatasetInitializeFromSource:
    """Tests for _initialize_from_source property inheritance."""

    def test_initialize_from_source_property_inheritance(self, create_test_files: Path) -> None:
        """Properties are inherited from source_dataset when not explicitly provided.

        Properties are conditionally inherited.
        """
        folder_path = create_test_files

        # Create base dataset with specific settings
        base_ds = ChannelFrameDataset(
            str(folder_path),
            sampling_rate=8000,
            signal_length=16000,
            lazy_loading=True,
            recursive=False,
        )

        # Apply transform - this creates a new dataset with source_dataset set
        transformed_ds = base_ds.apply(lambda f: f)

        # Verify inheritance when not explicitly provided in apply()
        assert transformed_ds.sampling_rate == 8000  # Inherited from base
        assert transformed_ds.signal_length == 16000  # Inherited from base
        assert transformed_ds._recursive == base_ds._recursive
        assert transformed_ds.folder_path == base_ds.folder_path

    def test_initialize_from_source_explicit_override(self, create_test_files: Path) -> None:
        """Explicit parameters override inherited values."""
        folder_path = create_test_files

        # Create base dataset with specific settings
        base_ds = ChannelFrameDataset(
            str(folder_path),
            sampling_rate=8000,
            signal_length=16000,
            lazy_loading=True,
        )

        # Create transformed dataset with explicit override
        def identity_transform(f: ChannelFrame) -> ChannelFrame:
            return f

        transformed_ds = ChannelFrameDataset(
            str(folder_path),
            sampling_rate=16000,  # Override base's 8000
            signal_length=None,  # Should inherit from base (16000)
            lazy_loading=True,
            source_dataset=base_ds,
            transform=identity_transform,
        )

        # Verify explicit override is used
        assert transformed_ds.sampling_rate == 16000  # Explicit value overrides inherited

        # Verify inheritance when not explicitly provided (None)
        assert transformed_ds.signal_length == 16000  # Inherited from base


class TestFrameDatasetSampleEdgeCases:
    """Tests for sample() method edge cases."""

    def test_sample_ratio_greater_than_one(self, create_test_files: Path) -> None:
        """Sample with ratio > 1.0 caps at total.

        N is capped at total.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        n_total = len(dataset)

        # Ratio > 1 should cap at total
        sampled = dataset.sample(ratio=2.0, seed=42)
        assert isinstance(sampled, _SampledFrameDataset)
        assert len(sampled) == n_total  # Should be capped at total

    def test_sample_ratio_exactly_one(self, create_test_files: Path) -> None:
        """Sample with ratio = 1.0 returns all items."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        n_total = len(dataset)

        sampled = dataset.sample(ratio=1.0, seed=42)
        assert isinstance(sampled, _SampledFrameDataset)
        assert len(sampled) == n_total

    def test_sample_ratio_small_value(self, create_test_files: Path) -> None:
        """Sample with small ratio returns at least 1 item."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Very small ratio should still return at least 1
        sampled = dataset.sample(ratio=0.001, seed=42)
        assert isinstance(sampled, _SampledFrameDataset)
        assert len(sampled) >= 1


class TestSpectrogramFrameDatasetPlotEdgeCases:
    """Tests for SpectrogramFrameDataset.plot() edge cases."""

    def test_plot_when_frame_is_none(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Plot returns early when _ensure_loaded returns None.

        We check if frame is None.
        """
        folder_path = create_test_files
        channel_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        def failing_transform(frame: ChannelFrame) -> SpectrogramFrame | None:
            raise ValueError("Transform failed")

        spec_ds = SpectrogramFrameDataset(
            str(folder_path),
            source_dataset=channel_ds,
            transform=failing_transform,
        )

        with caplog.at_level(logging.WARNING):
            spec_ds.plot(0)  # Should log warning and return early

        # Verify warning was logged
        assert any("failed to load/transform" in rec.message for rec in caplog.records if rec.levelname == "WARNING")

    def test_plot_when_frame_has_no_plot_method(
        self, create_test_files: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Plot logs warning when frame has no plot method.

        We check for plot method.
        """
        folder_path = create_test_files
        channel_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        spec_ds = channel_ds.stft()

        # Mock the plot method to be None (simulate missing plot method)
        with patch.object(SpectrogramFrame, "plot", None):
            with caplog.at_level(logging.WARNING):
                spec_ds.plot(0)  # Should log warning about missing plot method

            assert any(
                "does not have a plot method" in rec.message for rec in caplog.records if rec.levelname == "WARNING"
            )


class TestGetMetadataEdgeCases:
    """Tests for get_metadata() edge cases."""

    def test_get_metadata_with_sampled_dataset(self, create_test_files: Path) -> None:
        """Get_metadata correctly identifies sampled dataset.

        Is_sampled is checked.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        sampled_ds = dataset.sample(n=2, seed=42)

        meta = sampled_ds.get_metadata()
        assert meta["is_sampled"] is True
        assert isinstance(sampled_ds, _SampledFrameDataset)


class TestChannelFrameDatasetTransformEdgeCases:
    """Tests for ChannelFrameDataset transform methods edge cases."""

    def test_resample_with_none_frame(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Resample handles None frames gracefully.

        We check if frame is None.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a transform that returns None for some frames
        def partial_fail_transform(frame: ChannelFrame) -> ChannelFrame | None:
            if frame.label == "test1":
                return None  # Simulate failure
            return frame.resampling(target_sr=8000)

        resampled_ds = dataset.apply(partial_fail_transform)

        result = resampled_ds[0]  # test1.wav - should be None
        assert result is None

    def test_trim_with_none_frame(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Trim handles None frames gracefully.

        We check if frame is None.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a transform that returns None for some frames
        def partial_fail_transform(frame: ChannelFrame) -> ChannelFrame | None:
            if frame.label == "test1":
                return None  # Simulate failure
            return frame.trim(start=0.1, end=0.5)

        trimmed_ds = dataset.apply(partial_fail_transform)

        result = trimmed_ds[0]  # test1.wav - should be None
        assert result is None

    def test_normalize_with_none_frame(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Normalize handles None frames gracefully.

        We check if frame is None.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a transform that returns None for some frames
        def partial_fail_transform(frame: ChannelFrame) -> ChannelFrame | None:
            if frame.label == "test1":
                return None  # Simulate failure
            return frame.normalize()

        normalized_ds = dataset.apply(partial_fail_transform)

        result = normalized_ds[0]  # test1.wav - should be None
        assert result is None


class TestFrameDatasetGetItemTypeError:
    """Tests for __getitem__ TypeError case."""

    def test_getitem_invalid_type_raises_error(self, create_test_files: Path) -> None:
        """__getitem__ raises TypeError for invalid key types.

        We raise TypeError.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Test with float key
        with pytest.raises(TypeError, match="Invalid key type.*float"):
            _ = dataset[1.5]  # ty: ignore[invalid-argument-type]

        # Test with tuple key
        with pytest.raises(TypeError, match="Invalid key type.*tuple"):
            _ = dataset[(0, 1)]  # ty: ignore[invalid-argument-type]


class TestFrameDatasetInitializeFromSourceEdgeCases:
    """Tests for _initialize_from_source() edge cases."""

    def test_initialize_from_source_with_none_source(self, tmp_path: Path) -> None:
        """_initialize_from_source returns early when source_dataset is None.

        We return early if no source.
        """
        # Create an empty directory with no matching files
        folder_path = tmp_path / "empty"
        folder_path.mkdir()

        dataset = ChannelFrameDataset(
            str(folder_path),
            lazy_loading=True,
            source_dataset=None,  # Explicitly None
        )

        # _initialize_from_source should have returned early without doing anything
        assert len(dataset._lazy_frames) == 0


class TestFrameDatasetLoadAllFilesErrorHandling:
    """Tests for _load_all_files() error handling during eager loading."""

    def test_load_all_files_error_handling(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Errors during eager loading are caught and logged.

        Errors during eager loading are caught and logged as warnings while
        continuing to process other files.
        Patches _ensure_loaded to raise for the first index while allowing others to proceed.
        """
        sr = 8000
        t = np.linspace(0, 0.01, int(sr * 0.01), endpoint=False)
        signal = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        file1 = tmp_path / "file1.wav"
        file2 = tmp_path / "file2.wav"
        sf.write(file1, signal, sr)
        sf.write(file2, signal, sr)

        original_ensure_loaded = ChannelFrameDataset._ensure_loaded

        def _ensure_loaded_side_effect(self: ChannelFrameDataset, index: int) -> None:
            if index == 0:
                raise RuntimeError("Simulated load error for index 0")
            original_ensure_loaded(self, index)

        with patch.object(
            ChannelFrameDataset,
            "_ensure_loaded",
            autospec=True,
            side_effect=_ensure_loaded_side_effect,
        ):
            with caplog.at_level(logging.WARNING, logger="wandas.utils.frame_dataset"):
                dataset = ChannelFrameDataset(str(tmp_path), lazy_loading=False)

        warning_records = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and record.name == "wandas.utils.frame_dataset"
        ]
        assert warning_records, "Expected a warning log from _load_all_files error handling"

        assert len(dataset._lazy_frames) == 2
        loaded_frames = [lf for lf in dataset._lazy_frames if lf.frame is not None]
        assert loaded_frames, "Expected at least one successfully loaded frame despite an earlier error"


class TestFrameDatasetLoadFromSourceEdgeCases:
    """Tests for _load_from_source() edge cases."""

    def test_load_from_source_no_transform(self, create_test_files: Path) -> None:
        """_load_from_source returns None when no transform is set.

        We return None if no transform.
        """
        folder_path = create_test_files
        channel_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a SpectrogramFrameDataset without a transform (edge case)
        spec_ds = SpectrogramFrameDataset(
            str(folder_path),
            source_dataset=channel_ds,
            transform=None,  # No transform
        )

        result = spec_ds._load_from_source(0)
        assert result is None

    def test_load_from_source_no_source_frame(self, create_test_files: Path) -> None:
        """_load_from_source returns None when source frame is None.

        We return None if source frame failed to load.
        """
        folder_path = create_test_files
        channel_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a transform that fails on the first file
        def failing_transform(frame: ChannelFrame) -> SpectrogramFrame | None:
            raise ValueError("Transform failed")

        spec_ds = SpectrogramFrameDataset(
            str(folder_path),
            source_dataset=channel_ds,
            transform=failing_transform,
        )

        result = spec_ds._load_from_source(0)
        assert result is None

    def test_load_from_source_success(self, create_test_files: Path) -> None:
        """_load_from_source succeeds when conditions are met.

        This covers the success path.
        Uses ChannelFrameDataset so that the identity transform returning a ChannelFrame
        matches the expected frame type.
        """
        folder_path = create_test_files
        source_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        def identity_transform(frame: ChannelFrame) -> ChannelFrame | None:
            return frame  # Return the same frame

        target_ds = ChannelFrameDataset(
            str(folder_path),
            source_dataset=source_ds,
            transform=identity_transform,
        )

        result = target_ds._load_from_source(0)
        assert result is not None
        assert isinstance(result, ChannelFrame)


class TestFrameDatasetEnsureLoadedExceptionHandling:
    """Tests for _ensure_loaded() exception handling."""

    def test_ensure_loaded_exception_handling(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Exceptions in _ensure_loaded are caught and logged.

        We handle exceptions.
        """
        # Create a corrupted file that will fail to load
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_text("this is not a wav file")

        with caplog.at_level(logging.ERROR, logger="wandas.utils.frame_dataset"):
            dataset = ChannelFrameDataset(str(tmp_path), lazy_loading=True)

            # Access the frame - should handle exception gracefully
            result = dataset._ensure_loaded(0)
            assert result is None

        # Verify the frame state was updated correctly
        assert dataset._lazy_frames[0].frame is None
        assert dataset._lazy_frames[0].is_loaded is True
        assert dataset._lazy_frames[0].load_attempted is True

        assert any(
            record.name == "wandas.utils.frame_dataset" and record.levelno >= logging.ERROR for record in caplog.records
        )


class TestFrameDatasetSampleElseBranch:
    """Tests for sample() method else branch."""

    def test_sample_else_branch_n_and_ratio_both_none(self, create_test_files: Path) -> None:
        """Sample default behavior when both n and ratio are None.

        We use the default formula.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        total = len(dataset)

        # When both n and ratio are None, should use: max(1, min(10, int(total * 0.1)))
        sampled = dataset.sample(seed=42)
        expected_n = max(1, min(10, int(total * 0.1)))
        assert len(sampled) == expected_n

    def test_sample_else_branch_explicit_values(self, create_test_files: Path) -> None:
        """Sample with both n and ratio provided uses the elif branch.

        When both n and ratio are provided, the first matching condition (n is not None) is used.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # When n is provided, it should be used regardless of ratio value
        sampled = dataset.sample(n=5, ratio=0.1, seed=42)  # ratio is ignored since n is set
        assert len(sampled) == min(5, len(dataset))


class TestGetMetadataExceptionHandling:
    """Tests for get_metadata() exception handling."""

    def test_get_metadata_exception_on_first_frame(
        self, create_test_files: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exceptions accessing first frame are caught and logged.

        We catch exceptions.
        We inject a frame whose sampling_rate access raises to trigger the warning path.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # First access the frame to mark it as loaded
        _ = dataset[0]

        class BadFrame:
            """Stub that deliberately raises on sampling_rate to trigger exception handling."""

            @property
            def sampling_rate(self) -> float:
                raise RuntimeError("sampling_rate access failed")

        dataset._lazy_frames[0].frame = BadFrame()  # ty: ignore[invalid-assignment]

        with caplog.at_level(logging.WARNING, logger="wandas.utils.frame_dataset"):
            meta = dataset.get_metadata()

        assert isinstance(meta, dict)
        assert "folder_path" in meta

        warning_records = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and record.name == "wandas.utils.frame_dataset"
        ]
        assert warning_records


class TestSampledFrameDatasetEnsureLoadedExceptionHandling:
    """Tests for _SampledFrameDataset._ensure_loaded() exception handling."""

    def test_sampled_ensure_loaded_exception_handling(
        self, create_test_files: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exceptions in sampled dataset loading are caught and logged.

        We handle exceptions.
        We mock the class method to ensure the exception is raised during access.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a sampled dataset normally first
        sampled_ds = dataset.sample(n=2, seed=42)

        with caplog.at_level(logging.ERROR, logger="wandas.utils.frame_dataset"):
            with patch.object(type(dataset), "__getitem__", side_effect=RuntimeError("Access error")):
                # Reset the lazy frame state before triggering load
                sampled_ds._lazy_frames[0].reset()

                result = sampled_ds._ensure_loaded(0)
                assert result is None

                # Verify the frame state was updated correctly due to exception handling
                assert sampled_ds._lazy_frames[0].frame is None
                assert sampled_ds._lazy_frames[0].is_loaded is True
                assert sampled_ds._lazy_frames[0].load_attempted is True


class TestChannelFrameDatasetTransformNoneChecks:
    """Tests for ChannelFrameDataset transform methods handling None frames."""

    def test_resample_none_frame_in_transform(self, create_test_files: Path) -> None:
        """Resample handles None frame input correctly.

        We check if frame is None.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a custom transform that returns None for the first frame
        def conditional_resample(frame: ChannelFrame) -> ChannelFrame | None:
            if frame.label == "test1":
                return None  # Return None for this specific file
            return frame.resampling(target_sr=8000)

        resampled_ds = dataset.apply(conditional_resample)

        result = resampled_ds[0]  # test1.wav - should be None due to transform returning None
        assert result is None

    def test_stft_exception_in_transform(self, create_test_files: Path) -> None:
        """STFT handles exceptions in the transform function.

        We catch exceptions and return None.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a custom STFT that fails on specific frames
        def failing_stft(frame: ChannelFrame) -> SpectrogramFrame | None:
            if frame.label == "test1":
                raise RuntimeError("STFT failed")
            return frame.stft()

        spec_ds = SpectrogramFrameDataset(
            str(folder_path),
            source_dataset=dataset,
            transform=failing_stft,
        )

        result = spec_ds[0]  # test1.wav - should be None due to exception
        assert result is None

        # Verify the frame state was updated correctly
        assert spec_ds._lazy_frames[0].frame is None
        assert spec_ds._lazy_frames[0].is_loaded is True


class TestChannelFrameDatasetTrimNoneCheck:
    """Tests for ChannelFrameDataset.trim() handling of None frames."""

    def test_trim_none_frame_in_transform(self, create_test_files: Path) -> None:
        """Trim handles None frame input correctly.

        We check if frame is None.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a custom transform that returns None for the first frame
        def conditional_trim(frame: ChannelFrame) -> ChannelFrame | None:
            if frame.label == "test1":
                return None  # Return None for this specific file
            return frame.trim(start=0.1, end=0.5)

        trimmed_ds = dataset.apply(conditional_trim)

        result = trimmed_ds[0]  # test1.wav - should be None due to transform returning None
        assert result is None


class TestChannelFrameDatasetNormalizeNoneCheck:
    """Tests for ChannelFrameDataset.normalize() handling of None frames."""

    def test_normalize_none_frame_in_transform(self, create_test_files: Path) -> None:
        """Normalize handles None frame input correctly.

        We check if frame is None.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a custom transform that returns None for the first frame
        def conditional_normalize(frame: ChannelFrame) -> ChannelFrame | None:
            if frame.label == "test1":
                return None  # Return None for this specific file
            return frame.normalize()

        normalized_ds = dataset.apply(conditional_normalize)

        result = normalized_ds[0]  # test1.wav - should be None due to transform returning None
        assert result is None


class TestChannelFrameDatasetStftNoneCheck:
    """Tests for ChannelFrameDataset.stft() handling of None frames."""

    def test_stft_none_frame_in_transform(self, create_test_files: Path) -> None:
        """Stft handles None frame input correctly.

        We check if frame is None.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a custom transform that returns None for the first frame
        def conditional_stft(frame: ChannelFrame) -> SpectrogramFrame | None:
            if frame.label == "test1":
                return None  # Return None for this specific file
            return frame.stft()

        spec_ds = SpectrogramFrameDataset(
            str(folder_path),
            source_dataset=dataset,
            transform=conditional_stft,
        )

        result = spec_ds[0]  # test1.wav - should be None due to transform returning None
        assert result is None


class TestGetMetadataExceptionPath:
    """Tests for get_metadata() exception path."""

    def test_get_metadata_exception_path(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Exceptions accessing first frame trigger the warning log.

        We catch exceptions and log warnings.
        We inject a frame whose sampling_rate access raises to deterministically trigger the path.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Load the first frame so it is marked as loaded
        real_frame = dataset[0]

        class FaultyFrame:
            """Proxy that delegates most attributes to a real frame but
            raises RuntimeError when sampling_rate is accessed, to trigger the
            exception path in FrameDataset.get_metadata."""

            def __init__(self, base: ChannelFrame) -> None:
                self._base = base

            @property
            def sampling_rate(self) -> float:
                raise RuntimeError("Failed to access sampling_rate")

            def __getattr__(self, name: str):
                return getattr(self._base, name)

        dataset._lazy_frames[0].frame = FaultyFrame(real_frame)  # ty: ignore[invalid-assignment, invalid-argument-type]

        with caplog.at_level(logging.WARNING, logger="wandas.utils.frame_dataset"):
            meta = dataset.get_metadata()

        assert any(record.levelno == logging.WARNING for record in caplog.records)
        assert isinstance(meta, dict)
        assert "folder_path" in meta
        assert meta["loaded_count"] >= 1


class TestFrameDatasetLoadAllFilesEdgeCases:
    """Tests for _load_all_files() edge cases."""

    def test_load_all_files_exception_handling(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Errors during eager loading are caught and logged.

        Errors during eager loading are caught and logged as warnings while
        continuing to process other files.
        Patches _ensure_loaded to raise for a specific index to reliably trigger
        the outer except block in _load_all_files.
        """
        sample_rate = 8000
        duration_samples = sample_rate // 10
        data = np.zeros(duration_samples, dtype=np.float32)

        num_files = 3
        for i in range(num_files):
            sf.write(tmp_path / f"file_{i}.wav", data, sample_rate)

        original_ensure_loaded = ChannelFrameDataset._ensure_loaded

        def _ensure_loaded_side_effect(self: ChannelFrameDataset, index: int) -> None:
            if index == 1:
                raise RuntimeError("synthetic load error for testing")
            original_ensure_loaded(self, index)

        with patch.object(
            ChannelFrameDataset,
            "_ensure_loaded",
            side_effect=_ensure_loaded_side_effect,
            autospec=True,
        ):
            with caplog.at_level(logging.WARNING, logger="wandas.utils.frame_dataset"):
                dataset = ChannelFrameDataset(str(tmp_path), lazy_loading=False)

        assert len(dataset._lazy_frames) == num_files

        warnings_from_logger = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and record.name == "wandas.utils.frame_dataset"
        ]
        assert warnings_from_logger, "Expected a warning from _load_all_files error handling"


class TestSampleElseBranch:
    """Tests for sample() method when both n and ratio are provided."""

    def test_sample_n_takes_precedence_over_ratio(self, create_test_files: Path) -> None:
        """When both n and ratio are provided, n is used and ratio is ignored.

        This indirectly confirms that the elif `n is not None` branch is taken
        and the final else branch is not used in this scenario.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # When both n and ratio are provided, the elif n is not None branch is used
        sampled = dataset.sample(n=2, ratio=0.5, seed=42)
        assert len(sampled) == 2  # Uses n value, not ratio


# --- Test ChannelFrameDataset Normalize Edge Cases ---


class TestChannelFrameDatasetNormalizeEdgeCases:
    """Tests for normalize() method edge cases and exception handling."""

    def test_normalize_with_various_kwargs(self, create_test_files: Path) -> None:
        """Normalize with different normalization kwargs.

        Normalize handles various kwargs.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Test with norm='max' (default)
        normalized_ds = dataset.normalize(norm=np.inf)
        result = normalized_ds[0]  # Should load and normalize successfully

        assert result is not None
        assert isinstance(result, ChannelFrame)

    def test_normalize_with_l2_norm(self, create_test_files: Path) -> None:
        """Normalize with L2 norm."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        normalized_ds = dataset.normalize(norm=2)
        result = normalized_ds[0]

        assert result is not None
        assert isinstance(result, ChannelFrame)

    def test_normalize_with_axis_parameter(self, create_test_files: Path) -> None:
        """Normalize with axis parameter."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        normalized_ds = dataset.normalize(norm=np.inf, axis=0)
        result = normalized_ds[0]

        assert result is not None
        assert isinstance(result, ChannelFrame)

    def test_normalize_exception_in_transform(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Normalize handles exceptions in frame.normalize() gracefully.

        Normalization errors are caught.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a transform that raises an exception for testing
        def failing_normalize(frame: ChannelFrame) -> ChannelFrame | None:
            if frame is not None:
                raise ValueError("Simulated normalization error")
            return None

        normalized_ds = dataset.apply(failing_normalize)
        result = normalized_ds[0]

        assert result is None  # Transform failed, returns None
        assert any(
            "Failed to transform" in record.message and record.levelname == "WARNING" for record in caplog.records
        )

    def test_normalize_kwargs_logging(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Normalize logs kwargs when an error occurs.

        Kwargs are included in the log message.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Known kwargs to verify they appear in the log message
        kwargs = {"norm": 2, "axis": 1}

        caplog.set_level(logging.WARNING)

        with patch.object(ChannelFrame, "normalize", side_effect=ValueError("Test error")):
            normalized_ds = dataset.normalize(**kwargs)
            _ = normalized_ds[0]  # Trigger normalization and error handling

        assert any(
            "Normalization error" in record.message and "norm" in record.message and record.levelname == "WARNING"
            for record in caplog.records
        )


# --- Test ChannelFrameDataset Resample Edge Cases ---


class TestChannelFrameDatasetResampleEdgeCases:
    """Tests for resample() method edge cases and exception handling."""

    def test_resample_with_valid_target_sr(self, create_test_files: Path) -> None:
        """Resample with a valid target sampling rate.

        Resample handles normal operation.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Resample to 22050 Hz (valid rate)
        resampled_ds = dataset.resample(target_sr=22050)
        result = resampled_ds[0]

        assert result is not None
        assert isinstance(result, ChannelFrame)
        assert result.sampling_rate == 22050

    def test_resample_with_different_target_sr(self, create_test_files: Path) -> None:
        """Resample with various target sampling rates."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Test multiple target sample rates
        for target_sr in [8000, 16000, 44100, 48000]:
            resampled_ds = dataset.resample(target_sr=target_sr)
            result = resampled_ds[0]

            assert result is not None
            assert isinstance(result, ChannelFrame)
            assert result.sampling_rate == target_sr

    def test_resample_exception_in_transform(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Resample handles exceptions in frame.resampling() gracefully.

        Resampling errors are caught.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a transform that raises an exception for testing
        def failing_resample(frame: ChannelFrame) -> ChannelFrame | None:
            if frame is not None:
                raise ValueError("Simulated resampling error")
            return None

        resampled_ds = dataset.apply(failing_resample)
        result = resampled_ds[0]

        assert result is None  # Transform failed, returns None
        assert any(
            "Failed to transform" in record.message and record.levelname == "WARNING" for record in caplog.records
        )

    def test_resample_with_zero_target_sr(self, create_test_files: Path) -> None:
        """Resample with invalid target_sr=0.

        An invalid target_sr causes an error.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Resample to invalid rate (0) - should fail gracefully
        resampled_ds = dataset.resample(target_sr=0)
        result = resampled_ds[0]

        assert result is None  # Should fail and return None due to exception handling


# --- Test ChannelFrameDataset Exception Edge Cases ---


class TestChannelFrameDatasetExceptionEdgeCases:
    """Tests for exception edge cases in ChannelFrameDataset methods."""

    def test_normalize_with_invalid_kwargs(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Normalize handles invalid kwargs gracefully.

        Invalid kwargs cause an error.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Mock the normalize method to raise an exception using side_effect
        with patch.object(ChannelFrame, "normalize", side_effect=ValueError("Simulated normalization error")):
            normalized_ds = dataset.normalize()
            result = normalized_ds[0]

            assert result is None  # Should fail due to mocked exception and return None

    def test_resample_with_negative_target_sr(self, create_test_files: Path) -> None:
        """Resample with negative target sampling rate.

        A negative sr causes an error.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Resample to invalid negative rate - should fail gracefully
        resampled_ds = dataset.resample(target_sr=-1000)
        result = resampled_ds[0]

        assert result is None  # Should fail and return None due to exception handling

    def test_stft_with_invalid_n_fft(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """STFT handles invalid parameters gracefully.

        STFT errors are caught.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Mock the stft method to raise an exception using side_effect
        with patch.object(ChannelFrame, "stft", side_effect=ValueError("Simulated STFT error")):
            stft_ds = dataset.stft()
            result = stft_ds[0]

            assert result is None  # Should fail due to mocked exception and return None

    def test_stft_exception_in_frame_transform(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """STFT handles exceptions raised by frame.stft() directly.

        Frame.stft() raises an exception during transform.

        We use pytest's side_effect to mock frame.stft() and force an error without
        actually computing STFT with problematic parameters.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a transform that calls frame.stft() - will be mocked to raise exception
        def failing_stft_transform(frame: ChannelFrame) -> SpectrogramFrame | None:
            if frame is not None:
                return frame.stft(n_fft=2048, hop_length=512)  # Normal params, but mocked to fail
            return None

        # Mock stft method to raise an exception with side_effect
        with patch.object(ChannelFrame, "stft", side_effect=ValueError("Simulated STFT computation error")):
            stft_ds = dataset.apply(failing_stft_transform)
            result = stft_ds[0]

            # The transform should fail and return None due to exception handling in apply()
            assert result is None  # Should fail and return None


# --- Test SpectrogramFrameDataset Exception Edge Cases ---


class TestSpectrogramFrameDatasetExceptionEdgeCases:
    """Tests for exception edge cases in SpectrogramFrameDataset methods."""

    def test_plot_with_frame_none(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Plot handles None frame gracefully.

        A None frame is handled.
        Force STFT to fail so the frame at index 0 is deterministically None.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        caplog.set_level(logging.WARNING)
        folder_path = create_test_files
        # Force STFT to fail so that the resulting frame at index 0 is None.
        with patch.object(ChannelFrame, "stft", side_effect=RuntimeError("forced STFT failure")):
            dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
            stft_ds = dataset.stft()

            # Access an index that failed to transform
            result = stft_ds[0]
            assert result is None

            # If frame failed, plot should handle it gracefully and log a warning.
            stft_ds.plot(0)  # Should not raise exception
        assert any("Cannot plot" in record.message and record.levelname == "WARNING" for record in caplog.records)

    def test_plot_with_no_plot_method(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Plot handles frame without plot method gracefully.

        Frames without plot are handled.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        stft_ds = dataset.stft()

        result = stft_ds[0]
        assert result is not None

        with patch.object(type(result), "plot", None):
            with caplog.at_level(logging.WARNING):
                stft_ds.plot(0)  # Should not raise, but log a warning
        assert any(
            "does not have a plot method" in record.message and record.levelname == "WARNING"
            for record in caplog.records
        )


# --- Test LazyFrame Exception Edge Cases ---


class TestLazyFrameExceptionEdgeCases:
    """Tests for exception edge cases in LazyFrame class."""

    def test_ensure_loaded_with_multiple_exceptions(self, tmp_path: Path) -> None:
        """Ensure_loaded handles multiple exception scenarios.

        Exceptions during loading are handled.
        """
        file_path = tmp_path / "test.wav"
        file_path.write_bytes(b"invalid wav data")
        lazy_frame: LazyFrame[ChannelFrame] = LazyFrame(file_path)

        # Try to load invalid file - should fail gracefully
        result = lazy_frame.ensure_loaded(ChannelFrame.from_file)

        assert result is None
        assert lazy_frame.is_loaded is True  # Loading was attempted
        assert lazy_frame.load_attempted is True
        assert lazy_frame.frame is None


# --- Test FrameDataset Constructor Exception Edge Cases ---


class TestFrameDatasetConstructorExceptionEdgeCases:
    """Tests for exception edge cases in FrameDataset constructor."""

    def test_init_with_nonexistent_folder(self, tmp_path: Path) -> None:
        """Initializing with a non-existent folder raises FileNotFoundError.

        Folder existence is checked.
        """
        nonexistent_folder = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Folder does not exist"):
            ChannelFrameDataset(str(nonexistent_folder))

    def test_init_with_empty_folder(self, tmp_path: Path) -> None:
        """Initializing with an empty folder works correctly."""
        dataset = ChannelFrameDataset(str(tmp_path), lazy_loading=True)
        assert len(dataset) == 0


# --- Test apply() Method Exception Edge Cases ---


class TestApplyExceptionEdgeCases:
    """Tests for exception edge cases in the apply() method."""

    def test_apply_with_transform_that_raises(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Apply handles transforms that raise exceptions.

        Transform errors are caught.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        def always_raises(frame: ChannelFrame) -> ChannelFrame | None:
            raise RuntimeError("Always fails")

        new_ds = dataset.apply(always_raises)
        result = new_ds[0]

        assert result is None  # Transform failed, returns None
        assert any(
            "Failed to transform" in record.message and record.levelname == "WARNING" for record in caplog.records
        )

    def test_apply_with_transform_that_returns_none(self, create_test_files: Path) -> None:
        """Apply handles transforms that return None.

        Transform returns None.
        """
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        def always_returns_none(frame: ChannelFrame) -> ChannelFrame | None:
            return None

        new_ds = dataset.apply(always_returns_none)
        result = new_ds[0]

        assert result is None  # Transform returns None


# --- Test SampledFrameDataset Exception Edge Cases ---


class TestSampledFrameDatasetExceptionEdgeCases:
    """Tests for exception edge cases in _SampledFrameDataset class."""

    def test_sample_with_out_of_range_indices(self, create_test_files: Path) -> None:
        """Sampled dataset raises IndexError with out-of-range indices.

        Out-of-range indices are caught.
        """
        folder_path = create_test_files
        original_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a sampled dataset with invalid indices directly via _SampledFrameDataset
        try:
            from wandas.utils.frame_dataset import _SampledFrameDataset

            _SampledFrameDataset(original_ds, [100, 200])  # Out of range
            pytest.fail("Should have raised IndexError")
        except IndexError as e:
            assert "out of range" in str(e).lower() or "Indices are out of range" in str(e)

    def test_sampled_ensure_loaded_with_source_exception(
        self, create_test_files: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Sampled dataset handles exceptions when loading from source.

        Exception handling occurs.
        """
        folder_path = create_test_files
        original_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # Create a sampled dataset - should work normally
        sampled_ds = original_ds.sample(n=1, seed=42)
        result = sampled_ds[0]

        assert result is not None  # Should load successfully


# --- Coverage gap tests for frame_dataset.py ---


class TestFrameDatasetCoverageGaps:
    """Tests targeting uncovered lines in frame_dataset.py."""

    def test_initialize_from_source_with_none_source(self, create_test_files: Path) -> None:
        """_initialize_from_source returns early when source_dataset is None (line 116)."""
        dataset = ChannelFrameDataset(str(create_test_files), lazy_loading=True)
        # Force None and call directly — the guard at line 115 returns immediately
        dataset._source_dataset = None
        original_frames = list(dataset._lazy_frames)
        dataset._initialize_from_source()
        # Nothing should change
        assert dataset._lazy_frames == original_frames

    def test_load_from_source_none_source_frame(self, create_test_files: Path) -> None:
        """_load_from_source returns None when source returns None (line 169)."""
        dataset = ChannelFrameDataset(str(create_test_files), lazy_loading=True)
        derived = dataset.resample(target_sr=8000)

        with patch.object(dataset, "_ensure_loaded", return_value=None):
            result = derived._load_from_source(0)
        assert result is None

    def test_ensure_loaded_exception_path(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """_ensure_loaded catches exceptions, logs an error, and returns None (lines 202-208)."""
        dataset = ChannelFrameDataset(str(create_test_files), lazy_loading=True)

        with patch.object(
            dataset._lazy_frames[0],
            "ensure_loaded",
            side_effect=OSError("corrupted file"),
        ):
            import logging

            with caplog.at_level(logging.ERROR):
                result = dataset._ensure_loaded(0)

        assert result is None
        assert dataset._lazy_frames[0].is_loaded is True
        assert dataset._lazy_frames[0].frame is None

    def test_sample_with_both_n_and_ratio_uses_n(self, create_test_files: Path) -> None:
        """Sample() with n provided alongside ratio uses n (else branch)."""
        dataset = ChannelFrameDataset(str(create_test_files), lazy_loading=True)
        # When n is provided (regardless of ratio), the else branch applies
        sampled = dataset.sample(n=1, ratio=0.99, seed=0)
        assert len(sampled) == 1

    def test_resample_inner_none_frame(self, create_test_files: Path) -> None:
        """Resample inner function returns None when frame is None (line 573)."""
        dataset = ChannelFrameDataset(str(create_test_files), lazy_loading=True)
        resampled = dataset.resample(target_sr=8000)
        assert resampled._transform is not None
        result = resampled._transform(None)
        assert result is None

    def test_trim_inner_none_frame(self, create_test_files: Path) -> None:
        """Trim inner function returns None when frame is None (line 588)."""
        dataset = ChannelFrameDataset(str(create_test_files), lazy_loading=True)
        trimmed = dataset.trim(start=0.0, end=0.5)
        assert trimmed._transform is not None
        result = trimmed._transform(None)
        assert result is None

    def test_trim_inner_exception_returns_none(self, create_test_files: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Trim inner function catches exceptions and returns None (lines 591-593)."""
        import logging

        dataset = ChannelFrameDataset(str(create_test_files), lazy_loading=True)
        trimmed = dataset.trim(start=0.0, end=0.5)
        assert trimmed._transform is not None

        class _RaisingFrame:
            def trim(self, **kwargs):
                raise ValueError("trim failed")

        with caplog.at_level(logging.WARNING):
            result = trimmed._transform(_RaisingFrame())
        assert result is None

    def test_normalize_inner_none_frame(self, create_test_files: Path) -> None:
        """Normalize inner function returns None when frame is None (line 603)."""
        dataset = ChannelFrameDataset(str(create_test_files), lazy_loading=True)
        normalized = dataset.normalize()
        assert normalized._transform is not None
        result = normalized._transform(None)
        assert result is None

    def test_stft_inner_none_frame(self, create_test_files: Path) -> None:
        """Stft inner function returns None when frame is None (line 625)."""
        dataset = ChannelFrameDataset(str(create_test_files), lazy_loading=True)
        stft_ds = dataset.stft()
        assert stft_ds._transform is not None
        result = stft_ds._transform(None)
        assert result is None
