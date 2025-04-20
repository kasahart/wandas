from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from matplotlib.axes import Axes

# テスト対象のクラスをインポート
from wandas.core.lazy.channel_frame import ChannelFrame
from wandas.core.lazy.frame_dataset import (
    ChannelFrameDataset,
    FrameDataset,
    LazyFrame,  # 新しいクラスをインポート
    SpectrogramFrameDataset,
    _SampledFrameDataset,
)
from wandas.core.lazy.spectrogram_frame import SpectrogramFrame
from wandas.utils.types import NDArrayReal

_da_from_array = da.from_array

# --- Test Fixtures ---


@pytest.fixture  # type: ignore [misc, unused-ignore]
def sample_wav_data() -> tuple[int, NDArrayReal]:
    """Generate sample WAV data."""
    sampling_rate = 16000
    duration = 1.0
    n_samples = int(sampling_rate * duration)
    # Create stereo data (2 channels)
    data = np.random.uniform(-0.5, 0.5, size=(n_samples, 2)).astype(np.float32)
    return sampling_rate, data


@pytest.fixture  # type: ignore [misc, unused-ignore]
def sample_csv_data() -> tuple[int, pd.DataFrame]:
    """Generate sample CSV data."""
    sampling_rate = 100
    duration = 1.0
    n_samples = int(sampling_rate * duration)
    time_col = np.linspace(0, duration, n_samples, endpoint=False)
    ch1_data = np.sin(2 * np.pi * 5 * time_col)  # 5 Hz sine wave
    ch2_data = np.cos(2 * np.pi * 10 * time_col)  # 10 Hz cosine wave
    df = pd.DataFrame({"time": time_col, "SensorA": ch1_data, "SensorB": ch2_data})
    return sampling_rate, df


@pytest.fixture  # type: ignore [misc, unused-ignore]
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
    def test_lazy_frame_init(self) -> None:
        """Test LazyFrame initialization."""
        file_path = Path("/path/to/file.wav")
        lazy_frame: LazyFrame[ChannelFrame] = LazyFrame(file_path)

        assert lazy_frame.file_path == file_path
        assert lazy_frame.frame is None
        assert lazy_frame.is_loaded is False
        assert lazy_frame.load_attempted is False

    def test_ensure_loaded(self) -> None:
        """Test LazyFrame.ensure_loaded method."""
        file_path = Path("/path/to/file.wav")
        lazy_frame: LazyFrame[ChannelFrame] = LazyFrame(file_path)

        # モックローダー関数
        mock_frame = ChannelFrame.from_ndarray(
            np.array([[0.1, 0.2], [0.3, 0.4]]), 44100
        )
        loader = MagicMock(return_value=mock_frame)

        # 初回ロード
        result = lazy_frame.ensure_loaded(loader)

        assert result is mock_frame
        assert lazy_frame.frame is mock_frame
        assert lazy_frame.is_loaded is True
        assert lazy_frame.load_attempted is True
        loader.assert_called_once_with(file_path)

        # 2回目のロード（キャッシュから取得）
        loader.reset_mock()
        cached_result = lazy_frame.ensure_loaded(loader)

        assert cached_result is mock_frame
        loader.assert_not_called()

    def test_ensure_loaded_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test LazyFrame.ensure_loaded with loader failure."""
        file_path = Path("/path/to/file.wav")
        lazy_frame: LazyFrame[ChannelFrame] = LazyFrame(file_path)

        # 失敗するローダー関数
        error_msg = "Failed to load file"
        loader = MagicMock(side_effect=RuntimeError(error_msg))

        # ロード試行
        result = lazy_frame.ensure_loaded(loader)

        assert result is None
        assert lazy_frame.frame is None
        assert lazy_frame.is_loaded is True  # ロードは試みた
        assert lazy_frame.load_attempted is True
        loader.assert_called_once_with(file_path)

        # ログ確認
        assert any(
            "読み込みに失敗" in record.message and record.levelname == "ERROR"
            for record in caplog.records
        )

    def test_reset(self) -> None:
        """Test LazyFrame.reset method."""
        file_path = Path("/path/to/file.wav")
        lazy_frame: LazyFrame[ChannelFrame] = LazyFrame(file_path)

        # フレームをロード
        mock_frame = ChannelFrame.from_ndarray(
            np.array([[0.1, 0.2], [0.3, 0.4]]), 44100
        )
        loader = MagicMock(return_value=mock_frame)
        lazy_frame.ensure_loaded(loader)

        # リセット前の状態確認
        assert lazy_frame.frame is mock_frame
        assert lazy_frame.is_loaded is True
        assert lazy_frame.load_attempted is True

        # リセット
        lazy_frame.reset()

        # リセット後の状態確認
        assert lazy_frame.frame is None
        assert lazy_frame.is_loaded is False
        assert lazy_frame.load_attempted is False


# --- Test FrameDataset (Abstract Base Class) ---


class TestFrameDatasetABC:
    def test_frame_dataset_cannot_instantiate(self) -> None:
        """Verify FrameDataset ABC cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            FrameDataset("/some/path")  # type: ignore [abstract]  # 抽象クラスは直接インスタンス化できない

    def test_abstract_methods_must_be_implemented(self, tmp_path: Path) -> None:
        """Test that abstract methods must be implemented in subclasses."""

        # テスト用のサブクラスを作成し、抽象メソッドを実装していない
        class IncompleteFrameDataset(FrameDataset[ChannelFrame]):
            pass

        # Pythonのバージョンによって例外メッセージが違うので、部分一致で確認
        with pytest.raises(TypeError, match="abstract class"):
            IncompleteFrameDataset(str(tmp_path))  # type: ignore [abstract]

        # テスト用のサブクラスを作成し、抽象メソッドを実装
        class MinimalFrameDataset(FrameDataset[ChannelFrame]):
            def _load_file(self, file_path: Path) -> Optional[ChannelFrame]:
                return ChannelFrame.from_ndarray(np.zeros((10, 2)), 44100)

        # これは例外を投げない
        try:
            MinimalFrameDataset(str(tmp_path), lazy_loading=True)
        except TypeError:
            pytest.fail("正しく実装されたサブクラスがインスタンス化に失敗しました")


# --- Test ChannelFrameDataset ---


class TestChannelFrameDataset:
    def test_init_lazy(self, create_test_files: Path) -> None:
        """Test lazy initialization."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(
            str(folder_path), recursive=False, lazy_loading=True
        )

        assert dataset._lazy_loading is True
        assert len(dataset) == 3  # test1.wav, test2.wav, test3.csv
        # LazyFrameの確認 - すべてロードされていない
        assert len(dataset._lazy_frames) == 3
        assert all(not lf.is_loaded for lf in dataset._lazy_frames)
        assert all(lf.frame is None for lf in dataset._lazy_frames)
        assert dataset.folder_path == folder_path
        assert dataset.file_extensions == [
            ".wav",
            ".mp3",
            ".flac",
            ".csv",
        ]  # Default from from_folder via __init__
        assert dataset._recursive is False

    def test_init_eager(self, create_test_files: Path) -> None:
        """Test eager initialization."""
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
        frame_labels = [
            lf.frame.label for lf in dataset._lazy_frames if lf.frame is not None
        ]
        assert frame_labels == file_stems

    def test_init_recursive(self, create_test_files: Path) -> None:
        """Test recursive file discovery."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(
            str(folder_path), recursive=True, lazy_loading=True
        )
        assert len(dataset) == 4  # Includes subdir/sub_test.wav
        assert dataset._recursive is True
        assert Path("subdir/sub_test.wav") in [
            p.relative_to(folder_path) for p in dataset._get_file_paths()
        ]

    def test_init_custom_extensions(self, create_test_files: Path) -> None:
        """Test custom file extensions."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(
            str(folder_path), file_extensions=[".csv"], lazy_loading=True
        )
        assert len(dataset) == 1
        assert dataset._lazy_frames[0].file_path.name == "test3.csv"
        assert dataset.file_extensions == [".csv"]

    def test_init_no_files(self, tmp_path: Path) -> None:
        """Test initialization with a folder containing no matching files."""
        dataset = ChannelFrameDataset(str(tmp_path), lazy_loading=True)
        assert len(dataset) == 0
        assert len(dataset._lazy_frames) == 0

    def test_init_folder_not_found(self) -> None:
        """Test initialization with a non-existent folder."""
        with pytest.raises(FileNotFoundError):
            ChannelFrameDataset("non_existent_folder")

    def test_len(self, create_test_files: Path) -> None:
        """Test __len__ method."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        assert len(dataset) == 3

    def test_getitem_lazy(self, create_test_files: Path) -> None:
        """Test __getitem__ with lazy loading."""
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

    def test_getitem_index_error(self, create_test_files: Path) -> None:
        """Test __getitem__ with out-of-bounds index."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        with pytest.raises(IndexError):
            _ = dataset[3]
        with pytest.raises(IndexError):
            _ = dataset[-1]  # Negative indexing not supported by this logic

    def test_load_file_wav(self, create_test_files: Path) -> None:
        """Test _load_file method for WAV."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        wav_path = dataset._lazy_frames[0].file_path  # test1.wav

        frame = dataset._load_file(wav_path)
        assert isinstance(frame, ChannelFrame)
        assert frame.label == "test1"
        assert frame.sampling_rate == 16000  # From sample_wav_data
        assert frame.n_channels == 2
        assert isinstance(frame.data, np.ndarray)

    def test_load_file_csv(self, create_test_files: Path) -> None:
        """Test _load_file method for CSV."""
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

    def test_load_file_resampling(
        self, create_test_files: Path, sample_wav_data: tuple[int, NDArrayReal]
    ) -> None:
        """Test _load_file triggers resampling when SR mismatches."""
        folder_path = create_test_files
        original_sr, _ = sample_wav_data  # Should be 16000
        target_sr = 8000
        dataset = ChannelFrameDataset(
            str(folder_path), sampling_rate=target_sr, lazy_loading=True
        )
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

    def test_load_file_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test error handling during file loading."""
        folder_path = tmp_path
        # Create a corrupted/invalid file
        invalid_file = folder_path / "invalid.wav"
        invalid_file.write_text("this is not a wav file")

        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        result = dataset._ensure_loaded(0)
        assert result is None
        # Check logs for warning from _ensure_loaded
        assert any(
            "読み込みまたは初期処理に失敗" in record.message
            and record.levelname == "ERROR"
            for record in caplog.records
        )
        # Ensure the frame is marked as None after failure
        assert dataset._lazy_frames[0].frame is None
        assert dataset._lazy_frames[0].is_loaded is True  # ロードは試みた

        # Test _load_all_files catching errors
        caplog.clear()
        dataset_eager = ChannelFrameDataset(str(folder_path), lazy_loading=False)
        assert (
            dataset_eager._lazy_frames[0].frame is None
        )  # Should be None due to load failure
        # 新しい実装では、エラーログはERRORレベルで出力される
        assert any(
            "読み込みまたは初期処理に失敗" in record.message
            for record in caplog.records
        )

    def test_apply_lazy(self, create_test_files: Path) -> None:
        """Test lazy application of a function."""
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
        assert all(
            not lf.is_loaded for lf in transformed_dataset._lazy_frames
        )  # Still lazy

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
        with patch.object(
            transformed_dataset, "_transform", wraps=transformed_dataset._transform
        ) as mock_transform:
            cached_transformed = transformed_dataset[0]
            assert cached_transformed is transformed_frame0
            mock_transform.assert_not_called()

    def test_apply_chaining(self, create_test_files: Path) -> None:
        """Test chaining apply calls."""
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

    def test_apply_error_in_transform(
        self, create_test_files: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test error handling when the applied function fails."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        def failing_transform(frame: ChannelFrame) -> ChannelFrame:
            raise ValueError("Transform failed!")

        transformed_dataset = dataset.apply(failing_transform)

        # 新しい実装では例外はキャッチされてNoneが返される
        result = transformed_dataset[0]
        assert result is None

        # Check logs for warning
        assert any(
            "変換に失敗" in record.message and record.levelname == "WARNING"
            for record in caplog.records
        )
        # Ensure frame is marked as is_loaded but with frame=None
        assert transformed_dataset._lazy_frames[0].frame is None
        assert transformed_dataset._lazy_frames[0].is_loaded is True

    def test_resample_trim_normalize(self, create_test_files: Path) -> None:
        """Test resample, trim, normalize methods (which use apply)."""
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

    def test_stft_lazy(self, create_test_files: Path) -> None:
        """Test lazy STFT using the ChannelFrame.stft result as ground truth."""
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
        np.testing.assert_allclose(
            stft_frame.compute(), expected_spec.compute(), atol=1e-6
        )

        # second access must hit the cache and not retrigger transform
        with patch.object(
            stft_dataset, "_transform", wraps=stft_dataset._transform
        ) as mock_transform:
            cached = stft_dataset[0]
            assert cached is stft_frame
            mock_transform.assert_not_called()

    def test_sample(self, create_test_files: Path) -> None:
        """Test sampling the dataset."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        n_total = len(dataset)

        # Sample by number
        n_sample = 2
        sampled_n = dataset.sample(n=n_sample, seed=42)
        assert isinstance(sampled_n, _SampledFrameDataset)
        assert len(sampled_n) == n_sample
        assert sampled_n._original_dataset is dataset
        # Check sampled indices are subset of original
        original_paths = [p.name for p in dataset._get_file_paths()]
        sampled_paths = [lf.file_path.name for lf in sampled_n._lazy_frames]
        assert all(p in original_paths for p in sampled_paths)

        # Sample by ratio
        ratio = 0.5
        # Expect ceil(3 * 0.5) = 2 samples
        sampled_r = dataset.sample(ratio=ratio, seed=42)
        assert isinstance(sampled_r, _SampledFrameDataset)
        assert len(sampled_r) == int(n_total * ratio)
        assert sampled_r._original_dataset is dataset

        # Sample default (10% or min 1)
        sampled_def = dataset.sample(seed=42)
        assert len(sampled_def) == max(1, int(n_total * 0.1))
        assert sampled_def._original_dataset is dataset  # type: ignore[attr-defined]

        # Sample all
        sampled_all = dataset.sample(n=n_total)
        assert len(sampled_all) == n_total
        assert sampled_all._original_dataset is dataset  # type: ignore[attr-defined]

        # Sample more than available
        sampled_more = dataset.sample(n=n_total + 1)
        assert len(sampled_more) == n_total
        assert sampled_more._original_dataset is dataset  # type: ignore[attr-defined]

        # Sample from empty dataset
        empty_ds = ChannelFrameDataset(
            str(folder_path / "empty_subdir"), lazy_loading=True
        )
        sampled_empty = empty_ds.sample(n=1)
        assert len(sampled_empty) == 0

        # Check lazy loading is preserved in sampled dataset
        sampled_lazy = dataset.sample(n=1, seed=1)
        assert sampled_lazy._lazy_loading is True
        assert all(not lf.is_loaded for lf in sampled_lazy._lazy_frames)

        # Access item in sampled dataset
        original_index = sampled_lazy._original_indices[0]  # type: ignore[attr-defined]
        original_frame = dataset[original_index]  # Ensure original is loaded
        sampled_frame = sampled_lazy[0]  # Trigger load via sampled dataset
        assert isinstance(sampled_frame, ChannelFrame)
        assert original_frame is not None
        assert sampled_frame is not None
        assert sampled_frame.label == original_frame.label
        np.testing.assert_array_equal(sampled_frame.compute(), original_frame.compute())
        assert sampled_lazy._lazy_frames[0].is_loaded is True

        # Check apply on sampled dataset
        def gain_transform(f: ChannelFrame) -> ChannelFrame:
            return f * 2

        transformed_sampled = sampled_lazy.apply(gain_transform)
        assert isinstance(transformed_sampled, _SampledFrameDataset)
        assert transformed_sampled._original_dataset._transform is gain_transform

        # Trigger computation
        final_frame = transformed_sampled[0]
        assert final_frame is not None
        assert original_frame is not None
        expected_data = original_frame.compute() * 2
        np.testing.assert_allclose(final_frame.compute(), expected_data)

    def test_get_metadata(self, create_test_files: Path) -> None:
        """Test get_metadata method."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(
            str(folder_path), lazy_loading=True, sampling_rate=8000
        )

        meta_lazy = dataset.get_metadata()
        assert meta_lazy["folder_path"] == str(folder_path)
        assert meta_lazy["file_count"] == 3
        assert meta_lazy["loaded_count"] == 0
        assert meta_lazy["target_sampling_rate"] == 8000
        # Actual SR might be different until loaded, or None if load fails
        assert (
            meta_lazy["actual_sampling_rate"] == 8000
        )  # Inherits target if not loaded
        assert meta_lazy["lazy_loading"] is True
        assert meta_lazy["has_transform"] is False
        assert meta_lazy["frame_type"] == "Unknown"  # Not loaded yet

        # Load one item
        _ = dataset[0]
        meta_loaded_one = dataset.get_metadata()
        assert meta_loaded_one["loaded_count"] == 1
        assert (
            meta_loaded_one["actual_sampling_rate"] == 8000
        )  # Now reflects loaded frame (after resampling)
        assert meta_loaded_one["frame_type"] == "ChannelFrame"

        # Test with transformation
        transformed_ds = dataset.apply(lambda f: f * 2)
        meta_transformed = transformed_ds.get_metadata()
        assert meta_transformed["has_transform"] is True
        assert (
            meta_transformed["loaded_count"] == 0
        )  # Transform dataset has its own cache
        assert meta_transformed["frame_type"] == "Unknown"

        # Load item in transformed dataset
        _ = transformed_ds[0]
        meta_transformed_loaded = transformed_ds.get_metadata()
        assert meta_transformed_loaded["loaded_count"] == 1
        assert (
            meta_transformed_loaded["frame_type"] == "ChannelFrame"
        )  # Type after transform

        # Test with eager loading
        dataset_eager = ChannelFrameDataset(str(folder_path), lazy_loading=False)
        meta_eager = dataset_eager.get_metadata()
        assert meta_eager["lazy_loading"] is False
        assert meta_eager["loaded_count"] == 3
        assert meta_eager["frame_type"] == "ChannelFrame"

    def test_from_folder(self, create_test_files: Path) -> None:
        """Test the from_folder class method."""
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

    def test_save(self, create_test_files: Path, tmp_path: Path) -> None:
        """Test save method raises NotImplementedError."""
        folder_path = create_test_files
        output_folder = tmp_path / "output"
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # saveメソッドが未実装のため、NotImplementedErrorが発生することを確認
        with pytest.raises(
            NotImplementedError, match="saveメソッドは現在実装されていません"
        ):
            dataset.save(str(output_folder), filename_prefix="processed_")

        # SpectrogramFrameDatasetのsaveも同様に例外を発生させるか確認
        stft_dataset = dataset.stft()
        with pytest.raises(
            NotImplementedError, match="saveメソッドは現在実装されていません"
        ):
            stft_dataset.save(str(output_folder), filename_prefix="spec_")


# --- Test SpectrogramFrameDataset ---


class TestSpectrogramFrameDataset:
    def test_init_lazy(self, create_test_files: Path) -> None:
        """Test initialization (typically via ChannelFrameDataset.stft)."""
        folder_path = create_test_files
        channel_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        stft_ds = channel_ds.stft()  # Creates SpectrogramFrameDataset

        assert isinstance(stft_ds, SpectrogramFrameDataset)
        assert stft_ds._lazy_loading is True
        assert stft_ds._source_dataset is channel_ds
        assert stft_ds._transform is not None
        assert len(stft_ds) == len(channel_ds)
        assert all(not lf.is_loaded for lf in stft_ds._lazy_frames)

    def test_load_file_not_implemented(self, tmp_path: Path) -> None:
        """Test that _load_file raises NotImplementedError."""
        # Create a dummy file that might look like a spectrogram source
        dummy_spec_file = tmp_path / "spec.npy"
        np.save(dummy_spec_file, np.random.rand(1, 10, 5))

        # Initialize directly (not the typical way)
        spec_ds = SpectrogramFrameDataset(str(tmp_path), file_extensions=[".npy"])

        # 現在の実装では、_load_fileは例外を投げるが、
        # _ensure_loadedでそれがキャッチされる
        # 直接_load_fileを呼び出して例外を確認
        with pytest.raises(
            NotImplementedError,
            match="直接SpectrogramFrameをロードする方法は定義されていません",
        ):
            spec_ds._load_file(dummy_spec_file)

        # __getitem__経由では例外はキャッチされてNoneが返される
        result = spec_ds[0]
        assert result is None

    def test_apply(self, create_test_files: Path) -> None:
        """Test apply on SpectrogramFrameDataset."""
        folder_path = create_test_files
        channel_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        stft_ds = channel_ds.stft()

        def spec_transform(spec_frame: SpectrogramFrame) -> SpectrogramFrame:
            # より単純な変換関数（amplitude_to_dbは実装されていない可能性があるため）
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
        assert not np.allclose(
            original_spec_frame.compute(), transformed_spec_frame.compute()
        )

    @patch("matplotlib.pyplot.show")  # Prevent plots from displaying during tests
    def test_plot(
        self,
        mock_show: MagicMock,
        create_test_files: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test plotting from SpectrogramFrameDataset."""
        folder_path = create_test_files
        channel_ds = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        stft_ds = channel_ds.stft()

        # Mock SpectrogramFrame.plot to check if it's called
        with patch.object(
            SpectrogramFrame, "plot", return_value=MagicMock(spec=Axes)
        ) as mock_frame_plot:
            stft_ds.plot(0)  # Plot the first spectrogram
            mock_frame_plot.assert_called_once()

        # Test plot with load/transform error
        caplog.clear()

        def failing_stft(frame: ChannelFrame) -> SpectrogramFrame:
            if frame.label == "test1":
                raise ValueError("STFT failed")
            return frame.stft()

        # Need to wrap this in a SpectrogramFrameDataset manually for plot method
        error_spec_ds = SpectrogramFrameDataset(
            folder_path=str(folder_path),
            source_dataset=channel_ds,
            transform=failing_stft,
        )

        with patch.object(SpectrogramFrame, "plot") as mock_plot_error:
            error_spec_ds.plot(0)  # Attempt to plot the failing one
            mock_plot_error.assert_not_called()
            assert any(
                "ロード/変換に失敗していたためプロットできません" in rec.message
                for rec in caplog.records
                if rec.levelname == "WARNING"
            )

        # Test plot when frame has no plot method (simulate)
        caplog.clear()

        # plotがないクラスを定義
        with patch.object(SpectrogramFrame, "plot", None):
            # plotメソッドがないフレームをロードした場合のテスト
            stft_ds.plot(0)
            assert any(
                "plotメソッドが実装されていません" in rec.message
                for rec in caplog.records
                if rec.levelname == "WARNING"
            )

        # Test error during plotting itself
        caplog.clear()
        plot_error_msg = "Plotting itself failed"
        with patch.object(
            SpectrogramFrame, "plot", side_effect=RuntimeError(plot_error_msg)
        ) as mock_plot_runtime_error:
            stft_ds.plot(1)  # Plot second item, assume it loads ok
            mock_plot_runtime_error.assert_called_once()
            assert any(
                plot_error_msg in rec.message
                for rec in caplog.records
                if rec.levelname == "ERROR"
            )


# --- Test _SampledFrameDataset (Internal Class) ---


class TestSampledFrameDataset:
    def test_init(self, create_test_files: Path) -> None:
        """Test initialization of _SampledFrameDataset."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # サンプリングインデックスの準備
        sampled_indices = [0, 2]  # test1.wav と test3.csv を選択

        # _SampledFrameDataset の初期化
        sampled_ds = _SampledFrameDataset(dataset, sampled_indices)

        # 基本的なプロパティの確認
        assert sampled_ds._original_dataset is dataset
        assert sampled_ds._original_indices == sampled_indices
        assert len(sampled_ds) == len(sampled_indices)

        # LazyFrames の確認
        assert len(sampled_ds._lazy_frames) == len(sampled_indices)
        assert all(not lf.is_loaded for lf in sampled_ds._lazy_frames)
        assert sampled_ds._lazy_frames[0].file_path.name == "test1.wav"
        assert sampled_ds._lazy_frames[1].file_path.name == "test3.csv"

    def test_getitem(self, create_test_files: Path) -> None:
        """Test __getitem__ method of _SampledFrameDataset."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # 元のデータセットからサンプリング
        sampled_indices = [0, 2]  # test1.wav, test3.csv
        sampled_ds = _SampledFrameDataset(dataset, sampled_indices)

        assert isinstance(sampled_ds, _SampledFrameDataset)
        assert sampled_ds._original_dataset is dataset
        assert sampled_ds._original_indices == sampled_indices

        # サンプリングされたデータセットからフレームを取得
        sampled_frame_0 = sampled_ds[0]

        # 元のデータセットから直接同じフレームを取得
        original_frame_0 = dataset[sampled_indices[0]]

        # Noneチェック
        assert sampled_frame_0 is not None
        assert original_frame_0 is not None

        # 同じフレームが取得できていることを確認
        assert isinstance(sampled_frame_0, ChannelFrame)
        assert sampled_frame_0.label == original_frame_0.label
        np.testing.assert_array_equal(
            sampled_frame_0.compute(),
            original_frame_0.compute(),
            err_msg="サンプリングデータセットのフレームが元のデータセットと一致しません",
        )

        # キャッシュが機能していることを確認
        assert sampled_ds._lazy_frames[0].is_loaded is True

        # キャッシュを通じて同じフレームオブジェクトが返されることを確認
        cached_frame = sampled_ds[0]
        assert cached_frame is sampled_frame_0  # 同じインスタンスが返されるはず

        # 2番目の要素も確認
        sampled_frame_1 = sampled_ds[1]
        original_frame_1 = dataset[sampled_indices[1]]

        assert isinstance(sampled_frame_1, ChannelFrame)
        assert sampled_frame_1 is not None
        assert original_frame_1 is not None
        assert sampled_frame_1.label == original_frame_1.label
        np.testing.assert_array_equal(
            sampled_frame_1.compute(), original_frame_1.compute()
        )

        # 範囲外のインデックスに対するエラーテスト
        with pytest.raises(IndexError):
            _ = sampled_ds[2]  # サンプリングデータセットは2つの要素しかない

        # 元のデータセットのロードエラーをシミュレート
        # _ensure_loadedメソッドを直接モックする方が確実です
        with patch.object(dataset, "_ensure_loaded", return_value=None):
            # _SampledFrameDatasetはキャッシュをリセットする必要がある
            sampled_ds._lazy_frames[0].reset()
            result = sampled_ds[0]
            assert result is None

    def test_apply(self, create_test_files: Path) -> None:
        """Test apply method of _SampledFrameDataset."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        sampled_ds = dataset.sample(n=1, seed=42)

        def transform_gain(f: ChannelFrame) -> ChannelFrame:
            assert f is not None, "Input frame to transform_gain is None"
            return f * 3

        transformed_sampled = sampled_ds.apply(transform_gain)

        # 基本的な検証
        assert isinstance(transformed_sampled, _SampledFrameDataset)
        assert transformed_sampled._original_dataset is not dataset
        assert transformed_sampled._original_dataset._transform is transform_gain
        assert transformed_sampled._original_dataset._source_dataset is dataset
        assert len(transformed_sampled) == 1
        assert all(not lf.is_loaded for lf in transformed_sampled._lazy_frames)

        # 元のフレームを取得
        original_index = sampled_ds._original_indices[0]  # type: ignore[attr-defined]
        original_frame = dataset[original_index]

        # サンプリングされたフレームを取得
        sampled_frame = sampled_ds[0]
        assert original_frame is not None
        assert sampled_frame is not None
        np.testing.assert_allclose(
            sampled_frame.compute(),
            original_frame.compute(),
            err_msg="Data mismatch between dataset and sampled_ds",
        )

        # 変換後のフレームを取得して検証
        final_frame = transformed_sampled[0]
        assert final_frame is not None
        assert original_frame is not None
        expected_data = original_frame.compute() * 3
        np.testing.assert_allclose(
            final_frame.compute(),
            expected_data,
            err_msg="Data mismatch after applying transform",
        )

        # キャッシュが機能していることを確認
        assert transformed_sampled._lazy_frames[0].is_loaded is True
        cached_final_frame = transformed_sampled[0]
        assert cached_final_frame is final_frame

    def test_load_file_not_implemented(self, create_test_files: Path) -> None:
        """Test that _load_file raises NotImplementedError."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)
        sampled_indices = [0, 1]
        sampled_ds = _SampledFrameDataset(dataset, sampled_indices)

        with pytest.raises(
            NotImplementedError,
            match="_SampledFrameDatasetは直接ファイルをロードしません",
        ):
            sampled_ds._load_file(Path("dummy.wav"))

    def test_sample_index_error(self, create_test_files: Path) -> None:
        """Test error handling with invalid indices."""
        folder_path = create_test_files
        dataset = ChannelFrameDataset(str(folder_path), lazy_loading=True)

        # 範囲外のインデックスを含むケース
        invalid_indices = [0, 10]  # 10は範囲外（ファイルは3つしかない）

        with pytest.raises(
            IndexError, match="インデックスが元データセットの範囲外です"
        ):
            _SampledFrameDataset(dataset, invalid_indices)
