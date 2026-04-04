import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import wandas.io.readers as readers
from wandas.io.readers import (
    CSVFileReader,
    SoundFileReader,
    _file_readers,
    _normalize_extension,
    _prepare_file_source,
    get_file_reader,
    register_file_reader,
)
from wandas.utils.types import NDArrayReal


class TestSoundFileReader:
    # Constants for the standard test audio file
    SAMPLE_RATE: int = 16000
    DURATION: float = 0.5
    N_SAMPLES: int = int(SAMPLE_RATE * DURATION)
    N_CHANNELS: int = 2

    @pytest.fixture(autouse=True)
    def _setup_wav(self, tmp_path: Path) -> None:
        """Create a standard stereo WAV file using seeded RNG (Grand Policy)."""
        self.reader = SoundFileReader()
        self.test_file = tmp_path / "test_audio.wav"

        # Seeded RNG for reproducibility (Grand Policy: no unseeded random data)
        rng = np.random.default_rng(42)
        test_data: NDArrayReal = rng.random((self.N_SAMPLES, self.N_CHANNELS)).astype(np.float32)
        sf.write(self.test_file, test_data, self.SAMPLE_RATE)

        re_test_data, _ = sf.read(self.test_file)
        self.expected_data: NDArrayReal = re_test_data.T  # (channels, samples)

    def test_get_data_full_file(self) -> None:
        """Test reading the entire audio file."""
        data = self.reader.get_data(self.test_file, channels=[0, 1], start_idx=0, frames=self.N_SAMPLES, normalize=True)

        assert isinstance(data, np.ndarray)
        assert data.shape == (self.N_CHANNELS, self.N_SAMPLES)
        np.testing.assert_allclose(np.asarray(data), np.asarray(self.expected_data))

    def test_get_data_single_channel(self) -> None:
        """Test reading a single channel."""
        data = self.reader.get_data(self.test_file, channels=[0], start_idx=0, frames=self.N_SAMPLES, normalize=True)

        assert isinstance(data, np.ndarray)
        assert data.shape == (1, self.N_SAMPLES)
        np.testing.assert_allclose(np.asarray(data), np.asarray(self.expected_data[0:1]))

    @pytest.mark.parametrize("offset", [200, 1000], ids=["offset_200", "offset_1000"])
    def test_get_data_with_offset(self, offset: int) -> None:
        """Test reading with various start offsets preserves data slice."""
        data = self.reader.get_data(
            self.test_file,
            channels=[0, 1],
            start_idx=offset,
            frames=self.N_SAMPLES - offset,
            normalize=True,
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (self.N_CHANNELS, self.N_SAMPLES - offset)
        # Exact match: same reader, same file, data slice comparison
        np.testing.assert_allclose(np.asarray(data), np.asarray(self.expected_data[:, offset:]))

    @pytest.mark.parametrize("frames", [500, 2000], ids=["frames_500", "frames_2000"])
    def test_get_data_frame_limit(self, frames: int) -> None:
        """Test reading with various frame limits returns correct shape and data."""
        data = self.reader.get_data(self.test_file, channels=[0, 1], start_idx=0, frames=frames, normalize=True)

        assert isinstance(data, np.ndarray)
        assert data.shape == (self.N_CHANNELS, frames)
        # Exact match: same reader, same file, frame-limited comparison
        np.testing.assert_allclose(np.asarray(data), np.asarray(self.expected_data[:, :frames]))

    def test_get_data_file_not_found(self) -> None:
        """Test error handling when file doesn't exist."""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            self.reader.get_data("nonexistent_file.wav", channels=[0], start_idx=0, frames=1000)

    def test_get_data_unexpected_array_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test error when unexpected data type is returned."""
        monkeypatch.setattr(readers.np, "ndarray", tuple)

        with pytest.raises(ValueError, match="Unexpected data type after reading file"):
            self.reader.get_data(self.test_file, channels=[0, 1], start_idx=0, frames=self.N_SAMPLES)

    def test_get_data_invalid_channels(self) -> None:
        """Test error handling with invalid channel indices."""
        with pytest.raises(IndexError):
            self.reader.get_data(self.test_file, channels=[10], start_idx=0, frames=self.N_SAMPLES)


class TestCSVFileReader:
    # Constants for the standard CSV test file
    N_ROWS: int = 1000
    N_CHANNELS: int = 3

    @pytest.fixture(autouse=True)
    def _setup_csv(self, tmp_path: Path) -> None:
        """Create a standard CSV file using seeded RNG (Grand Policy)."""
        self.reader = CSVFileReader()
        self.tmp_path = tmp_path
        self.test_file = tmp_path / "test_data.csv"

        # Seeded RNG for reproducibility (Grand Policy: no unseeded random data)
        rng = np.random.default_rng(99)
        sample_rate: int = 1000
        time_values: NDArrayReal = np.arange(self.N_ROWS) / sample_rate  # 1kHz
        data_values: NDArrayReal = rng.random((self.N_ROWS, self.N_CHANNELS))

        df: pd.DataFrame = pd.DataFrame(
            np.column_stack([time_values, data_values]),
            columns=["time", "ch1", "ch2", "ch3"],
        )
        df.to_csv(self.test_file, index=False)

        self.expected_data: NDArrayReal = data_values.T  # (channels, samples)

    def test_get_file_info_basic(self) -> None:
        """Test basic functionality of get_file_info method."""
        info = self.reader.get_file_info(self.test_file)

        # Verify all expected keys and values explicitly (no magic loops)
        assert "samplerate" in info, "Missing 'samplerate' key"
        assert "channels" in info, "Missing 'channels' key"
        assert "frames" in info, "Missing 'frames' key"
        assert "format" in info, "Missing 'format' key"
        assert "duration" in info, "Missing 'duration' key"

        assert info["format"] == "CSV"
        assert info["channels"] == self.N_CHANNELS
        assert info["frames"] == self.N_ROWS
        assert info["duration"] == 1  # 1000 rows / 1000 Hz = 1 second

    def test_get_file_info_samplerate_calculation(self) -> None:
        """Test samplerate estimation from evenly spaced time values."""
        # Create a CSV with consistent time intervals for precise samplerate calculation
        temp_file = self.tmp_path / "even_intervals.csv"

        # Create data with exact time intervals (0.01 seconds = 100Hz sample rate)
        rng = np.random.default_rng(10)
        n_rows = 100
        time_values = np.arange(0, n_rows * 0.01, 0.01)  # 100Hz sampling
        data_values = rng.random((n_rows, 2))

        df = pd.DataFrame(
            np.column_stack([time_values, data_values]),
            columns=["time", "ch1", "ch2"],
        )
        df.to_csv(temp_file, index=False)

        # Get file info
        info = self.reader.get_file_info(temp_file)

        # Check the estimated sample rate (should be close to 100Hz)
        assert info["samplerate"] == 100
        assert info["channels"] == 2

    def test_get_file_info_time_column_string(self) -> None:
        """Test samplerate estimation using a named time column."""
        temp_file = self.tmp_path / "time_column_name.csv"

        rng = np.random.default_rng(11)
        n_rows = 50
        time_values = np.arange(0, n_rows * 0.01, 0.01)
        data_values = rng.random((n_rows, 2))

        df = pd.DataFrame(
            np.column_stack([time_values, data_values]),
            columns=["time", "ch1", "ch2"],
        )
        df.to_csv(temp_file, index=False)

        info = self.reader.get_file_info(temp_file, time_column="time")

        assert info["samplerate"] == 100
        assert info["channels"] == 2

    def test_get_file_info_single_row(self) -> None:
        """
        Test behavior with a CSV file containing only one row.
        (can't calculate samplerate)
        """
        # Create a CSV with only one row
        temp_file = self.tmp_path / "single_row.csv"

        df = pd.DataFrame([[0.0, 0.5, 0.3]], columns=["time", "ch1", "ch2"])
        df.to_csv(temp_file, index=False)

        # Get file info
        info = self.reader.get_file_info(temp_file)

        # Check that samplerate is 0 (can't calculate from single row)
        assert info["samplerate"] == 0
        assert info["channels"] == 2
        assert info["format"] == "CSV"
        assert info["duration"] is None

    def test_get_file_info_no_time_column(self) -> None:
        """Test behavior with a CSV file that has non-numeric first column."""
        # Create a CSV with string first column
        temp_file = self.tmp_path / "no_time_column.csv"

        rng = np.random.default_rng(12)
        n_rows = 50
        data_values = rng.random((n_rows, 2))
        labels = [f"label_{i}" for i in range(n_rows)]

        df = pd.DataFrame(
            np.column_stack([np.array(labels).reshape(-1, 1), data_values]),
            columns=["label", "ch1", "ch2"],
        )
        df.to_csv(temp_file, index=False)

        # Get file info - should handle the exception when
        # trying to calculate samplerate
        info = self.reader.get_file_info(temp_file)

        # Check that samplerate is 0 (can't calculate from non-numeric column)
        assert info["samplerate"] == 0
        assert info["channels"] == 2
        assert info["format"] == "CSV"

    def test_get_data_full_file(self) -> None:
        """Test reading the entire CSV file."""
        data = self.reader.get_data(self.test_file, channels=[], start_idx=0, frames=self.N_ROWS)

        assert isinstance(data, np.ndarray)
        assert data.shape == (self.N_CHANNELS, self.N_ROWS)
        np.testing.assert_allclose(np.asarray(data), np.asarray(self.expected_data))

    def test_csv_channel_count_equals_data_columns_minus_time(self) -> None:
        """CSV channel count = number of data columns excluding time column (I/O Policy).

        A CSV with columns [time, ch1, ch2, ch3] must report 3 channels.
        """
        info = self.reader.get_file_info(self.test_file)
        # 4 total columns - 1 time column = 3 data channels
        assert info["channels"] == 3, "Channel count must equal data columns minus time column"

    def test_get_data_subset_channels(self) -> None:
        """Test reading a subset of channels."""
        channels: list[int] = [
            0,
            2,
        ]  # First and third data channels (after time column removed)
        data = self.reader.get_data(self.test_file, channels=channels, start_idx=0, frames=self.N_ROWS)

        assert isinstance(data, np.ndarray)
        assert data.shape == (len(channels), self.N_ROWS)
        np.testing.assert_allclose(np.asarray(data), np.asarray(self.expected_data[channels]))

    def test_get_data_channel_out_of_range(self) -> None:
        """Test error when requesting out-of-range channels."""
        with pytest.raises(ValueError, match="Requested channels"):
            self.reader.get_data(self.test_file, channels=[99], start_idx=0, frames=self.N_ROWS)

    def test_get_data_unexpected_array_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test error when unexpected data type is returned."""

        class FakeValues:
            T = [[1.0, 2.0], [3.0, 4.0]]

        monkeypatch.setattr(pd.DataFrame, "values", property(lambda _self: FakeValues()))

        with pytest.raises(ValueError, match="Unexpected data type after reading file"):
            self.reader.get_data(self.test_file, channels=[0, 1], start_idx=0, frames=self.N_ROWS)


class TestFileReaderHelpers:
    def test_normalize_extension(self) -> None:
        assert _normalize_extension(None) is None
        assert _normalize_extension("wav") == ".wav"
        assert _normalize_extension(".csv") == ".csv"

    def test_prepare_file_source_nonseekable(self) -> None:
        class NonSeekableIO(io.BytesIO):
            def seek(self, *args, **kwargs):
                raise OSError("seek not supported")

        stream = NonSeekableIO(b"dummy")
        prepared = _prepare_file_source(stream)

        assert prepared is stream


class TestGetFileReader:
    def test_get_file_reader_wav(self) -> None:
        """WAV extension dispatches to SoundFileReader."""
        reader = get_file_reader("test.wav")
        assert isinstance(reader, SoundFileReader)

    def test_get_file_reader_csv(self) -> None:
        """CSV extension dispatches to CSVFileReader."""
        reader = get_file_reader("test.csv")
        assert isinstance(reader, CSVFileReader)

    def test_get_file_reader_unsupported_raises_error(self) -> None:
        """Unsupported extension raises ValueError."""
        with pytest.raises(ValueError, match="No suitable file reader found"):
            get_file_reader("test.xyz")

    def test_get_file_reader_file_type_normalization(self) -> None:
        reader = get_file_reader("ignored", file_type="wav")
        assert isinstance(reader, SoundFileReader)

    def test_get_file_reader_requires_file_type_for_in_memory(self) -> None:
        with pytest.raises(ValueError, match="File type is required when the extension is missing"):
            get_file_reader(b"in-memory")


class TestRegisterFileReader:
    def test_register_custom_reader_adds_to_registry(self) -> None:
        """Registering a custom reader makes it retrievable by extension."""

        class CustomFileReader(SoundFileReader):
            supported_extensions: list[str] = [".custom"]

        original_count: int = len(_file_readers)
        register_file_reader(CustomFileReader)

        assert len(_file_readers) == original_count + 1
        reader = get_file_reader("test.custom")
        assert isinstance(reader, CustomFileReader)
