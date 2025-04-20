import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from wandas.core.lazy.file_readers import (
    CSVFileReader,
    FileReader,
    SoundFileReader,
    _file_readers,
    get_file_reader,
    register_file_reader,
)
from wandas.utils.types import NDArrayReal


class TestSoundFileReader:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.reader: SoundFileReader = SoundFileReader()

        # Create a temporary wav file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file: Path = Path(self.temp_dir.name) / "test_audio.wav"

        # Create sample audio data: 2 channels, 0.5 seconds at 16kHz
        sample_rate: int = 16000
        duration: float = 0.5
        samples: int = int(sample_rate * duration)
        test_data: NDArrayReal = np.random.random((samples, 2)).astype(np.float32)
        sf.write(self.test_file, test_data, sample_rate)

        re_test_data, _ = sf.read(self.test_file)
        # Store expected data for tests
        self.expected_data: NDArrayReal = (
            re_test_data.T
        )  # Transpose to get (channels, samples)
        self.sample_rate: int = sample_rate
        self.n_samples: int = samples
        self.n_channels: int = 2

    def teardown_method(self) -> None:
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_get_data_full_file(self) -> None:
        """Test reading the entire audio file."""
        data = self.reader.get_data(
            self.test_file, channels=[0, 1], start_idx=0, frames=self.n_samples
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (self.n_channels, self.n_samples)
        np.testing.assert_allclose(data, self.expected_data)

    def test_get_data_single_channel(self) -> None:
        """Test reading a single channel."""
        data = self.reader.get_data(
            self.test_file, channels=[0], start_idx=0, frames=self.n_samples
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (1, self.n_samples)
        np.testing.assert_allclose(data, self.expected_data[0:1])

    def test_get_data_with_offset(self) -> None:
        """Test reading with a start offset."""
        offset: int = 1000
        data = self.reader.get_data(
            self.test_file,
            channels=[0, 1],
            start_idx=offset,
            frames=self.n_samples - offset,
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (self.n_channels, self.n_samples - offset)
        np.testing.assert_allclose(data, self.expected_data[:, offset:])

    def test_get_data_frame_limit(self) -> None:
        """Test reading with a specified number of frames."""
        frames: int = 2000
        data = self.reader.get_data(
            self.test_file, channels=[0, 1], start_idx=0, frames=frames
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (self.n_channels, frames)
        np.testing.assert_allclose(data, self.expected_data[:, :frames])

    def test_get_data_file_not_found(self) -> None:
        """Test error handling when file doesn't exist."""
        with pytest.raises(RuntimeError):
            self.reader.get_data(
                "nonexistent_file.wav", channels=[0], start_idx=0, frames=1000
            )


class TestCSVFileReader:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.reader: CSVFileReader = CSVFileReader()

        # Create a temporary CSV file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file: Path = Path(self.temp_dir.name) / "test_data.csv"

        # Create sample data: time column + 3 data columns, 1000 rows
        n_rows: int = 1000
        sample_rate: int = 1000
        time_values: NDArrayReal = np.arange(n_rows) / sample_rate  # 1kHz sample rate
        data_values: NDArrayReal = np.random.random((n_rows, 3))

        # Create DataFrame and save to CSV
        df: pd.DataFrame = pd.DataFrame(
            np.column_stack([time_values, data_values]),
            columns=["time", "ch1", "ch2", "ch3"],
        )
        df.to_csv(self.test_file, index=False)

        # Store expected data for tests
        self.expected_data: NDArrayReal = (
            data_values.T
        )  # Transpose to get (channels, samples)
        self.n_samples: int = n_rows
        self.n_channels: int = 3

    def teardown_method(self) -> None:
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_get_file_info_basic(self) -> None:
        """Test basic functionality of get_file_info method."""
        info = self.reader.get_file_info(self.test_file)

        # Check if all expected keys are present
        expected_keys = ["samplerate", "channels", "frames", "format", "duration"]
        for key in expected_keys:
            assert key in info

        # Check specific values
        assert info["format"] == "CSV"
        assert info["channels"] == self.n_channels
        assert info["frames"] == 1000
        assert info["duration"] == 1

    def test_get_file_info_samplerate_calculation(self) -> None:
        """Test samplerate estimation from evenly spaced time values."""
        # Create a CSV with consistent time intervals for precise samplerate calculation
        temp_file = Path(self.temp_dir.name) / "even_intervals.csv"

        # Create data with exact time intervals (0.01 seconds = 100Hz sample rate)
        n_rows = 100
        time_values = np.arange(0, n_rows * 0.01, 0.01)  # 100Hz sampling
        data_values = np.random.random((n_rows, 2))

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

    def test_get_file_info_single_row(self) -> None:
        """
        Test behavior with a CSV file containing only one row.
        (can't calculate samplerate)
        """
        # Create a CSV with only one row
        temp_file = Path(self.temp_dir.name) / "single_row.csv"

        df = pd.DataFrame(
            [[0.0, 0.5, 0.3], [1, 0.5, 0.3]],
            columns=["time", "ch1", "ch2"],
        )
        df.to_csv(temp_file, index=False)

        # Get file info
        info = self.reader.get_file_info(temp_file)

        # Check that samplerate is 0 (can't calculate from single row)
        assert info["samplerate"] == 1
        assert info["channels"] == 2
        assert info["format"] == "CSV"

    def test_get_file_info_no_time_column(self) -> None:
        """Test behavior with a CSV file that has non-numeric first column."""
        # Create a CSV with string first column
        temp_file = Path(self.temp_dir.name) / "no_time_column.csv"

        n_rows = 50
        data_values = np.random.random((n_rows, 2))
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
        data = self.reader.get_data(
            self.test_file, channels=[], start_idx=0, frames=self.n_samples
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (self.n_channels, self.n_samples)
        np.testing.assert_allclose(data, self.expected_data)

    def test_get_data_subset_channels(self) -> None:
        """Test reading a subset of channels."""
        channels: list[int] = [
            0,
            2,
        ]  # First and third data channels (after time column removed)
        data = self.reader.get_data(
            self.test_file, channels=channels, start_idx=0, frames=self.n_samples
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (len(channels), self.n_samples)
        np.testing.assert_allclose(data, self.expected_data[channels])

    def test_get_data_with_offset(self) -> None:
        """Test reading with a start offset."""
        offset: int = 200
        data = self.reader.get_data(
            self.test_file, channels=[], start_idx=offset, frames=self.n_samples
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (self.n_channels, self.n_samples - offset)
        np.testing.assert_allclose(data, self.expected_data[:, offset:])

    def test_get_data_frame_limit(self) -> None:
        """Test reading with a specified number of frames."""
        frames: int = 500
        data = self.reader.get_data(
            self.test_file, channels=[], start_idx=0, frames=frames
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (self.n_channels, frames)
        np.testing.assert_allclose(data, self.expected_data[:, :frames])

    def test_get_data_invalid_channels(self) -> None:
        """Test error handling with invalid channel indices."""
        with pytest.raises(ValueError, match="Requested channels.*out of range"):
            self.reader.get_data(
                self.test_file, channels=[10], start_idx=0, frames=self.n_samples
            )

    def test_get_data_file_not_found(self) -> None:
        """Test error handling when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.reader.get_data(
                "nonexistent_file.csv", channels=[], start_idx=0, frames=1000
            )


class TestGetFileReader:
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.wav_path: str = "test.wav"
        self.csv_path: str = "test.csv"
        self.unsupported_path: str = "test.xyz"

    def test_get_file_reader_wav(self) -> None:
        """Test getting reader for WAV file."""

        with mock.patch(
            "pathlib.Path.suffix", new_callable=mock.PropertyMock
        ) as mock_suffix:
            mock_suffix.return_value = ".wav"
            reader: FileReader = get_file_reader(self.wav_path)
            assert isinstance(reader, SoundFileReader)

    def test_get_file_reader_csv(self) -> None:
        """Test getting reader for CSV file."""

        with mock.patch(
            "pathlib.Path.suffix", new_callable=mock.PropertyMock
        ) as mock_suffix:
            mock_suffix.return_value = ".csv"
            reader: FileReader = get_file_reader(self.csv_path)
            assert isinstance(reader, CSVFileReader)

    def test_get_file_reader_unsupported(self) -> None:
        """Test error when no suitable reader found."""

        with mock.patch(
            "pathlib.Path.suffix", new_callable=mock.PropertyMock
        ) as mock_suffix:
            mock_suffix.return_value = ".xyz"
            with pytest.raises(ValueError, match="No suitable file reader found"):
                get_file_reader(self.unsupported_path)


class TestRegisterFileReader:
    def test_register_file_reader(self) -> None:
        """Test registering a new file reader."""

        # Create a custom reader class
        class CustomFileReader(SoundFileReader):
            supported_extensions: list[str] = [".custom"]

        # Get the original count of readers
        original_count: int = len(_file_readers)

        # Register the new reader
        register_file_reader(CustomFileReader)

        # Verify reader was added
        assert len(_file_readers) == original_count + 1

        # Verify the new reader can be retrieved
        with mock.patch(
            "pathlib.Path.suffix", new_callable=mock.PropertyMock
        ) as mock_suffix:
            mock_suffix.return_value = ".custom"
            reader = get_file_reader("test.custom")
            assert isinstance(reader, CustomFileReader)
