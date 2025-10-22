"""Tests for WAV I/O error messages."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from wandas.io.wav_io import read_wav


class TestReadWavErrors:
    """Test error messages in read_wav function."""

    def test_file_not_found(self) -> None:
        """Test error message when file doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            read_wav("nonexistent_file.wav")

        error_msg = str(exc_info.value)
        assert "WAV file not found" in error_msg
        assert "nonexistent_file.wav" in error_msg
        assert "Solution" in error_msg
        assert "Verify the file path" in error_msg

    def test_file_not_found_absolute_path(self) -> None:
        """Test error message with absolute path."""
        abs_path = "/tmp/nonexistent_directory/test.wav"

        with pytest.raises(FileNotFoundError) as exc_info:
            read_wav(abs_path)

        error_msg = str(exc_info.value)
        assert "WAV file not found" in error_msg
        assert abs_path in error_msg
        assert "Absolute path" in error_msg

    def test_corrupted_wav_file(self) -> None:
        """Test error message when WAV file is corrupted."""
        # Create a corrupted WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write invalid data
            f.write(b"INVALID WAV DATA")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                read_wav(temp_path)

            error_msg = str(exc_info.value)
            assert "Failed to read WAV file" in error_msg
            assert temp_path in error_msg
            assert "Solution" in error_msg
            assert "valid WAV file" in error_msg
            assert "corrupted" in error_msg
            assert "Background" in error_msg
            assert "RIFF WAVE format" in error_msg
        finally:
            Path(temp_path).unlink()

    @patch("wandas.io.wav_io.requests.get")
    def test_url_download_failure(self, mock_get: Mock) -> None:
        """Test error message when URL download fails."""
        import requests

        mock_get.side_effect = requests.RequestException("Connection timeout")

        with pytest.raises(ValueError) as exc_info:
            read_wav("https://example.com/audio.wav")

        error_msg = str(exc_info.value)
        assert "Failed to download WAV file from URL" in error_msg
        assert "https://example.com/audio.wav" in error_msg
        assert "Connection timeout" in error_msg
        assert "Solution" in error_msg
        assert "Verify the URL is accessible" in error_msg
        assert "Background" in error_msg

    @patch("wandas.io.wav_io.requests.get")
    @patch("wandas.io.wav_io.wavfile.read")
    def test_url_invalid_wav_data(self, mock_wavfile_read: Mock, mock_get: Mock) -> None:
        """Test error message when URL returns invalid WAV data."""
        # Mock successful download but invalid WAV data
        mock_response = Mock()
        mock_response.content = b"INVALID DATA"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        mock_wavfile_read.side_effect = Exception("Invalid file format")

        with pytest.raises(ValueError) as exc_info:
            read_wav("https://example.com/audio.wav")

        error_msg = str(exc_info.value)
        assert "Failed to read WAV data from URL" in error_msg
        assert "https://example.com/audio.wav" in error_msg
        assert "Invalid file format" in error_msg
        assert "Solution" in error_msg
        assert "valid WAV file" in error_msg
        assert "Background" in error_msg

    def test_valid_wav_file_creation(self) -> None:
        """Test that valid WAV files are read successfully."""
        import soundfile as sf

        # Create a valid WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Write valid audio data
            data = np.random.random(16000).astype(np.float32)
            sf.write(temp_path, data, 16000)

            # Should not raise
            cf = read_wav(temp_path)
            assert cf is not None
            assert cf.sampling_rate == 16000
            assert cf.n_samples == 16000
        finally:
            Path(temp_path).unlink()

    def test_current_directory_in_error_message(self) -> None:
        """Test that current directory is shown in error for relative paths."""
        with pytest.raises(FileNotFoundError) as exc_info:
            read_wav("relative/path/to/file.wav")

        error_msg = str(exc_info.value)
        assert "Current directory" in error_msg
