"""Tests for improved error messages (Phase 1)."""
import os
import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.processing.filters import BandPassFilter, HighPassFilter, LowPassFilter
from wandas.processing.temporal import ReSampling


class TestLowPassFilterErrorMessages:
    """Test error messages for LowPassFilter."""

    def test_cutoff_too_low_error_message(self) -> None:
        """Test error message when cutoff is too low."""
        sampling_rate = 16000
        cutoff = 0  # Invalid: too low

        with pytest.raises(ValueError) as exc_info:
            LowPassFilter(sampling_rate, cutoff)

        error_msg = str(exc_info.value)
        assert "too low" in error_msg.lower()
        assert str(cutoff) in error_msg
        assert "Solution:" in error_msg
        assert "Background:" in error_msg
        assert "positive" in error_msg.lower()

    def test_cutoff_too_high_error_message(self) -> None:
        """Test error message when cutoff exceeds Nyquist frequency."""
        sampling_rate = 16000
        cutoff = 10000  # Invalid: above Nyquist (8000 Hz)
        nyquist = sampling_rate / 2

        with pytest.raises(ValueError) as exc_info:
            LowPassFilter(sampling_rate, cutoff)

        error_msg = str(exc_info.value)
        assert "too high" in error_msg.lower()
        assert str(cutoff) in error_msg
        assert str(nyquist) in error_msg
        assert "Solution:" in error_msg
        assert "Background:" in error_msg
        assert "Nyquist" in error_msg
        assert "aliasing" in error_msg.lower()
        # Should suggest reducing cutoff or increasing sampling rate
        assert str(nyquist) in error_msg
        assert str(cutoff * 2) in error_msg


class TestHighPassFilterErrorMessages:
    """Test error messages for HighPassFilter."""

    def test_cutoff_too_low_error_message(self) -> None:
        """Test error message when cutoff is too low."""
        sampling_rate = 16000
        cutoff = 0  # Invalid: too low

        with pytest.raises(ValueError) as exc_info:
            HighPassFilter(sampling_rate, cutoff)

        error_msg = str(exc_info.value)
        assert "too low" in error_msg.lower()
        assert str(cutoff) in error_msg
        assert "Solution:" in error_msg
        assert "Background:" in error_msg

    def test_cutoff_too_high_error_message(self) -> None:
        """Test error message when cutoff exceeds Nyquist frequency."""
        sampling_rate = 16000
        cutoff = 10000  # Invalid: above Nyquist (8000 Hz)
        nyquist = sampling_rate / 2

        with pytest.raises(ValueError) as exc_info:
            HighPassFilter(sampling_rate, cutoff)

        error_msg = str(exc_info.value)
        assert "too high" in error_msg.lower()
        assert str(cutoff) in error_msg
        assert str(nyquist) in error_msg
        assert "Solution:" in error_msg
        assert "Background:" in error_msg
        assert "Nyquist" in error_msg


class TestBandPassFilterErrorMessages:
    """Test error messages for BandPassFilter."""

    def test_low_cutoff_too_low_error_message(self) -> None:
        """Test error message when low cutoff is too low."""
        sampling_rate = 16000
        low_cutoff = 0  # Invalid
        high_cutoff = 1000

        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate, low_cutoff, high_cutoff)

        error_msg = str(exc_info.value)
        assert "lower cutoff" in error_msg.lower()
        assert "too low" in error_msg.lower()
        assert str(low_cutoff) in error_msg
        assert "Solution:" in error_msg

    def test_high_cutoff_too_high_error_message(self) -> None:
        """Test error message when high cutoff is too high."""
        sampling_rate = 16000
        low_cutoff = 300
        high_cutoff = 10000  # Invalid: above Nyquist
        nyquist = sampling_rate / 2

        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate, low_cutoff, high_cutoff)

        error_msg = str(exc_info.value)
        assert "higher cutoff" in error_msg.lower()
        assert "too high" in error_msg.lower()
        assert str(high_cutoff) in error_msg
        assert str(nyquist) in error_msg
        assert "Solution:" in error_msg
        assert "Background:" in error_msg

    def test_low_cutoff_greater_than_high_cutoff_error_message(self) -> None:
        """Test error message when low_cutoff >= high_cutoff."""
        sampling_rate = 16000
        low_cutoff = 1000
        high_cutoff = 500  # Invalid: less than low_cutoff

        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate, low_cutoff, high_cutoff)

        error_msg = str(exc_info.value)
        assert "lower cutoff" in error_msg.lower()
        assert "higher cutoff" in error_msg.lower()
        assert str(low_cutoff) in error_msg
        assert str(high_cutoff) in error_msg
        assert "Solution:" in error_msg
        assert "Background:" in error_msg
        assert "band-pass" in error_msg.lower()


class TestChannelFrameErrorMessages:
    """Test error messages for ChannelFrame initialization."""

    def test_invalid_data_shape_error_message(self) -> None:
        """Test error message when data has invalid shape."""
        data = np.random.random((2, 3, 4))  # 3D array - invalid
        dask_data = da.from_array(data)
        sampling_rate = 16000

        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(dask_data, sampling_rate)

        error_msg = str(exc_info.value)
        assert "1-dimensional or 2-dimensional" in error_msg.lower()
        assert str(data.shape) in error_msg
        assert "Solution:" in error_msg
        assert "Background:" in error_msg
        assert "channels" in error_msg.lower()
        assert "samples" in error_msg.lower()

    def test_invalid_sampling_rate_error_message(self) -> None:
        """Test error message when sampling rate is not positive."""
        data = np.random.random((2, 1000))
        dask_data = da.from_array(data)
        sampling_rate = -100  # Invalid: negative

        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(dask_data, sampling_rate)

        error_msg = str(exc_info.value)
        assert "sampling rate must be positive" in error_msg.lower()
        assert str(sampling_rate) in error_msg
        assert "Solution:" in error_msg
        assert "Background:" in error_msg
        assert "44100" in error_msg or "48000" in error_msg  # Common values

    def test_from_numpy_invalid_shape_error_message(self) -> None:
        """Test error message for from_numpy with invalid shape."""
        data = np.random.random((2, 3, 4))  # 3D array - invalid
        sampling_rate = 16000

        with pytest.raises(ValueError) as exc_info:
            ChannelFrame.from_numpy(data, sampling_rate)

        error_msg = str(exc_info.value)
        assert "1-dimensional or 2-dimensional" in error_msg.lower()
        assert str(data.shape) in error_msg
        assert "Solution:" in error_msg
        assert "Background:" in error_msg

    def test_from_file_not_found_error_message(self) -> None:
        """Test error message when file is not found."""
        nonexistent_file = "/path/to/nonexistent/file.wav"

        with pytest.raises(FileNotFoundError) as exc_info:
            ChannelFrame.from_file(nonexistent_file)

        error_msg = str(exc_info.value)
        assert nonexistent_file in error_msg
        assert "not found" in error_msg.lower()
        assert "Solution:" in error_msg
        assert "check" in error_msg.lower()


class TestResamplingErrorMessages:
    """Test error messages for resampling operation."""

    def test_negative_target_sr_error_message(self) -> None:
        """Test error message when target sampling rate is negative."""
        sampling_rate = 16000
        target_sr = -8000  # Invalid: negative

        with pytest.raises(ValueError) as exc_info:
            ReSampling(sampling_rate, target_sr)

        error_msg = str(exc_info.value)
        assert "must be positive" in error_msg.lower()
        assert str(target_sr) in error_msg
        assert "Solution:" in error_msg
        assert "Background:" in error_msg

    def test_zero_target_sr_error_message(self) -> None:
        """Test error message when target sampling rate is zero."""
        sampling_rate = 16000
        target_sr = 0  # Invalid: zero

        with pytest.raises(ValueError) as exc_info:
            ReSampling(sampling_rate, target_sr)

        error_msg = str(exc_info.value)
        assert "must be positive" in error_msg.lower()
        assert str(target_sr) in error_msg
        assert "Solution:" in error_msg


class TestReadWavErrorMessages:
    """Test error messages for read_wav function."""

    def test_file_not_found_error_message(self) -> None:
        """Test error message when WAV file is not found."""
        from wandas.io.wav_io import read_wav

        nonexistent_file = "/path/to/nonexistent/audio.wav"

        with pytest.raises(FileNotFoundError) as exc_info:
            read_wav(nonexistent_file)

        error_msg = str(exc_info.value)
        assert nonexistent_file in error_msg
        assert "not found" in error_msg.lower()
        assert "Solution:" in error_msg
        assert "absolute path" in error_msg.lower()

    def test_invalid_wav_file_error_message(self) -> None:
        """Test error message when file is not a valid WAV."""
        from wandas.io.wav_io import read_wav

        # Create a temporary non-WAV file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".wav", delete=False
        ) as tmp_file:
            tmp_file.write("This is not a WAV file")
            tmp_path = tmp_file.name

        try:
            with pytest.raises(ValueError) as exc_info:
                read_wav(tmp_path)

            error_msg = str(exc_info.value)
            assert "Failed to read" in error_msg
            assert "Solution:" in error_msg
            assert "valid WAV format" in error_msg
            assert "corrupted" in error_msg.lower()
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
