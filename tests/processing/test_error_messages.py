"""Test error message improvements for medium-priority errors.

This module tests that operations raise appropriate errors with helpful messages
when given invalid parameters.
"""

import numpy as np
import pytest

from wandas.processing.effects import Normalize
from wandas.processing.spectral import FFT, STFT, Welch
from wandas.processing.temporal import ReSampling


class TestFFTErrorMessages:
    """Test FFT error messages for invalid parameters."""

    def test_fft_negative_n_fft(self) -> None:
        """Test that FFT raises error for negative n_fft."""
        with pytest.raises(ValueError, match=r"FFT size must be a positive integer"):
            FFT(sampling_rate=16000, n_fft=-1024)

    def test_fft_zero_n_fft(self) -> None:
        """Test that FFT raises error for zero n_fft."""
        with pytest.raises(ValueError, match=r"FFT size must be a positive integer"):
            FFT(sampling_rate=16000, n_fft=0)

    def test_fft_non_integer_n_fft(self) -> None:
        """Test that FFT raises error for non-integer n_fft."""
        with pytest.raises(ValueError, match=r"FFT size must be an integer"):
            FFT(sampling_rate=16000, n_fft=1024.5)  # type: ignore[arg-type]

    def test_fft_valid_n_fft(self) -> None:
        """Test that FFT accepts valid n_fft values."""
        # Should not raise any errors
        fft_512 = FFT(sampling_rate=16000, n_fft=512)
        assert fft_512.n_fft == 512

        fft_2048 = FFT(sampling_rate=16000, n_fft=2048)
        assert fft_2048.n_fft == 2048

        fft_none = FFT(sampling_rate=16000, n_fft=None)
        assert fft_none.n_fft is None


class TestSTFTErrorMessages:
    """Test STFT error messages for invalid parameters."""

    def test_stft_negative_n_fft(self) -> None:
        """Test that STFT raises error for negative n_fft."""
        with pytest.raises(
            ValueError, match=r"FFT size \(n_fft\) must be a positive integer"
        ):
            STFT(sampling_rate=16000, n_fft=-2048)

    def test_stft_zero_n_fft(self) -> None:
        """Test that STFT raises error for zero n_fft."""
        with pytest.raises(
            ValueError, match=r"FFT size \(n_fft\) must be a positive integer"
        ):
            STFT(sampling_rate=16000, n_fft=0)

    def test_stft_negative_win_length(self) -> None:
        """Test that STFT raises error for negative win_length."""
        with pytest.raises(
            ValueError, match=r"Window length \(win_length\) must be a positive integer"
        ):
            STFT(sampling_rate=16000, n_fft=2048, win_length=-1024)

    def test_stft_zero_win_length(self) -> None:
        """Test that STFT raises error for zero win_length."""
        with pytest.raises(
            ValueError, match=r"Window length \(win_length\) must be a positive integer"
        ):
            STFT(sampling_rate=16000, n_fft=2048, win_length=0)

    def test_stft_win_length_exceeds_n_fft(self) -> None:
        """Test that STFT raises error when win_length > n_fft."""
        with pytest.raises(
            ValueError,
            match=r"Window length \(win_length=4096\) cannot exceed "
            r"FFT size \(n_fft=2048\)",
        ):
            STFT(sampling_rate=16000, n_fft=2048, win_length=4096)

    def test_stft_negative_hop_length(self) -> None:
        """Test that STFT raises error for negative hop_length."""
        with pytest.raises(
            ValueError, match=r"Hop length \(hop_length\) must be a positive integer"
        ):
            STFT(sampling_rate=16000, n_fft=2048, hop_length=-512)

    def test_stft_zero_hop_length(self) -> None:
        """Test that STFT raises error for zero hop_length."""
        with pytest.raises(
            ValueError, match=r"Hop length \(hop_length\) must be a positive integer"
        ):
            STFT(sampling_rate=16000, n_fft=2048, hop_length=0)

    def test_stft_valid_parameters(self) -> None:
        """Test that STFT accepts valid parameter combinations."""
        # Should not raise any errors
        stft_default = STFT(sampling_rate=16000)
        assert stft_default.n_fft == 2048
        assert stft_default.win_length == 2048
        assert stft_default.hop_length == 512

        stft_custom = STFT(
            sampling_rate=16000, n_fft=1024, win_length=1024, hop_length=256
        )
        assert stft_custom.n_fft == 1024
        assert stft_custom.win_length == 1024
        assert stft_custom.hop_length == 256

        # win_length < n_fft is valid
        stft_short_win = STFT(sampling_rate=16000, n_fft=2048, win_length=1024)
        assert stft_short_win.n_fft == 2048
        assert stft_short_win.win_length == 1024


class TestWelchErrorMessages:
    """Test Welch error messages for invalid parameters."""

    def test_welch_negative_n_fft(self) -> None:
        """Test that Welch raises error for negative n_fft."""
        with pytest.raises(
            ValueError, match=r"FFT size \(n_fft\) must be a positive integer"
        ):
            Welch(sampling_rate=16000, n_fft=-2048)

    def test_welch_zero_n_fft(self) -> None:
        """Test that Welch raises error for zero n_fft."""
        with pytest.raises(
            ValueError, match=r"FFT size \(n_fft\) must be a positive integer"
        ):
            Welch(sampling_rate=16000, n_fft=0)

    def test_welch_negative_win_length(self) -> None:
        """Test that Welch raises error for negative win_length."""
        with pytest.raises(
            ValueError, match=r"Window length \(win_length\) must be a positive integer"
        ):
            Welch(sampling_rate=16000, n_fft=2048, win_length=-1024)

    def test_welch_zero_win_length(self) -> None:
        """Test that Welch raises error for zero win_length."""
        with pytest.raises(
            ValueError, match=r"Window length \(win_length\) must be a positive integer"
        ):
            Welch(sampling_rate=16000, n_fft=2048, win_length=0)

    def test_welch_win_length_exceeds_n_fft(self) -> None:
        """Test that Welch raises error when win_length > n_fft."""
        with pytest.raises(
            ValueError,
            match=r"Window length \(win_length=4096\) cannot exceed "
            r"FFT size \(n_fft=2048\)",
        ):
            Welch(sampling_rate=16000, n_fft=2048, win_length=4096)

    def test_welch_negative_hop_length(self) -> None:
        """Test that Welch raises error for negative hop_length."""
        with pytest.raises(
            ValueError, match=r"Hop length \(hop_length\) must be a positive integer"
        ):
            Welch(sampling_rate=16000, n_fft=2048, hop_length=-512)

    def test_welch_zero_hop_length(self) -> None:
        """Test that Welch raises error for zero hop_length."""
        with pytest.raises(
            ValueError, match=r"Hop length \(hop_length\) must be a positive integer"
        ):
            Welch(sampling_rate=16000, n_fft=2048, hop_length=0)

    def test_welch_invalid_average_method(self) -> None:
        """Test that Welch raises error for invalid average method."""
        with pytest.raises(
            ValueError, match=r"Averaging method 'invalid' is not supported"
        ):
            Welch(sampling_rate=16000, average="invalid")

    def test_welch_valid_parameters(self) -> None:
        """Test that Welch accepts valid parameter combinations."""
        # Should not raise any errors
        welch_default = Welch(sampling_rate=16000)
        assert welch_default.n_fft == 2048
        assert welch_default.average == "mean"

        welch_mean = Welch(sampling_rate=16000, average="mean")
        assert welch_mean.average == "mean"

        welch_median = Welch(sampling_rate=16000, average="median")
        assert welch_median.average == "median"


class TestNormalizeErrorMessages:
    """Test Normalize error messages for invalid parameters."""

    def test_normalize_invalid_norm_type(self) -> None:
        """Test that Normalize raises error for invalid norm type."""
        with pytest.raises(
            ValueError,
            match=r"Norm parameter must be a number, np.inf, -np.inf, or None",
        ):
            Normalize(sampling_rate=16000, norm="invalid")  # type: ignore[arg-type]

    def test_normalize_invalid_axis_type(self) -> None:
        """Test that Normalize raises error for invalid axis type."""
        with pytest.raises(
            ValueError, match=r"Axis parameter must be an integer or None"
        ):
            Normalize(sampling_rate=16000, axis="invalid")  # type: ignore[arg-type]

    def test_normalize_valid_norm_values(self) -> None:
        """Test that Normalize accepts valid norm values."""
        # Should not raise any errors
        norm_inf = Normalize(sampling_rate=16000, norm=np.inf)
        assert norm_inf.norm == np.inf

        norm_neg_inf = Normalize(sampling_rate=16000, norm=-np.inf)
        assert norm_neg_inf.norm == -np.inf

        norm_2 = Normalize(sampling_rate=16000, norm=2)
        assert norm_2.norm == 2

        norm_1 = Normalize(sampling_rate=16000, norm=1.0)
        assert norm_1.norm == 1.0

        norm_none = Normalize(sampling_rate=16000, norm=None)
        assert norm_none.norm is None

    def test_normalize_valid_axis_values(self) -> None:
        """Test that Normalize accepts valid axis values."""
        # Should not raise any errors
        norm_axis_neg1 = Normalize(sampling_rate=16000, axis=-1)
        assert norm_axis_neg1.axis == -1

        norm_axis_0 = Normalize(sampling_rate=16000, axis=0)
        assert norm_axis_0.axis == 0

        norm_axis_none = Normalize(sampling_rate=16000, axis=None)
        assert norm_axis_none.axis is None


class TestReSamplingErrorMessages:
    """Test ReSampling error messages for invalid parameters."""

    def test_resampling_negative_target_sr(self) -> None:
        """Test that ReSampling raises error for negative target_sr."""
        with pytest.raises(ValueError, match=r"Target sampling rate must be positive"):
            ReSampling(sampling_rate=16000, target_sr=-8000)

    def test_resampling_zero_target_sr(self) -> None:
        """Test that ReSampling raises error for zero target_sr."""
        with pytest.raises(ValueError, match=r"Target sampling rate must be positive"):
            ReSampling(sampling_rate=16000, target_sr=0)

    def test_resampling_invalid_type_target_sr(self) -> None:
        """Test that ReSampling raises error for invalid type target_sr."""
        with pytest.raises(ValueError, match=r"Target sampling rate must be a number"):
            ReSampling(sampling_rate=16000, target_sr="invalid")  # type: ignore[arg-type]

    def test_resampling_valid_target_sr(self) -> None:
        """Test that ReSampling accepts valid target_sr values."""
        # Should not raise any errors
        resample_8k = ReSampling(sampling_rate=16000, target_sr=8000)
        assert resample_8k.target_sr == 8000

        resample_44k = ReSampling(sampling_rate=16000, target_sr=44100)
        assert resample_44k.target_sr == 44100

        resample_48k = ReSampling(sampling_rate=16000, target_sr=48000)
        assert resample_48k.target_sr == 48000

    def test_resampling_extreme_ratio_warning(self, caplog) -> None:  # type: ignore[no-untyped-def]
        """Test that ReSampling warns for extreme resampling ratios."""
        import logging

        # Test downsampling by more than 10x
        with caplog.at_level(logging.WARNING):
            ReSampling(sampling_rate=48000, target_sr=4000)
        assert "Extreme resampling ratio" in caplog.text

        caplog.clear()

        # Test upsampling by more than 10x
        with caplog.at_level(logging.WARNING):
            ReSampling(sampling_rate=4000, target_sr=48000)
        assert "Extreme resampling ratio" in caplog.text
