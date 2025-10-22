"""Tests for improved error messages in filter operations."""

import pytest

from wandas.processing.filters import (
    BandPassFilter,
    HighPassFilter,
    LowPassFilter,
)


class TestLowPassFilterErrors:
    """Test error messages for LowPassFilter."""

    def test_cutoff_too_low_error_message(self) -> None:
        """Test that zero or negative cutoff raises informative error."""
        with pytest.raises(ValueError) as exc_info:
            LowPassFilter(sampling_rate=16000, cutoff=0)

        error_msg = str(exc_info.value)
        assert "Cutoff frequency is too low" in error_msg
        assert "Given: 0" in error_msg
        assert "Minimum: > 0 Hz" in error_msg
        assert "Solution" in error_msg
        assert "positive cutoff frequency" in error_msg
        assert "Background" in error_msg

    def test_negative_cutoff_error_message(self) -> None:
        """Test that negative cutoff raises informative error."""
        with pytest.raises(ValueError) as exc_info:
            LowPassFilter(sampling_rate=16000, cutoff=-100)

        error_msg = str(exc_info.value)
        assert "Cutoff frequency is too low" in error_msg
        assert "Given: -100" in error_msg

    def test_cutoff_exceeds_nyquist_error_message(self) -> None:
        """Test that cutoff above Nyquist raises informative error."""
        with pytest.raises(ValueError) as exc_info:
            LowPassFilter(sampling_rate=16000, cutoff=10000)

        error_msg = str(exc_info.value)
        assert "exceeds Nyquist limit" in error_msg
        assert "Given: 10000" in error_msg
        assert "Nyquist frequency (limit): 8000" in error_msg
        assert "Sampling rate: 16000" in error_msg
        assert "Solution" in error_msg
        assert "Use cutoff < 8000" in error_msg
        assert "resample" in error_msg
        assert "Background" in error_msg
        assert "aliasing" in error_msg

    def test_cutoff_at_nyquist_error_message(self) -> None:
        """Test that cutoff at Nyquist frequency raises informative error."""
        with pytest.raises(ValueError) as exc_info:
            LowPassFilter(sampling_rate=16000, cutoff=8000)

        error_msg = str(exc_info.value)
        assert "exceeds Nyquist limit" in error_msg
        assert "Given: 8000" in error_msg

    def test_valid_cutoff_no_error(self) -> None:
        """Test that valid cutoff does not raise error."""
        lpf = LowPassFilter(sampling_rate=16000, cutoff=1000)
        assert lpf.cutoff == 1000
        assert lpf.sampling_rate == 16000


class TestHighPassFilterErrors:
    """Test error messages for HighPassFilter."""

    def test_cutoff_too_low_error_message(self) -> None:
        """Test that zero or negative cutoff raises informative error."""
        with pytest.raises(ValueError) as exc_info:
            HighPassFilter(sampling_rate=16000, cutoff=0)

        error_msg = str(exc_info.value)
        assert "Cutoff frequency is too low" in error_msg
        assert "Given: 0" in error_msg
        assert "Solution" in error_msg
        assert "Background" in error_msg

    def test_cutoff_exceeds_nyquist_error_message(self) -> None:
        """Test that cutoff above Nyquist raises informative error."""
        with pytest.raises(ValueError) as exc_info:
            HighPassFilter(sampling_rate=16000, cutoff=10000)

        error_msg = str(exc_info.value)
        assert "exceeds Nyquist limit" in error_msg
        assert "Given: 10000" in error_msg
        assert "Nyquist frequency (limit): 8000" in error_msg
        assert "resample" in error_msg
        assert "aliasing" in error_msg

    def test_valid_cutoff_no_error(self) -> None:
        """Test that valid cutoff does not raise error."""
        hpf = HighPassFilter(sampling_rate=16000, cutoff=100)
        assert hpf.cutoff == 100
        assert hpf.sampling_rate == 16000


class TestBandPassFilterErrors:
    """Test error messages for BandPassFilter."""

    def test_low_cutoff_too_low_error_message(self) -> None:
        """Test that zero or negative low cutoff raises informative error."""
        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate=16000, low_cutoff=0, high_cutoff=1000)

        error_msg = str(exc_info.value)
        assert "Lower cutoff frequency is too low" in error_msg
        assert "Given: 0" in error_msg
        assert "Solution" in error_msg
        assert "Background" in error_msg

    def test_low_cutoff_exceeds_nyquist_error_message(self) -> None:
        """Test that low cutoff above Nyquist raises informative error."""
        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate=16000, low_cutoff=10000, high_cutoff=12000)

        error_msg = str(exc_info.value)
        assert "Lower cutoff frequency exceeds Nyquist limit" in error_msg
        assert "Given: 10000" in error_msg
        assert "Nyquist frequency (limit): 8000" in error_msg

    def test_high_cutoff_too_low_error_message(self) -> None:
        """Test that zero or negative high cutoff raises informative error."""
        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate=16000, low_cutoff=100, high_cutoff=0)

        error_msg = str(exc_info.value)
        assert "Higher cutoff frequency is too low" in error_msg
        assert "Given: 0" in error_msg

    def test_high_cutoff_exceeds_nyquist_error_message(self) -> None:
        """Test that high cutoff above Nyquist raises informative error."""
        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate=16000, low_cutoff=100, high_cutoff=10000)

        error_msg = str(exc_info.value)
        assert "Higher cutoff frequency exceeds Nyquist limit" in error_msg
        assert "Given: 10000" in error_msg

    def test_low_cutoff_greater_than_high_cutoff_error_message(self) -> None:
        """Test that low cutoff >= high cutoff raises informative error."""
        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate=16000, low_cutoff=2000, high_cutoff=1000)

        error_msg = str(exc_info.value)
        assert "Invalid cutoff frequency range" in error_msg
        assert "Lower cutoff: 2000" in error_msg
        assert "Higher cutoff: 1000" in error_msg
        assert "Solution" in error_msg
        assert "lower cutoff < higher cutoff" in error_msg
        assert "Background" in error_msg

    def test_equal_cutoffs_error_message(self) -> None:
        """Test that equal cutoffs raise informative error."""
        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate=16000, low_cutoff=1000, high_cutoff=1000)

        error_msg = str(exc_info.value)
        assert "Invalid cutoff frequency range" in error_msg
        assert "Lower cutoff: 1000" in error_msg
        assert "Higher cutoff: 1000" in error_msg

    def test_valid_cutoffs_no_error(self) -> None:
        """Test that valid cutoffs do not raise error."""
        bpf = BandPassFilter(sampling_rate=16000, low_cutoff=300, high_cutoff=3000)
        assert bpf.low_cutoff == 300
        assert bpf.high_cutoff == 3000
        assert bpf.sampling_rate == 16000
