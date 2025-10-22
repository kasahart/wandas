"""Tests for filter error messages."""

import pytest

from wandas.processing.filters import BandPassFilter, HighPassFilter, LowPassFilter


class TestLowPassFilterErrors:
    """Test error messages in LowPassFilter."""

    def test_cutoff_zero(self) -> None:
        """Test error message when cutoff is zero."""
        with pytest.raises(ValueError) as exc_info:
            LowPassFilter(sampling_rate=16000, cutoff=0)

        error_msg = str(exc_info.value)
        assert "Cutoff frequency is too low" in error_msg
        assert "Given: 0" in error_msg
        assert "positive number" in error_msg
        assert "Solution" in error_msg
        assert "Background" in error_msg

    def test_cutoff_negative(self) -> None:
        """Test error message when cutoff is negative."""
        with pytest.raises(ValueError) as exc_info:
            LowPassFilter(sampling_rate=16000, cutoff=-100)

        error_msg = str(exc_info.value)
        assert "Cutoff frequency is too low" in error_msg
        assert "Given: -100" in error_msg
        assert "Solution" in error_msg
        assert "Background" in error_msg

    def test_cutoff_above_nyquist(self) -> None:
        """Test error message when cutoff is above Nyquist frequency."""
        with pytest.raises(ValueError) as exc_info:
            LowPassFilter(sampling_rate=16000, cutoff=10000)

        error_msg = str(exc_info.value)
        assert "Cutoff frequency is too high" in error_msg
        assert "Given: 10000" in error_msg
        assert "Nyquist frequency (limit): 8000.0" in error_msg
        assert "Solution" in error_msg
        assert "increase sampling rate" in error_msg
        assert "Background" in error_msg
        assert "aliasing" in error_msg

    def test_cutoff_equal_nyquist(self) -> None:
        """Test error message when cutoff equals Nyquist frequency."""
        with pytest.raises(ValueError) as exc_info:
            LowPassFilter(sampling_rate=16000, cutoff=8000)

        error_msg = str(exc_info.value)
        assert "Cutoff frequency is too high" in error_msg
        assert "Given: 8000" in error_msg
        assert "Nyquist frequency (limit): 8000.0" in error_msg

    def test_valid_cutoff(self) -> None:
        """Test that valid cutoff values are accepted."""
        # Should not raise
        lpf = LowPassFilter(sampling_rate=16000, cutoff=4000)
        assert lpf.cutoff == 4000


class TestHighPassFilterErrors:
    """Test error messages in HighPassFilter."""

    def test_cutoff_zero(self) -> None:
        """Test error message when cutoff is zero."""
        with pytest.raises(ValueError) as exc_info:
            HighPassFilter(sampling_rate=16000, cutoff=0)

        error_msg = str(exc_info.value)
        assert "Cutoff frequency is too low" in error_msg
        assert "Given: 0" in error_msg
        assert "positive number" in error_msg
        assert "Solution" in error_msg
        assert "Background" in error_msg

    def test_cutoff_negative(self) -> None:
        """Test error message when cutoff is negative."""
        with pytest.raises(ValueError) as exc_info:
            HighPassFilter(sampling_rate=16000, cutoff=-50)

        error_msg = str(exc_info.value)
        assert "Cutoff frequency is too low" in error_msg
        assert "Given: -50" in error_msg

    def test_cutoff_above_nyquist(self) -> None:
        """Test error message when cutoff is above Nyquist frequency."""
        with pytest.raises(ValueError) as exc_info:
            HighPassFilter(sampling_rate=16000, cutoff=10000)

        error_msg = str(exc_info.value)
        assert "Cutoff frequency is too high" in error_msg
        assert "Given: 10000" in error_msg
        assert "Nyquist frequency (limit): 8000.0" in error_msg
        assert "Solution" in error_msg
        assert "Background" in error_msg
        assert "aliasing" in error_msg

    def test_valid_cutoff(self) -> None:
        """Test that valid cutoff values are accepted."""
        # Should not raise
        hpf = HighPassFilter(sampling_rate=16000, cutoff=100)
        assert hpf.cutoff == 100


class TestBandPassFilterErrors:
    """Test error messages in BandPassFilter."""

    def test_low_cutoff_zero(self) -> None:
        """Test error message when low cutoff is zero."""
        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate=16000, low_cutoff=0, high_cutoff=4000)

        error_msg = str(exc_info.value)
        assert "Lower cutoff frequency is too low" in error_msg
        assert "Given: 0" in error_msg
        assert "Solution" in error_msg
        assert "Background" in error_msg

    def test_low_cutoff_negative(self) -> None:
        """Test error message when low cutoff is negative."""
        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate=16000, low_cutoff=-100, high_cutoff=4000)

        error_msg = str(exc_info.value)
        assert "Lower cutoff frequency is too low" in error_msg
        assert "Given: -100" in error_msg

    def test_low_cutoff_above_nyquist(self) -> None:
        """Test error message when low cutoff is above Nyquist."""
        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate=16000, low_cutoff=10000, high_cutoff=12000)

        error_msg = str(exc_info.value)
        assert "Lower cutoff frequency is too high" in error_msg
        assert "Given: 10000" in error_msg
        assert "Nyquist frequency (limit): 8000.0" in error_msg
        assert "aliasing" in error_msg

    def test_high_cutoff_zero(self) -> None:
        """Test error message when high cutoff is zero."""
        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate=16000, low_cutoff=100, high_cutoff=0)

        error_msg = str(exc_info.value)
        assert "Higher cutoff frequency is too low" in error_msg
        assert "Given: 0" in error_msg

    def test_high_cutoff_negative(self) -> None:
        """Test error message when high cutoff is negative."""
        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate=16000, low_cutoff=100, high_cutoff=-1000)

        error_msg = str(exc_info.value)
        assert "Higher cutoff frequency is too low" in error_msg
        assert "Given: -1000" in error_msg

    def test_high_cutoff_above_nyquist(self) -> None:
        """Test error message when high cutoff is above Nyquist."""
        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate=16000, low_cutoff=100, high_cutoff=10000)

        error_msg = str(exc_info.value)
        assert "Higher cutoff frequency is too high" in error_msg
        assert "Given: 10000" in error_msg
        assert "Nyquist frequency (limit): 8000.0" in error_msg

    def test_low_cutoff_greater_than_high_cutoff(self) -> None:
        """Test error message when low cutoff >= high cutoff."""
        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate=16000, low_cutoff=4000, high_cutoff=2000)

        error_msg = str(exc_info.value)
        assert "Invalid cutoff frequency range" in error_msg
        assert "Lower cutoff: 4000" in error_msg
        assert "Higher cutoff: 2000" in error_msg
        assert "Solution" in error_msg
        assert "low_cutoff < high_cutoff" in error_msg
        assert "Background" in error_msg

    def test_low_cutoff_equal_high_cutoff(self) -> None:
        """Test error message when low cutoff == high cutoff."""
        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(sampling_rate=16000, low_cutoff=2000, high_cutoff=2000)

        error_msg = str(exc_info.value)
        assert "Invalid cutoff frequency range" in error_msg
        assert "Lower cutoff: 2000" in error_msg
        assert "Higher cutoff: 2000" in error_msg

    def test_valid_cutoffs(self) -> None:
        """Test that valid cutoff values are accepted."""
        # Should not raise
        bpf = BandPassFilter(sampling_rate=16000, low_cutoff=100, high_cutoff=4000)
        assert bpf.low_cutoff == 100
        assert bpf.high_cutoff == 4000
