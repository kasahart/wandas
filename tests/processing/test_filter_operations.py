from unittest import mock

import numpy as np
import pytest
import scipy.signal as signal
from dask.array.core import Array as DaArray

from wandas.processing.base import create_operation, get_operation
from wandas.processing.filters import (
    AWeighting,
    BandPassFilter,
    HighPassFilter,
    LowPassFilter,
)

# ---------------------------------------------------------------------------
# Constants shared across filter tests
# ---------------------------------------------------------------------------
_SR: int = 16000
_CUTOFF_HPF = 500.0
_ORDER = 4
_LOW_FREQ = 50.0  # below typical cutoff — must be attenuated by HPF/LPF
_HIGH_FREQ = 1000.0  # above typical cutoff — must be preserved by HPF


class TestHighPassFilter:
    """High-pass filter: Layer 1 (unit) + Layer 2 (domain) + Layer 3 (wrapper)."""

    # -- Layer 1: Unit tests (input validation) ----------------------------

    def test_highpass_init_default_order_is_four(self) -> None:
        """Test HPF default order parameter."""
        hpf = HighPassFilter(_SR, _CUTOFF_HPF)
        assert hpf.sampling_rate == _SR
        assert hpf.cutoff == _CUTOFF_HPF
        assert hpf.order == 4  # documented default

    def test_highpass_init_custom_order_stored(self) -> None:
        """Test HPF stores custom order."""
        hpf = HighPassFilter(_SR, _CUTOFF_HPF, order=6)
        assert hpf.order == 6

    def test_highpass_cutoff_zero_raises_error(self) -> None:
        """Test that cutoff=0 is rejected."""
        with pytest.raises(ValueError):
            HighPassFilter(_SR, 0)

    def test_highpass_cutoff_above_nyquist_raises_error(self) -> None:
        """Test that cutoff above Nyquist is rejected."""
        with pytest.raises(ValueError):
            HighPassFilter(_SR, _SR / 2 + 1)

    def test_highpass_cutoff_above_nyquist_error_message_what_why_how(self) -> None:
        """Test WHAT/WHY/HOW structure of cutoff error message."""
        invalid_cutoff = 10000.0
        nyquist = _SR / 2

        with pytest.raises(ValueError) as exc_info:
            HighPassFilter(_SR, invalid_cutoff)

        error_msg = str(exc_info.value)
        # WHAT
        assert "Cutoff frequency out of valid range" in error_msg
        assert f"{invalid_cutoff}" in error_msg
        # WHY
        assert "Nyquist" in error_msg
        assert f"{nyquist}" in error_msg
        # Check HOW
        assert "Solutions:" in error_msg
        assert "resample" in error_msg.lower()

    # -- Layer 2: Domain tests (physics) -----------------------------------

    def test_highpass_composite_attenuates_below_cutoff(self, composite_50hz_1khz_dask: tuple[DaArray, int]) -> None:
        """50 Hz component (below 500 Hz cutoff) must be attenuated >20 dB."""
        dask_input, sr = composite_50hz_1khz_dask
        hpf = HighPassFilter(sr, _CUTOFF_HPF, _ORDER)
        input_copy = dask_input.compute().copy()

        # Act
        result_da = hpf.process(dask_input)

        # Assert 1: Immutability — input unchanged
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)

        # Assert 2: Dask graph preserved
        assert isinstance(result_da, DaArray)

        # Assert 3: Value — frequency-domain verification
        result = result_da.compute()
        fft_orig = np.abs(np.fft.rfft(input_copy[0]))
        fft_filt = np.abs(np.fft.rfft(result[0]))
        freq_bins = np.fft.rfftfreq(sr, 1 / sr)

        low_idx = np.argmin(np.abs(freq_bins - _LOW_FREQ))
        high_idx = np.argmin(np.abs(freq_bins - _HIGH_FREQ))

        # 50 Hz (below cutoff): >20 dB attenuation (ratio < 0.1)
        assert fft_filt[low_idx] < 0.1 * fft_orig[low_idx], (
            "HPF must attenuate 50 Hz component by >20 dB (4th-order Butterworth)"
        )
        # 1000 Hz (above cutoff): <1 dB attenuation (ratio > 0.89)
        assert fft_filt[high_idx] > 0.89 * fft_orig[high_idx], "HPF must preserve 1000 Hz component within 1 dB"

    # -- Layer 3: Integration test (wrapper equivalence) --------------------

    def test_highpass_matches_scipy_filtfilt(self, composite_50hz_1khz_dask: tuple[DaArray, int]) -> None:
        """HPF result must exactly match direct scipy.signal.filtfilt call."""
        dask_input, sr = composite_50hz_1khz_dask
        hpf = HighPassFilter(sr, _CUTOFF_HPF, _ORDER)

        result = hpf.process(dask_input).compute()

        # Reference: same Butterworth coefficients + filtfilt
        nyquist = sr / 2
        b, a = signal.butter(_ORDER, _CUTOFF_HPF / nyquist, btype="high")
        raw_input = dask_input.compute()
        expected = signal.filtfilt(b, a, raw_input, axis=1)

        # Same algorithm, exact numeric result expected
        np.testing.assert_allclose(result, expected)


class TestLowPassFilter:
    """Low-pass filter: Layer 1 (unit) + Layer 2 (domain) + Layer 3 (wrapper)."""

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_lowpass_init_default_order_is_four(self) -> None:
        """Test LPF default order parameter."""
        lpf = LowPassFilter(_SR, _CUTOFF_HPF)
        assert lpf.sampling_rate == _SR
        assert lpf.cutoff == _CUTOFF_HPF
        assert lpf.order == 4  # documented default

    def test_lowpass_init_custom_order_stored(self) -> None:
        """Test LPF stores custom order."""
        lpf = LowPassFilter(_SR, _CUTOFF_HPF, order=6)
        assert lpf.order == 6

    def test_lowpass_cutoff_zero_raises_error(self) -> None:
        """Test that cutoff=0 is rejected."""
        with pytest.raises(ValueError):
            LowPassFilter(_SR, 0)

    def test_lowpass_cutoff_above_nyquist_raises_error(self) -> None:
        """Test that cutoff above Nyquist is rejected."""
        with pytest.raises(ValueError):
            LowPassFilter(_SR, _SR / 2 + 1)

    def test_lowpass_cutoff_above_nyquist_error_message_what_why_how(self) -> None:
        """Test WHAT/WHY/HOW structure of cutoff error message."""
        invalid_cutoff = 10000.0
        nyquist = _SR / 2

        with pytest.raises(ValueError) as exc_info:
            LowPassFilter(_SR, invalid_cutoff)

        error_msg = str(exc_info.value)
        assert "Cutoff frequency out of valid range" in error_msg
        assert f"{invalid_cutoff}" in error_msg
        assert "Nyquist" in error_msg
        assert f"{nyquist}" in error_msg
        assert "Solutions:" in error_msg
        assert "resample" in error_msg.lower()

    # -- Layer 2: Domain tests (physics) -----------------------------------

    def test_lowpass_composite_attenuates_above_cutoff(self, composite_50hz_1khz_dask: tuple[DaArray, int]) -> None:
        """1000 Hz component (above 500 Hz cutoff) must be attenuated >20 dB."""
        dask_input, sr = composite_50hz_1khz_dask
        lpf = LowPassFilter(sr, _CUTOFF_HPF, _ORDER)
        input_copy = dask_input.compute().copy()

        # Act
        result_da = lpf.process(dask_input)

        # Assert 1: Immutability
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)

        # Assert 2: Dask graph preserved
        assert isinstance(result_da, DaArray)

        # Assert 3: Value — frequency-domain verification
        result = result_da.compute()
        fft_orig = np.abs(np.fft.rfft(input_copy[0]))
        fft_filt = np.abs(np.fft.rfft(result[0]))
        freq_bins = np.fft.rfftfreq(sr, 1 / sr)

        low_idx = np.argmin(np.abs(freq_bins - _LOW_FREQ))
        high_idx = np.argmin(np.abs(freq_bins - _HIGH_FREQ))

        # 50 Hz (below cutoff): preserved within 1 dB
        assert fft_filt[low_idx] > 0.89 * fft_orig[low_idx], "LPF must preserve 50 Hz component within 1 dB"
        # 1000 Hz (above cutoff): >20 dB attenuation
        assert fft_filt[high_idx] < 0.1 * fft_orig[high_idx], (
            "LPF must attenuate 1000 Hz component by >20 dB (4th-order Butterworth)"
        )

    # -- Layer 3: Integration test (wrapper equivalence) --------------------

    def test_lowpass_matches_scipy_filtfilt(self, composite_50hz_1khz_dask: tuple[DaArray, int]) -> None:
        """LPF result must exactly match direct scipy.signal.filtfilt call."""
        dask_input, sr = composite_50hz_1khz_dask
        lpf = LowPassFilter(sr, _CUTOFF_HPF, _ORDER)

        result = lpf.process(dask_input).compute()

        # Reference: same Butterworth coefficients + filtfilt
        nyquist = sr / 2
        b, a = signal.butter(_ORDER, _CUTOFF_HPF / nyquist, btype="low")
        raw_input = dask_input.compute()
        expected = signal.filtfilt(b, a, raw_input, axis=1)

        # Same algorithm, exact numeric result expected
        np.testing.assert_allclose(result, expected)


class TestAWeightingOperation:
    """A-weighting filter: Layer 1 (unit) + Layer 2 (domain) + Layer 3 (wrapper)."""

    _AW_SR = 300000  # High sr needed to capture A-weighting curve up to 20 kHz

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_aweighting_init_stores_sample_rate(self) -> None:
        """Test AWeighting stores sampling rate."""
        a_weight = AWeighting(self._AW_SR)
        assert a_weight.sampling_rate == self._AW_SR

    def test_aweighting_registry_returns_correct_class(self) -> None:
        """Test AWeighting is registered as 'a_weighting'."""
        assert get_operation("a_weighting") == AWeighting

        a_weight_op = create_operation("a_weighting", self._AW_SR)
        assert isinstance(a_weight_op, AWeighting)
        assert a_weight_op.sampling_rate == self._AW_SR

    # -- Layer 2: Domain tests (IEC 61672 / ANSI S1.4) ---------------------

    def test_aweighting_impulse_frequency_response_100hz_1khz_10khz(
        self, impulse_highsr_dask: tuple[DaArray, int]
    ) -> None:
        """Verify A-weighting frequency response at key frequencies.

        Reference: IEC 61672-1 Table A.1 approximate values:
        - 100 Hz: ~-19.1 dB
        - 1000 Hz: 0 dB (reference)
        - 10000 Hz: ~-2.5 dB
        """
        dask_input, sr = impulse_highsr_dask
        a_weight = AWeighting(sr)
        input_copy = dask_input.compute().copy()

        # Act
        result_da = a_weight.process(dask_input)

        # Assert 1: Immutability
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)

        # Assert 2: Dask graph preserved
        assert isinstance(result_da, DaArray)

        # Assert 3: Frequency response values
        result = result_da.compute()
        assert result.shape == input_copy.shape

        fft_orig = np.abs(np.fft.rfft(input_copy[0]))
        fft_filt = np.abs(np.fft.rfft(result[0]))
        freq_bins = np.fft.rfftfreq(sr, 1 / sr)

        low_idx = np.argmin(np.abs(freq_bins - 100.0))
        mid_idx = np.argmin(np.abs(freq_bins - 1000.0))
        high_idx = np.argmin(np.abs(freq_bins - 10000.0))

        # 100 Hz: ~-19 dB (IEC 61672-1 nominal: -19.1 dB, int truncation gives -19)
        gain_100 = 20 * np.log10(fft_filt[low_idx] / fft_orig[low_idx])
        np.testing.assert_allclose(
            int(gain_100),
            -19,
            atol=1,  # 1 dB tolerance for IEC approximation
        )

        # 1000 Hz: 0 dB reference (A-weighting is defined as 0 dB at 1 kHz)
        gain_1k = 20 * np.log10(fft_filt[mid_idx] / fft_orig[mid_idx])
        np.testing.assert_allclose(
            int(gain_1k),
            0,
            atol=1,  # 1 dB tolerance — 0 dB by definition
        )

        # 10000 Hz: ~-2.5 dB
        gain_10k = 20 * np.log10(fft_filt[high_idx] / fft_orig[high_idx])
        np.testing.assert_allclose(
            gain_10k * 10,
            -2.5 * 10,
            atol=5,  # 0.5 dB tolerance scaled x10
        )

    def test_aweighting_process_preserves_dask_laziness(self, impulse_highsr_dask: tuple[DaArray, int]) -> None:
        """Verify that process() does not eagerly compute the Dask graph."""
        dask_input, sr = impulse_highsr_dask
        a_weight = AWeighting(sr)

        with mock.patch.object(DaArray, "compute", return_value=dask_input.compute()) as mock_compute:
            result = a_weight.process(dask_input)
            mock_compute.assert_not_called()
            result.compute()
            mock_compute.assert_called_once()


class TestBandPassFilter:
    """Bandpass filter: Layer 1 (unit) + Layer 2 (domain) + Layer 3 (wrapper)."""

    _BPF_LOW = 300.0
    _BPF_HIGH = 1000.0

    # -- Layer 1: Unit tests ------------------------------------------------

    def test_bandpass_init_default_order_is_four(self) -> None:
        """Test BPF default order parameter."""
        bpf = BandPassFilter(_SR, self._BPF_LOW, self._BPF_HIGH)
        assert bpf.sampling_rate == _SR
        assert bpf.low_cutoff == self._BPF_LOW
        assert bpf.high_cutoff == self._BPF_HIGH
        assert bpf.order == 4  # documented default

    def test_bandpass_init_custom_order_stored(self) -> None:
        """Test BPF stores custom order."""
        bpf = BandPassFilter(_SR, self._BPF_LOW, self._BPF_HIGH, order=6)
        assert bpf.order == 6

    def test_bandpass_low_cutoff_zero_raises_error(self) -> None:
        """Test that low_cutoff=0 is rejected."""
        with pytest.raises(ValueError):
            BandPassFilter(_SR, 0, self._BPF_HIGH)

    def test_bandpass_high_cutoff_above_nyquist_raises_error(self) -> None:
        """Test that high_cutoff above Nyquist is rejected."""
        with pytest.raises(ValueError):
            BandPassFilter(_SR, self._BPF_LOW, _SR / 2 + 1)

    def test_bandpass_inverted_cutoffs_raises_error(self) -> None:
        """Test that low > high cutoff is rejected."""
        with pytest.raises(ValueError):
            BandPassFilter(_SR, 1000, 500)

    def test_bandpass_inverted_cutoffs_error_message_what_why_how(self) -> None:
        """Test WHAT/WHY/HOW structure of inverted cutoff error."""
        low, high = 1000.0, 500.0

        with pytest.raises(ValueError) as exc_info:
            BandPassFilter(_SR, low, high)

        error_msg = str(exc_info.value)
        assert "Invalid bandpass filter" in error_msg
        assert f"{low}" in error_msg
        assert f"{high}" in error_msg
        assert "Lower cutoff must be less than higher cutoff" in error_msg
        assert "bandpass filter passes frequencies between" in error_msg
        assert "low_cutoff < high_cutoff" in error_msg

    def test_bandpass_registry_returns_correct_class(self) -> None:
        """Test BandPassFilter is registered as 'bandpass_filter'."""
        assert get_operation("bandpass_filter") == BandPassFilter

        bpf_op = create_operation(
            "bandpass_filter",
            _SR,
            low_cutoff=self._BPF_LOW,
            high_cutoff=self._BPF_HIGH,
        )
        assert isinstance(bpf_op, BandPassFilter)
        assert bpf_op.sampling_rate == _SR
        assert bpf_op.low_cutoff == self._BPF_LOW
        assert bpf_op.high_cutoff == self._BPF_HIGH

    # -- Layer 2: Domain tests (physics) -----------------------------------

    def test_bandpass_composite_passes_inband_attenuates_outband(
        self, composite_100_500_1500hz_dask: tuple[DaArray, int]
    ) -> None:
        """500 Hz (in-band) preserved; 100 Hz and 1500 Hz (out-of-band) attenuated >20 dB."""
        dask_input, sr = composite_100_500_1500hz_dask
        bpf = BandPassFilter(sr, self._BPF_LOW, self._BPF_HIGH, _ORDER)
        input_copy = dask_input.compute().copy()

        # Act
        result_da = bpf.process(dask_input)

        # Assert 1: Immutability
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)

        # Assert 2: Dask graph preserved
        assert isinstance(result_da, DaArray)

        # Assert 3: Value — frequency-domain verification
        result = result_da.compute()
        fft_orig = np.abs(np.fft.rfft(input_copy[0]))
        fft_filt = np.abs(np.fft.rfft(result[0]))
        freq_bins = np.fft.rfftfreq(sr, 1 / sr)

        below_idx = np.argmin(np.abs(freq_bins - 100.0))
        in_idx = np.argmin(np.abs(freq_bins - 500.0))
        above_idx = np.argmin(np.abs(freq_bins - 1500.0))

        # 100 Hz (below band): >20 dB attenuation
        assert fft_filt[below_idx] < 0.1 * fft_orig[below_idx], "BPF must attenuate 100 Hz (below band) by >20 dB"
        # 500 Hz (in-band): preserved within 1 dB
        assert fft_filt[in_idx] > 0.89 * fft_orig[in_idx], "BPF must preserve 500 Hz (in-band) within 1 dB"
        # 1500 Hz (above band): >20 dB attenuation
        assert fft_filt[above_idx] < 0.1 * fft_orig[above_idx], "BPF must attenuate 1500 Hz (above band) by >20 dB"

    # -- Layer 3: Integration test (wrapper equivalence) --------------------

    def test_bandpass_matches_scipy_filtfilt(self, composite_100_500_1500hz_dask: tuple[DaArray, int]) -> None:
        """BPF result must exactly match direct scipy.signal.filtfilt call."""
        dask_input, sr = composite_100_500_1500hz_dask
        bpf = BandPassFilter(sr, self._BPF_LOW, self._BPF_HIGH, _ORDER)

        result = bpf.process(dask_input).compute()

        # Reference: same Butterworth coefficients + filtfilt
        nyquist = sr / 2
        b, a = signal.butter(_ORDER, [self._BPF_LOW / nyquist, self._BPF_HIGH / nyquist], btype="band")
        raw_input = dask_input.compute()
        expected = signal.filtfilt(b, a, raw_input, axis=1)

        # Same algorithm, exact numeric result expected
        np.testing.assert_allclose(result, expected)
