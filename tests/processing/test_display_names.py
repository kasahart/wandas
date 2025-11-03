"""Tests for display names in all audio operations."""


from wandas.processing.effects import (
    AddWithSNR,
    Fade,
    HpssHarmonic,
    HpssPercussive,
    Normalize,
    RemoveDC,
)
from wandas.processing.filters import (
    AWeighting,
    BandPassFilter,
    HighPassFilter,
    LowPassFilter,
)
from wandas.processing.spectral import (
    CSD,
    FFT,
    IFFT,
    ISTFT,
    STFT,
    Coherence,
    NOctSpectrum,
    NOctSynthesis,
    TransferFunction,
    Welch,
)
from wandas.processing.stats import ABS, Mean, Power, Sum, ChannelDifference
from wandas.processing.temporal import FixLength, ReSampling, RmsTrend, Trim


class TestFilterDisplayNames:
    """Test display names for filter operations."""

    def test_lowpass_filter_display_name(self) -> None:
        """Test that LowPassFilter returns 'lpf' as display name."""
        op = LowPassFilter(sampling_rate=44100, cutoff=1000)
        assert op.get_display_name() == "lpf"

    def test_highpass_filter_display_name(self) -> None:
        """Test that HighPassFilter returns 'hpf' as display name."""
        op = HighPassFilter(sampling_rate=44100, cutoff=1000)
        assert op.get_display_name() == "hpf"

    def test_bandpass_filter_display_name(self) -> None:
        """Test that BandPassFilter returns 'bpf' as display name."""
        op = BandPassFilter(sampling_rate=44100, low_cutoff=500, high_cutoff=2000)
        assert op.get_display_name() == "bpf"

    def test_a_weighting_display_name(self) -> None:
        """Test that AWeighting returns 'Aw' as display name."""
        op = AWeighting(sampling_rate=44100)
        assert op.get_display_name() == "Aw"


class TestSpectralDisplayNames:
    """Test display names for spectral operations."""

    def test_fft_display_name(self) -> None:
        """Test that FFT returns 'FFT' as display name."""
        op = FFT(sampling_rate=44100)
        assert op.get_display_name() == "FFT"

    def test_ifft_display_name(self) -> None:
        """Test that IFFT returns 'iFFT' as display name."""
        op = IFFT(sampling_rate=44100)
        assert op.get_display_name() == "iFFT"

    def test_stft_display_name(self) -> None:
        """Test that STFT returns 'STFT' as display name."""
        op = STFT(sampling_rate=44100)
        assert op.get_display_name() == "STFT"

    def test_istft_display_name(self) -> None:
        """Test that ISTFT returns 'iSTF' as display name."""
        op = ISTFT(sampling_rate=44100)
        assert op.get_display_name() == "iSTF"

    def test_welch_display_name(self) -> None:
        """Test that Welch returns 'PSD' as display name."""
        op = Welch(sampling_rate=44100)
        assert op.get_display_name() == "PSD"

    def test_noct_spectrum_display_name(self) -> None:
        """Test that NOctSpectrum returns 'Oct' as display name."""
        op = NOctSpectrum(sampling_rate=44100, fmin=20, fmax=20000)
        assert op.get_display_name() == "Oct"

    def test_noct_synthesis_display_name(self) -> None:
        """Test that NOctSynthesis returns 'iOct' as display name."""
        op = NOctSynthesis(sampling_rate=44100, fmin=20, fmax=20000)
        assert op.get_display_name() == "iOct"

    def test_coherence_display_name(self) -> None:
        """Test that Coherence returns 'Coh' as display name."""
        op = Coherence(
            sampling_rate=44100,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window="hann",
            detrend="constant",
        )
        assert op.get_display_name() == "Coh"

    def test_csd_display_name(self) -> None:
        """Test that CSD returns 'CSD' as display name."""
        op = CSD(
            sampling_rate=44100,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window="hann",
            detrend="constant",
            scaling="spectrum",
            average="mean",
        )
        assert op.get_display_name() == "CSD"

    def test_transfer_function_display_name(self) -> None:
        """Test that TransferFunction returns 'H' as display name."""
        op = TransferFunction(
            sampling_rate=44100,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window="hann",
            detrend="constant",
        )
        assert op.get_display_name() == "H"


class TestEffectDisplayNames:
    """Test display names for effect operations."""

    def test_hpss_harmonic_display_name(self) -> None:
        """Test that HpssHarmonic returns 'Hrm' as display name."""
        op = HpssHarmonic(sampling_rate=44100)
        assert op.get_display_name() == "Hrm"

    def test_hpss_percussive_display_name(self) -> None:
        """Test that HpssPercussive returns 'Prc' as display name."""
        op = HpssPercussive(sampling_rate=44100)
        assert op.get_display_name() == "Prc"

    def test_normalize_display_name(self) -> None:
        """Test that Normalize returns 'norm' as display name."""
        op = Normalize(sampling_rate=44100)
        assert op.get_display_name() == "norm"

    def test_remove_dc_display_name(self) -> None:
        """Test that RemoveDC returns 'dcRM' as display name."""
        op = RemoveDC(sampling_rate=44100)
        assert op.get_display_name() == "dcRM"

    def test_add_with_snr_display_name(self) -> None:
        """Test that AddWithSNR returns '+SNR' as display name."""
        import dask.array as da
        import numpy as np

        # Create a dummy noise signal as dask array
        noise = da.from_array(np.random.randn(1, 1000), chunks=(1, 1000))
        op = AddWithSNR(sampling_rate=44100, other=noise, snr=10.0)
        assert op.get_display_name() == "+SNR"

    def test_fade_display_name(self) -> None:
        """Test that Fade returns 'fade' as display name."""
        op = Fade(sampling_rate=44100, fade_ms=50)
        assert op.get_display_name() == "fade"


class TestTemporalDisplayNames:
    """Test display names for temporal operations."""

    def test_resampling_display_name(self) -> None:
        """Test that ReSampling returns 'rs' as display name."""
        op = ReSampling(sampling_rate=44100, target_sr=16000)
        assert op.get_display_name() == "rs"

    def test_trim_display_name(self) -> None:
        """Test that Trim returns 'trim' as display name."""
        op = Trim(sampling_rate=44100, start=0.0, end=1.0)
        assert op.get_display_name() == "trim"

    def test_fix_length_display_name(self) -> None:
        """Test that FixLength returns 'len' as display name."""
        op = FixLength(sampling_rate=44100, length=44100)
        assert op.get_display_name() == "len"

    def test_rms_trend_display_name(self) -> None:
        """Test that RmsTrend returns 'RMS' as display name."""
        op = RmsTrend(sampling_rate=44100)
        assert op.get_display_name() == "RMS"


class TestStatsDisplayNames:
    """Test display names for stats operations."""

    def test_abs_display_name(self) -> None:
        """Test that ABS returns 'abs' as display name."""
        op = ABS(sampling_rate=44100)
        assert op.get_display_name() == "abs"

    def test_power_display_name(self) -> None:
        """Test that Power returns 'pow' as display name."""
        op = Power(sampling_rate=44100, exponent=2.0)
        assert op.get_display_name() == "pow"

    def test_sum_display_name(self) -> None:
        """Test that Sum returns 'sum' as display name."""
        op = Sum(sampling_rate=44100)
        assert op.get_display_name() == "sum"

    def test_mean_display_name(self) -> None:
        """Test that Mean returns 'mean' as display name."""
        op = Mean(sampling_rate=44100)
        assert op.get_display_name() == "mean"

    def test_channel_difference_display_name(self) -> None:
        """Test that ChannelDifference returns 'diff' as display name."""
        op = ChannelDifference(sampling_rate=44100, other_channel=0)
        assert op.get_display_name() == "diff"
