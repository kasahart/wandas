# tests/utils/test_generate_sample.py

import dask.array as da
import numpy as np
import pytest

from tests.frame_helpers import channel_first_values
from wandas.frames.channel import ChannelFrame
from wandas.utils.generate_sample import generate_sin, generate_sin_lazy


class TestGenerateSin:
    """Test suite for generate_sin — Pillar 4: theoretical value verification."""

    @pytest.mark.parametrize("generator", [generate_sin, generate_sin_lazy])
    def test_defaults_create_one_lazy_channel(self, generator) -> None:
        signal = generator()

        assert isinstance(signal, ChannelFrame)
        assert signal.sampling_rate == 16000
        assert signal.duration == 1.0
        assert signal.n_channels == 1
        assert signal.n_samples == 16000
        assert isinstance(signal._data, da.Array)

    @pytest.mark.parametrize(
        "freq",
        [1000, 1000.0, np.int32(1000), np.int64(1000), np.float32(1000), np.float64(1000)],
    )
    @pytest.mark.parametrize("generator", [generate_sin, generate_sin_lazy])
    def test_real_scalar_types_are_normalized_consistently(self, generator, freq) -> None:
        signal = generator(freqs=freq, sampling_rate=8000, duration=0.01)

        assert signal.n_channels == 1
        np.testing.assert_allclose(
            channel_first_values(signal)[0],
            np.sin(2 * np.pi * 1000.0 * np.arange(80) / 8000),
            atol=1e-12,
        )

    @pytest.mark.parametrize("generator", [generate_sin, generate_sin_lazy])
    def test_mixed_python_and_numpy_real_list_is_normalized_without_mutation(self, generator) -> None:
        freqs = [250, np.float32(500), np.int64(1000)]
        original = freqs.copy()

        signal = generator(freqs=freqs, sampling_rate=8000, duration=0.01)

        assert freqs == original
        assert signal.n_channels == 3
        for index, expected_freq in enumerate([250.0, 500.0, 1000.0]):
            np.testing.assert_allclose(
                channel_first_values(signal)[index],
                np.sin(2 * np.pi * expected_freq * np.arange(80) / 8000),
                atol=1e-6,
            )

    @pytest.mark.parametrize("generator", [generate_sin, generate_sin_lazy])
    @pytest.mark.parametrize("freqs", ["1000", (1000,), True, np.bool_(True), 1000 + 0j])
    def test_invalid_collection_or_scalar_type_has_actionable_error(self, generator, freqs) -> None:
        with pytest.raises(TypeError) as exc_info:
            generator(freqs=freqs)

        message = str(exc_info.value)
        assert "Invalid freqs" in message
        assert "Expected: one real frequency or a non-empty list" in message
        assert "Pass a numeric scalar in Hz" in message

    @pytest.mark.parametrize("generator", [generate_sin, generate_sin_lazy])
    def test_empty_frequency_list_has_actionable_error(self, generator) -> None:
        with pytest.raises(ValueError) as exc_info:
            generator(freqs=[])

        message = str(exc_info.value)
        assert "Invalid freqs" in message
        assert "Got: an empty list" in message
        assert "Pass one frequency in Hz for each output channel." in message

    @pytest.mark.parametrize("generator", [generate_sin, generate_sin_lazy])
    @pytest.mark.parametrize("invalid", ["440", None, True, 1 + 2j])
    def test_invalid_list_element_has_indexed_actionable_error(self, generator, invalid) -> None:
        with pytest.raises(TypeError) as exc_info:
            generator(freqs=[440.0, invalid])

        message = str(exc_info.value)
        assert "Invalid frequency" in message
        assert "freqs[1]" in message
        assert "Expected: a finite real number greater than 0 Hz" in message
        assert "Replace the invalid element" in message

    @pytest.mark.parametrize("generator", [generate_sin, generate_sin_lazy])
    @pytest.mark.parametrize("invalid", [0, -440, np.nan, np.inf, -np.inf])
    def test_non_positive_or_non_finite_frequency_has_actionable_error(self, generator, invalid) -> None:
        with pytest.raises(ValueError) as exc_info:
            generator(freqs=[invalid])

        message = str(exc_info.value)
        assert "Invalid frequency" in message
        assert "freqs[0]" in message
        assert "positive finite frequency" in message

    def test_single_frequency_metadata(self) -> None:
        """Verify frame metadata for a single-frequency signal."""
        freq = 1000.0
        sampling_rate = 16000
        duration = 1.0
        signal = generate_sin(freqs=freq, sampling_rate=sampling_rate, duration=duration, label="Test Signal")

        assert isinstance(signal, ChannelFrame)
        assert signal.label == "Test Signal"
        assert len(signal) == 1
        assert signal.channels[0].label == "Channel 1"

        computed_data = channel_first_values(signal)
        expected_n_samples = int(sampling_rate * duration)
        assert computed_data.shape[1] == expected_n_samples

    def test_single_frequency_fft_peak_at_correct_bin(self) -> None:
        """FFT peak of generated 1 kHz sine must appear at the 1 kHz bin (within +/-1 bin)."""
        freq = 1000.0
        sampling_rate = 16000
        duration = 1.0
        signal = generate_sin(freqs=freq, sampling_rate=sampling_rate, duration=duration)

        data = channel_first_values(signal)[0]  # Single channel
        n_samples = len(data)
        spectrum = np.abs(np.fft.rfft(data))
        freqs_axis = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)
        peak_freq = freqs_axis[np.argmax(spectrum)]

        # Peak should be within 1 FFT bin of expected frequency
        freq_resolution = sampling_rate / n_samples
        assert abs(peak_freq - freq) <= freq_resolution, (
            f"FFT peak at {peak_freq} Hz, expected {freq} Hz (resolution: {freq_resolution} Hz)"
        )

    def test_multiple_frequencies_channel_count(self) -> None:
        """Verify channel count and labels for multi-frequency signal."""
        freqs = [500.0, 800.0, 1000.0]
        sampling_rate = 16000
        duration = 1.0
        signal = generate_sin(freqs=freqs, sampling_rate=sampling_rate, duration=duration, label="Test Signal")

        assert isinstance(signal, ChannelFrame)
        assert signal.label == "Test Signal"
        assert len(signal) == len(freqs)

        for idx, channel in enumerate(signal.channels):
            assert channel.label == f"Channel {idx + 1}"

        computed_data = channel_first_values(signal)
        expected_n_samples = int(sampling_rate * duration)
        assert computed_data.shape[1] == expected_n_samples

    def test_multiple_frequencies_each_channel_peak(self) -> None:
        """Each channel's FFT peak must correspond to its assigned frequency."""
        freqs = [500.0, 1000.0, 2000.0]
        sampling_rate = 16000
        duration = 1.0
        signal = generate_sin(freqs=freqs, sampling_rate=sampling_rate, duration=duration)

        data = channel_first_values(signal)
        n_samples = data.shape[1]
        freq_resolution = sampling_rate / n_samples  # 1 Hz for 1s at 16 kHz

        for ch_idx, expected_freq in enumerate(freqs):
            spectrum = np.abs(np.fft.rfft(data[ch_idx]))
            freqs_axis = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)
            peak_freq = freqs_axis[np.argmax(spectrum)]

            assert abs(peak_freq - expected_freq) <= freq_resolution, (
                f"Channel {ch_idx}: FFT peak at {peak_freq} Hz, expected {expected_freq} Hz"
            )
