# tests/utils/test_generate_sample.py

import numpy as np

from wandas.frames.channel import ChannelFrame
from wandas.utils.generate_sample import generate_sin


class TestGenerateSin:
    """Test suite for generate_sin — Pillar 4: theoretical value verification."""

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

        computed_data = signal.compute()
        expected_n_samples = int(sampling_rate * duration)
        assert computed_data.shape[1] == expected_n_samples

    def test_single_frequency_fft_peak_at_correct_bin(self) -> None:
        """FFT peak of generated 1 kHz sine must appear at the 1 kHz bin (within +/-1 bin)."""
        freq = 1000.0
        sampling_rate = 16000
        duration = 1.0
        signal = generate_sin(freqs=freq, sampling_rate=sampling_rate, duration=duration)

        data = signal.compute()[0]  # Single channel
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

        computed_data = signal.compute()
        expected_n_samples = int(sampling_rate * duration)
        assert computed_data.shape[1] == expected_n_samples

    def test_multiple_frequencies_each_channel_peak(self) -> None:
        """Each channel's FFT peak must correspond to its assigned frequency."""
        freqs = [500.0, 1000.0, 2000.0]
        sampling_rate = 16000
        duration = 1.0
        signal = generate_sin(freqs=freqs, sampling_rate=sampling_rate, duration=duration)

        data = signal.compute()
        n_samples = data.shape[1]
        freq_resolution = sampling_rate / n_samples  # 1 Hz for 1s at 16 kHz

        for ch_idx, expected_freq in enumerate(freqs):
            spectrum = np.abs(np.fft.rfft(data[ch_idx]))
            freqs_axis = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)
            peak_freq = freqs_axis[np.argmax(spectrum)]

            assert abs(peak_freq - expected_freq) <= freq_resolution, (
                f"Channel {ch_idx}: FFT peak at {peak_freq} Hz, expected {expected_freq} Hz"
            )
