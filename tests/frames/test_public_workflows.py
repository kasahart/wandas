"""Observable outcomes of representative public analysis workflows."""

from pathlib import Path

import numpy as np
import soundfile as sf

import wandas as wd


def test_read_cleanup_fft_retains_known_passband_peak(tmp_path: Path) -> None:
    sampling_rate = 8_000
    time = np.arange(sampling_rate) / sampling_rate
    values = 0.4 + np.sin(2 * np.pi * 50 * time) + 0.25 * np.sin(2 * np.pi * 1_000 * time)
    path = tmp_path / "known.wav"
    sf.write(path, values, sampling_rate)

    spectrum = wd.read(path).remove_dc().low_pass_filter(cutoff=200).fft(n_fft=sampling_rate, window="boxcar")

    peak_frequency = spectrum.freqs[int(np.argmax(np.abs(spectrum.data)))]
    assert peak_frequency == 50


def test_concatenated_recording_labels_survive_fft() -> None:
    sampling_rate = 8_000
    time = np.arange(8_000) / sampling_rate
    left = wd.from_numpy(np.sin(2 * np.pi * 50 * time), sampling_rate, ch_labels=["left"])
    right = wd.from_numpy(np.sin(2 * np.pi * 100 * time), sampling_rate, ch_labels=["right"])

    spectrum = left.concat_frame(right).fft(n_fft=sampling_rate)

    assert spectrum.labels == ["left", "right"]
