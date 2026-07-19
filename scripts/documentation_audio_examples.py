"""Shared numeric contracts for WAV-backed documentation examples."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

import wandas as wd

COMPARISON_SAMPLING_RATE = 44_100
COMPARISON_DURATION = 5.0
COMPARISON_SEED = 42
COMPARISON_PEAK_FS = 0.95
COMPARISON_LABELS = ("Normal", "Mild Abnormal", "Severe Abnormal")
COMPARISON_WELCH_YLIM = (-100.0, 10.0)
COMPARISON_NOCT_YLIM = (-60.0, 10.0)
COMPARISON_NOCT_FMAX = 16_000
DB_REFERENCE_FS = 1.0


def comparison_time() -> np.ndarray:
    """Return exact sample instants for the comparison's discrete-time contract."""
    sample_count = int(COMPARISON_SAMPLING_RATE * COMPARISON_DURATION)
    return np.arange(sample_count) / COMPARISON_SAMPLING_RATE


def comparison_signals() -> np.ndarray:
    """Return the three deterministic teaching signals scaled together to 0.95 FS."""
    rng = np.random.RandomState(COMPARISON_SEED)
    time = comparison_time()
    signals = np.stack(
        [
            np.sin(2 * np.pi * 100 * time) + 0.3 * np.sin(2 * np.pi * 200 * time) + noise_scale * rng.randn(time.size)
            for noise_scale in (0.1, 0.5, 2.0)
        ]
    )
    impulse_positions = rng.choice(time.size, size=5, replace=False)
    signals[2, impulse_positions] += 5.0
    return signals * (COMPARISON_PEAK_FS / np.max(np.abs(signals)))


def write_comparison_pcm16(directory: Path) -> tuple[Path, ...]:
    """Write the comparison signals as explicit mono PCM16 WAV files."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, (label, samples) in enumerate(zip(COMPARISON_LABELS, comparison_signals(), strict=True)):
        path = directory / f"signal_{index}_{label.lower().replace(' ', '_')}.wav"
        sf.write(path, samples, COMPARISON_SAMPLING_RATE, subtype="PCM_16")
        paths.append(path)
    return tuple(paths)


def read_comparison_pcm16(paths: Sequence[Path]) -> tuple[wd.ChannelFrame, ...]:
    """Read explicitly supplied PCM16 fixtures through the canonical public entrypoint."""
    paths_and_labels = zip(paths, COMPARISON_LABELS, strict=True)
    return tuple(wd.read(path, ch_labels=[label]) for path, label in paths_and_labels)
