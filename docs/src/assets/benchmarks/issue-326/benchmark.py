from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import dask
import numpy as np

import wandas as wd

SAMPLING_RATE = 48_000
SAMPLES = 10 * SAMPLING_RATE
REPEATED_ACCESSES = 3
TRIALS = 5
RUNS = 2


def measure(callable_) -> float:
    start = time.perf_counter()
    callable_()
    return time.perf_counter() - start


def make_spectrogram():
    time_axis = np.arange(SAMPLES, dtype=np.float64) / SAMPLING_RATE
    samples = np.sin(2 * np.pi * 1_000 * time_axis)
    return wd.ChannelFrame.from_numpy(samples, sampling_rate=SAMPLING_RATE).stft(
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        window="hann",
    )


revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
lock_sha256 = hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()
supports_cache = hasattr(wd.ChannelFrame, "cache")

with dask.config.set(scheduler="synchronous"):
    warmup = make_spectrogram()
    _ = warmup.dB
    if supports_cache:
        _ = warmup.cache().dB

    runs = []
    for run in range(RUNS):
        trials = []
        for trial in range(TRIALS):
            spectrogram = make_spectrogram()
            record = {
                "trial": trial + 1,
                "uncached_repeated_db_seconds": measure(lambda: [spectrogram.dB for _ in range(REPEATED_ACCESSES)]),
            }
            if supports_cache:
                cached_holder = []
                record["cache_creation_seconds"] = measure(lambda: cached_holder.append(spectrogram.cache()))
                cached = cached_holder[0]
                record["cached_repeated_db_seconds"] = measure(lambda: [cached.dB for _ in range(REPEATED_ACCESSES)])
            trials.append(record)
        runs.append({"run": run + 1, "trials": trials})

print(
    json.dumps(
        {
            "schema_version": 1,
            "revision": revision,
            "lock_sha256": lock_sha256,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "numpy": np.__version__,
                "dask": dask.__version__,
                "wandas": wd.__version__,
            },
            "scenario": {
                "sampling_rate": SAMPLING_RATE,
                "samples": SAMPLES,
                "channels": 1,
                "n_fft": 2048,
                "hop_length": 512,
                "win_length": 2048,
                "window": "hann",
                "scheduler": "synchronous",
                "repeated_accesses": REPEATED_ACCESSES,
                "trials_per_run": TRIALS,
                "runs": RUNS,
            },
            "supports_cache": supports_cache,
            "runs": runs,
        },
        indent=2,
    )
)
