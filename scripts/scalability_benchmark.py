"""Measure lazy graph and materialized WDF boundaries at representative sizes."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import dask.array as da

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan


def _rss_bytes() -> int:
    """Return Linux process high-water RSS in bytes."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _worker(channels: int, samples: int, sampling_rate: float) -> dict[str, Any]:
    total = channels * samples
    data = da.arange(total, chunks=samples, dtype=float).reshape((channels, samples))
    frame = ChannelFrame(data=data, sampling_rate=sampling_rate, label="scalability-benchmark")

    tracemalloc.start()
    lazy_started = time.perf_counter()
    processed = frame.remove_dc().normalize()
    plan = RecipePlan.from_frame(processed, input_names=("signal",))
    lazy_seconds = time.perf_counter() - lazy_started
    _, lazy_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_before_save = _rss_bytes()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "benchmark.wdf"
        save_started = time.perf_counter()
        processed.save(path, compress=None)
        save_seconds = time.perf_counter() - save_started
        file_bytes = path.stat().st_size
    rss_after_save = _rss_bytes()

    return {
        "channels": channels,
        "samples_per_channel": samples,
        "duration_seconds": samples / sampling_rate,
        "logical_data_bytes": total * 8,
        "lazy_graph_tasks": len(processed._data.dask),
        "recipe_nodes": len(plan.nodes),
        "lazy_build_seconds": lazy_seconds,
        "lazy_python_peak_bytes": lazy_peak_bytes,
        "wdf_save_seconds": save_seconds,
        "wdf_file_bytes": file_bytes,
        "wdf_save_high_water_rss_increase_bytes": max(0, rss_after_save - rss_before_save),
    }


def _run_isolated_case(script: Path, channels: int, samples: int, sampling_rate: float) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--worker",
            "--channels",
            str(channels),
            "--samples",
            str(samples),
            "--sampling-rate",
            str(sampling_rate),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--samples", type=int, nargs="+", default=[480_000, 4_800_000])
    parser.add_argument("--sampling-rate", type=float, default=48_000.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.channels <= 0 or any(samples <= 0 for samples in args.samples) or args.sampling_rate <= 0:
        parser.error("channels, samples, and sampling-rate must be positive")

    if args.worker:
        if len(args.samples) != 1:
            parser.error("worker mode accepts exactly one sample count")
        print(json.dumps(_worker(args.channels, args.samples[0], args.sampling_rate), sort_keys=True))
        return

    script = Path(__file__).resolve()
    cases = [_run_isolated_case(script, args.channels, samples, args.sampling_rate) for samples in args.samples]
    print(json.dumps({"schema": "wandas.scalability-benchmark", "version": 1, "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
