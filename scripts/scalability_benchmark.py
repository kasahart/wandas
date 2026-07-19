"""Measure lazy graph and materialized WDF boundaries at representative sizes."""

from __future__ import annotations

import argparse
import json
import math
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


def _normalize_unix_peak_rss_bytes(peak_rss: int, platform: str) -> int:
    """Normalize platform-dependent ``ru_maxrss`` units to bytes."""
    # macOS reports bytes; Linux and the other supported Unix runners report KiB.
    return peak_rss if platform == "darwin" else peak_rss * 1024


def _unix_peak_rss_bytes(platform: str) -> int:
    """Return this Unix process's absolute peak RSS in bytes."""
    import resource

    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return _normalize_unix_peak_rss_bytes(peak_rss, platform)


def _windows_peak_rss_bytes() -> int:
    """Return this Windows process's absolute peak working set in bytes."""
    if sys.platform != "win32":
        raise RuntimeError("Windows peak RSS measurement requires Windows")

    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def _peak_rss_bytes(platform: str | None = None) -> int:
    """Return this isolated worker's absolute process peak RSS in bytes."""
    current_platform = sys.platform if platform is None else platform
    if current_platform == "win32":
        return _windows_peak_rss_bytes()
    return _unix_peak_rss_bytes(current_platform)


def _positive_int(value: str) -> int:
    """Parse one positive integer CLI value."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_finite_float(value: str) -> float:
    """Parse one finite, positive floating-point CLI value."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _finite_duration_seconds(samples: int, sampling_rate: float) -> float:
    """Return a representable duration for one benchmark case."""
    try:
        duration = samples / sampling_rate
    except OverflowError as error:
        raise ValueError("sample count and sampling rate must produce a finite duration") from error
    if not math.isfinite(duration):
        raise ValueError("sample count and sampling rate must produce a finite duration")
    return duration


def _dask_graph_task_count(collection: Any) -> int:
    """Count concrete task keys through Dask's public collection protocol."""
    graph = collection.__dask_graph__()
    nkeys = getattr(graph, "nkeys", None)
    if callable(nkeys):
        return int(nkeys())

    keys = getattr(graph, "keys", None)
    if not callable(keys):
        raise TypeError("Dask collection graph must expose nkeys() or keys()")
    return sum(1 for _key in keys())


def _chunked_source_frame(
    channels: int,
    samples: int,
    chunk_samples: int,
    sampling_rate: float,
) -> ChannelFrame:
    """Build the benchmark-only WDF source without constructor rechunking."""
    total = channels * samples
    requested_time_chunk = min(chunk_samples, samples)
    data = (
        da.arange(total, chunks=chunk_samples, dtype=float)
        .reshape((channels, samples))
        .rechunk((1, requested_time_chunk))
    )
    frame = ChannelFrame(data=data, sampling_rate=sampling_rate, label="scalability-benchmark")

    # ChannelFrame intentionally normalizes normal public inputs to one complete
    # time chunk. This internal benchmark fixture restores the synthetic source's
    # chunks so the WDF writer boundary can be measured without changing that API.
    frame._xr = frame._xr.copy(deep=False, data=data)
    return frame


def _source_time_chunks(frame: ChannelFrame, requested_chunk_samples: int) -> tuple[int, ...]:
    """Return and validate the actual source chunks immediately before WDF save."""
    data = frame.xr.data
    if not isinstance(data, da.Array) or data.chunks is None:
        raise TypeError("benchmark source Frame must contain a chunked Dask array")
    if len(data.chunks) != 2:
        raise ValueError("benchmark source Frame must have channel and time dimensions")

    time_chunks = tuple(int(chunk) for chunk in data.chunks[1])
    effective_limit = min(requested_chunk_samples, frame.n_samples)
    if not time_chunks or max(time_chunks) > effective_limit:
        raise ValueError(
            "benchmark source time chunks exceed the requested chunk-samples limit: "
            f"actual={time_chunks}, limit={effective_limit}"
        )
    return time_chunks


def _worker(channels: int, samples: int, chunk_samples: int, sampling_rate: float) -> dict[str, Any]:
    total = channels * samples
    frame = _chunked_source_frame(channels, samples, chunk_samples, sampling_rate)

    tracemalloc.start()
    lazy_started = time.perf_counter()
    processed = frame.remove_dc().normalize()
    plan = RecipePlan.from_frame(processed, input_names=("signal",))
    lazy_seconds = time.perf_counter() - lazy_started
    _, lazy_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "benchmark.wdf"
        time_chunks = _source_time_chunks(frame, chunk_samples)
        save_started = time.perf_counter()
        frame.save(path, compress=None)
        save_seconds = time.perf_counter() - save_started
        file_bytes = path.stat().st_size

    return {
        "channels": channels,
        "samples_per_channel": samples,
        "time_chunk_samples": max(time_chunks),
        "chunks_per_channel": len(time_chunks),
        "duration_seconds": _finite_duration_seconds(samples, sampling_rate),
        "logical_data_bytes": total * 8,
        "lazy_graph_tasks": _dask_graph_task_count(processed.xr.data),
        "recipe_nodes": len(plan.nodes),
        "lazy_build_seconds": lazy_seconds,
        "lazy_python_peak_bytes": lazy_peak_bytes,
        "wdf_save_seconds": save_seconds,
        "wdf_file_bytes": file_bytes,
        "isolated_process_peak_rss_bytes": _peak_rss_bytes(),
    }


def _run_isolated_case(
    script: Path,
    channels: int,
    samples: int,
    chunk_samples: int,
    sampling_rate: float,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--worker",
            "--channels",
            str(channels),
            "--samples",
            str(samples),
            "--chunk-samples",
            str(chunk_samples),
            "--sampling-rate",
            str(sampling_rate),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        sys.stderr.write(completed.stderr)
        if completed.stdout:
            sys.stderr.write(completed.stdout)
        completed.check_returncode()
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=_positive_int, default=2)
    parser.add_argument("--samples", type=_positive_int, nargs="+", default=[480_000, 4_800_000])
    parser.add_argument("--chunk-samples", type=_positive_int, nargs="+", default=[48_000, 480_000])
    parser.add_argument("--sampling-rate", type=_positive_finite_float, default=48_000.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        for samples in args.samples:
            _finite_duration_seconds(samples, args.sampling_rate)
    except ValueError as error:
        parser.error(str(error))

    if args.worker:
        if len(args.samples) != 1 or len(args.chunk_samples) != 1:
            parser.error("worker mode accepts exactly one sample count and one chunk size")
        print(
            json.dumps(
                _worker(args.channels, args.samples[0], args.chunk_samples[0], args.sampling_rate),
                allow_nan=False,
                sort_keys=True,
            )
        )
        return

    script = Path(__file__).resolve()
    cases = [
        _run_isolated_case(script, args.channels, samples, chunk_samples, args.sampling_rate)
        for samples in args.samples
        for chunk_samples in args.chunk_samples
    ]
    report = {"schema": "wandas.scalability-benchmark", "version": 2, "cases": cases}
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
