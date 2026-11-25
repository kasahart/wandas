"""Time commonly used Frame operations so slowdowns can be noticed.

Each case builds the lazy Frame operation from an in-memory signal and reads
its result, so the time covers graph construction and materialization. The
same harness file can measure an older checkout: run it with that checkout's
environment and pass ``--repository-root`` so the report names the measured
revision. A case whose API does not exist in the measured revision is reported
as ``unavailable`` instead of failing the run, and so is a case whose optional
dependency is not installed. Sync with ``--extra io --extra psychoacoustic`` to
run every case.

Timings depend on the machine. Compare two reports only when they were taken
on the same machine with the same settings, and prefer alternating runs of the
two revisions over one long run each. ``--compare`` prints the per-case time
ratio; it does not decide whether a ratio is acceptable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
_SLICE_READS = 100
_SLICE_SECONDS = 0.1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _package_version(distribution: str) -> str | None:
    try:
        return distribution_version(distribution)
    except PackageNotFoundError:
        return None


def _git_output(repository_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _provenance(repository_root: Path) -> dict[str, Any]:
    lock_path = repository_root / "uv.lock"
    status = _git_output(repository_root, "status", "--short")
    return {
        "commit": _git_output(repository_root, "rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.is_file() else None,
        "command": [Path(sys.executable).name, *sys.argv],
    }


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": {name: _package_version(name) for name in ("wandas", "numpy", "scipy", "dask", "xarray")},
    }


def _signal(channels: int, samples: int) -> np.ndarray:
    # A fixed seed keeps the input identical across revisions and runs.
    return np.random.default_rng(0).standard_normal((channels, samples)).astype(np.float32)


def _cases(frame: Any, wav_path: Path) -> dict[str, Callable[[], Any]]:
    """Return the timed cases. Each callable must materialize its result."""
    import wandas as wd

    sampling_rate = frame.sampling_rate
    slice_length = max(1, int(_SLICE_SECONDS * sampling_rate))
    last_start = max(0, frame.n_samples - slice_length)
    slice_starts = [last_start * index // max(1, _SLICE_READS - 1) for index in range(_SLICE_READS)]

    # The cache is built during the warm-up call, so the timed calls measure
    # only the slice reads that downstream apps repeat on every cursor move.
    prepared: dict[str, Any] = {}

    def cached_slices() -> Any:
        if "frame" not in prepared:
            prepared["frame"] = frame.cache()
        cached = prepared["frame"]
        return [cached[:, start : start + slice_length].data for start in slice_starts]

    def cached_stft_columns() -> Any:
        if "stft" not in prepared:
            prepared["stft"] = frame.stft(n_fft=2048).cache()
        cached = prepared["stft"]
        columns = cached.shape[-1]
        return [cached[:, :, index : index + 1].data for index in np.linspace(0, columns - 1, _SLICE_READS, dtype=int)]

    return {
        "read_wav": lambda: wd.read_wav(str(wav_path)).data,
        "stft": lambda: frame.stft(n_fft=2048).data,
        "stft_db": lambda: frame.stft(n_fft=2048).dB,
        "welch": lambda: frame.welch(n_fft=2048).data,
        "fft": lambda: frame.fft().data,
        "noct_spectrum": lambda: frame.noct_spectrum(n=3).data,
        "low_pass_filter": lambda: frame.low_pass_filter(cutoff=1000).data,
        "a_weighting": lambda: frame.a_weighting().data,
        "sound_level_a_db": lambda: frame.sound_level(freq_weighting="A", dB=True).data,
        "rms_trend_db": lambda: frame.rms_trend(dB=True).data,
        "resampling": lambda: frame.resampling(target_sr=sampling_rate / 3).data,
        "cache_slice_reads": cached_slices,
        "cached_stft_column_reads": cached_stft_columns,
    }


def _time_case(case: Callable[[], Any], repeat: int) -> dict[str, Any]:
    try:
        case()  # Warm-up: imports, filter design caches, first allocations.
    except (AttributeError, TypeError, ImportError) as error:
        # The measured revision does not provide this API or parameter, or the
        # case needs an optional extra (noct_spectrum needs psychoacoustic).
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    seconds = []
    for _ in range(repeat):
        start = time.perf_counter()
        case()
        seconds.append(time.perf_counter() - start)
    return {
        "status": "ok",
        "median_seconds": statistics.median(seconds),
        "min_seconds": min(seconds),
        "seconds": seconds,
    }


def run(
    *,
    channels: int,
    seconds: float,
    sampling_rate: int,
    repeat: int,
    only: list[str] | None,
    repository_root: Path,
) -> dict[str, Any]:
    import wandas as wd

    samples = int(round(seconds * sampling_rate))
    signal = _signal(channels, samples)
    frame = wd.ChannelFrame.from_numpy(signal, sampling_rate=sampling_rate)
    with tempfile.TemporaryDirectory() as directory:
        wav_path = Path(directory) / "signal.wav"
        frame.to_wav(str(wav_path))
        cases = _cases(frame, wav_path)
        unknown = sorted(set(only or ()) - set(cases))
        if unknown:
            raise SystemExit(f"Unknown case(s): {', '.join(unknown)}. Known cases: {', '.join(cases)}")
        selected = [name for name in cases if only is None or name in only]
        results = {name: _time_case(cases[name], repeat) for name in selected}
    return {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "channels": channels,
            "samples": samples,
            "sampling_rate": sampling_rate,
            "dtype": "float32",
            "repeat": repeat,
        },
        "provenance": _provenance(repository_root),
        "environment": _environment(),
        "cases": results,
    }


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"{path}: expected schema_version {SCHEMA_VERSION}, got {report.get('schema_version')!r}")
    return report


def _best_median(reports: list[dict[str, Any]], name: str) -> float | None:
    """Return the smallest median over repeated reports of one revision."""
    medians = [
        report["cases"][name]["median_seconds"]
        for report in reports
        if report["cases"].get(name, {}).get("status") == "ok"
    ]
    return min(medians) if medians else None


def compare(base_paths: list[Path], candidate_paths: list[Path]) -> str:
    """Return a Markdown table of candidate/base time ratios per case."""
    base = [_load(path) for path in base_paths]
    candidate = [_load(path) for path in candidate_paths]
    inputs = {json.dumps(report["input"], sort_keys=True) for report in base + candidate}
    if len(inputs) != 1:
        raise SystemExit("Reports were taken with different input settings and cannot be compared.")
    names = list(dict.fromkeys(name for report in base + candidate for name in report["cases"]))

    def label(reports: list[dict[str, Any]]) -> str:
        commits = {str(report["provenance"].get("commit") or "unknown")[:12] for report in reports}
        return ", ".join(sorted(commits))

    lines = [
        f"Base: {label(base)} ({len(base)} run(s)); candidate: {label(candidate)} ({len(candidate)} run(s)).",
        "Each time is the smallest per-run median. Ratio = candidate / base; above 1 means slower.",
        "",
        "| Case | Base (ms) | Candidate (ms) | Ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in names:
        base_seconds = _best_median(base, name)
        candidate_seconds = _best_median(candidate, name)
        base_text = "unavailable" if base_seconds is None else f"{base_seconds * 1e3:.2f}"
        candidate_text = "unavailable" if candidate_seconds is None else f"{candidate_seconds * 1e3:.2f}"
        ratio = "-" if base_seconds is None or candidate_seconds is None else f"{candidate_seconds / base_seconds:.2f}"
        lines.append(f"| {name} | {base_text} | {candidate_text} | {ratio} |")
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Time commonly used Frame operations so slowdowns can be noticed.")
    parser.add_argument("--channels", type=_positive_int, default=2)
    parser.add_argument("--seconds", type=_positive_finite_float, default=10.0)
    parser.add_argument("--sampling-rate", type=_positive_int, default=48000)
    parser.add_argument("--repeat", type=_positive_int, default=5, help="timed runs per case after one warm-up")
    parser.add_argument("--only", nargs="+", metavar="CASE", help="run only these cases")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="checkout whose revision and lock are recorded (default: the current directory)",
    )
    parser.add_argument("--output", type=Path, help="write the JSON report here instead of stdout")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASE", "CANDIDATE"),
        help="print a ratio table instead of measuring; each side may list several reports separated by commas",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.compare:
        base, candidate = ([Path(part) for part in side.split(",")] for side in args.compare)
        sys.stdout.write(compare(base, candidate))
        return 0
    repository_root = (args.repository_root or Path.cwd()).resolve()
    report = run(
        channels=args.channels,
        seconds=args.seconds,
        sampling_rate=args.sampling_rate,
        repeat=args.repeat,
        only=args.only,
        repository_root=repository_root,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
