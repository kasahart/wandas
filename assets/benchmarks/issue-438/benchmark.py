"""Isolated resident-memory probe for Issue #438 cache dtype reduction."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _current_rss_bytes() -> int:
    """Return current Linux resident bytes from procfs."""
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _peak_rss_bytes() -> int:
    """Normalize ru_maxrss to bytes on Linux and macOS."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(mode: str, channels: int, samples: int) -> dict[str, Any]:
    """Build and retain one cache scenario, then report process memory."""
    project_root = Path.cwd().resolve()
    sys.path.insert(0, str(project_root))

    import dask
    import dask.array as da
    import numpy as np

    from wandas.frames.channel import ChannelFrame

    source = da.ones((channels, samples), chunks=(1, -1), dtype=np.float64)
    frame = ChannelFrame(source, sampling_rate=48_000.0)
    started = time.perf_counter_ns()
    cached = (frame.astype("float32") if mode == "compact" else frame).cache()
    materialization_ns = time.perf_counter_ns() - started
    del frame, source
    gc.collect()

    raw_bytes = int(cached._data.size * cached._data.dtype.itemsize)
    lock_path = project_root / "uv.lock"
    return {
        "schema_version": 1,
        "issue": 438,
        "mode": mode,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
        ).strip(),
        "worktree_clean": subprocess.call(
            ["git", "diff", "--quiet"],
            cwd=project_root,
        )
        == 0,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "dask": dask.__version__,
            "uv_lock_sha256": _sha256(lock_path),
        },
        "input": {
            "channels": channels,
            "samples": samples,
            "dtype": "float64",
        },
        "cache": {
            "dtype": cached._data.dtype.name,
            "raw_bytes": raw_bytes,
            "current_rss_bytes": _current_rss_bytes(),
            "peak_rss_bytes": _peak_rss_bytes(),
            "materialization_ns": materialization_ns,
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse one isolated measurement scenario."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("control", "compact"), required=True)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8_000_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(measure(args.mode, args.channels, args.samples), sort_keys=True))
