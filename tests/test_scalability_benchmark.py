"""Smoke test for the reproducible scalability benchmark artifact."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import dask.array as da
import pytest
from dask.callbacks import Callback
from dask.highlevelgraph import HighLevelGraph

from scripts import scalability_benchmark
from wandas.frames.channel import ChannelFrame

BENCHMARK_SCRIPT = Path(scalability_benchmark.__file__).resolve()


def _run_benchmark(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), *args],
        check=False,
        capture_output=True,
        cwd=cwd,
        text=True,
    )


def test_scalability_benchmark_small_case_reports_materialization_boundary() -> None:
    completed = _run_benchmark("--channels", "1", "--samples", "64")
    completed.check_returncode()

    report = json.loads(completed.stdout)
    assert report["schema"] == "wandas.scalability-benchmark"
    assert report["version"] == 1
    assert len(report["cases"]) == 1
    case = report["cases"][0]
    assert case["channels"] == 1
    assert case["samples_per_channel"] == 64
    assert case["recipe_nodes"] == 2
    assert case["lazy_graph_tasks"] > 0
    assert case["wdf_file_bytes"] > case["logical_data_bytes"]
    assert case["isolated_process_peak_rss_bytes"] > 0
    assert "wdf_save_high_water_rss_increase_bytes" not in case
    assert all(not isinstance(value, float) or math.isfinite(value) for value in case.values())


def test_scalability_benchmark_runs_outside_repository_root(tmp_path: Path) -> None:
    completed = _run_benchmark("--channels", "1", "--samples", "64", cwd=tmp_path)

    completed.check_returncode()
    assert json.loads(completed.stdout)["schema"] == "wandas.scalability-benchmark"


@pytest.mark.parametrize(
    ("platform", "raw_peak_rss", "expected_bytes"),
    [("linux", 8, 8 * 1024), ("freebsd14", 8, 8 * 1024), ("darwin", 8, 8)],
)
def test_unix_peak_rss_normalizes_platform_units(
    platform: str,
    raw_peak_rss: int,
    expected_bytes: int,
) -> None:
    assert scalability_benchmark._normalize_unix_peak_rss_bytes(raw_peak_rss, platform) == expected_bytes


def test_peak_rss_uses_windows_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scalability_benchmark, "_windows_peak_rss_bytes", lambda: 4096)

    assert scalability_benchmark._peak_rss_bytes("win32") == 4096


def test_dask_graph_task_count_counts_high_level_graph_task_keys() -> None:
    collection = da.arange(8, chunks=2)
    graph = collection.__dask_graph__()

    assert isinstance(graph, HighLevelGraph)
    assert scalability_benchmark._dask_graph_task_count(collection) == 4


def test_public_frame_xarray_graph_count_does_not_compute_samples() -> None:
    class RejectComputation(Callback):
        def _start(self, dsk: object) -> None:
            del dsk
            raise AssertionError("public lazy graph inspection must not compute samples")

    frame = ChannelFrame(
        data=da.arange(8, chunks=4, dtype=float).reshape((1, 8)),
        sampling_rate=8.0,
    )
    processed = frame.remove_dc().normalize()

    with RejectComputation():
        public_data = processed.xr.data
        task_count = scalability_benchmark._dask_graph_task_count(public_data)

    assert isinstance(public_data, da.Array)
    assert task_count > 0


def test_dask_graph_task_count_prefers_graph_nkeys() -> None:
    class Graph:
        def nkeys(self) -> int:
            return 7

        def keys(self) -> None:
            raise AssertionError("keys() must not be used when nkeys() is available")

    class Collection:
        def __dask_graph__(self) -> Graph:
            return Graph()

    assert scalability_benchmark._dask_graph_task_count(Collection()) == 7


def test_dask_graph_task_count_falls_back_to_task_mapping_keys() -> None:
    class Collection:
        def __dask_graph__(self) -> dict[tuple[str, int], object]:
            return {("task", index): object() for index in range(3)}

    assert scalability_benchmark._dask_graph_task_count(Collection()) == 3


@pytest.mark.parametrize("sampling_rate", ["nan", "inf", "-inf", "0", "-1"])
def test_scalability_benchmark_rejects_non_finite_or_non_positive_sampling_rate(sampling_rate: str) -> None:
    completed = _run_benchmark("--samples", "64", f"--sampling-rate={sampling_rate}")

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "finite positive number" in completed.stderr


def test_scalability_benchmark_rejects_non_finite_derived_duration() -> None:
    completed = _run_benchmark("--samples", "64", "--sampling-rate=1e-320")

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "sample count and sampling rate must produce a finite duration" in completed.stderr
    assert "Traceback" not in completed.stderr


@pytest.mark.parametrize("samples", ["0", "-1"])
def test_scalability_benchmark_rejects_non_positive_sample_count(samples: str) -> None:
    completed = _run_benchmark("--samples", samples)

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "positive integer" in completed.stderr
