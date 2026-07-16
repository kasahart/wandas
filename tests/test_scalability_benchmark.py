"""Smoke test for the reproducible scalability benchmark artifact."""

from __future__ import annotations

import json
import subprocess
import sys


def test_scalability_benchmark_small_case_reports_materialization_boundary() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/scalability_benchmark.py", "--channels", "1", "--samples", "64"],
        check=True,
        capture_output=True,
        text=True,
    )

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
