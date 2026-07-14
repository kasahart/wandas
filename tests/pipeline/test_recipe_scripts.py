from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "scripts" / "recipe_v2_test_audit.py"
BENCHMARK = ROOT / "scripts" / "recipe_v2_benchmark.py"


def _run_benchmark(iterations: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BENCHMARK), "--iterations", str(iterations)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_recipe_benchmark_rejects_fewer_than_two_iterations() -> None:
    result = _run_benchmark(1)

    assert result.returncode == 2
    assert "--iterations must be at least 2" in result.stderr


def test_recipe_benchmark_accepts_two_iterations() -> None:
    result = _run_benchmark(2)

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["api"] == "v2"
    assert payload["iterations"] == 2


def test_recipe_test_audit_uses_checked_in_inventory_without_git() -> None:
    result = subprocess.run(
        [sys.executable, str(AUDIT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PATH": ""},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "audited_cases\t192\tmigrated\t28\tremoved_contract\t164"
