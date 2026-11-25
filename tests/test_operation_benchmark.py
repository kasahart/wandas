"""Smoke tests for the operation timing harness (not a timing check)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import operation_benchmark

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _small_report(only: list[str] | None = None) -> dict[str, Any]:
    return operation_benchmark.run(
        channels=1,
        seconds=0.2,
        sampling_rate=48000,
        repeat=1,
        only=only,
        repository_root=REPOSITORY_ROOT,
    )


def test_every_case_runs_on_the_current_revision() -> None:
    report = _small_report()

    assert report["schema_version"] == operation_benchmark.SCHEMA_VERSION
    cases = report["cases"]
    assert isinstance(cases, dict)
    not_ok = {name: result for name, result in cases.items() if result["status"] != "ok"}
    assert not not_ok, (
        "Every benchmark case must run on the current revision; 'unavailable' is only for older "
        f"revisions that lack the API. Fix the case or the operation: {not_ok}"
    )
    for result in cases.values():
        assert result["min_seconds"] <= result["median_seconds"]


def test_unknown_case_names_the_known_cases() -> None:
    with pytest.raises(SystemExit, match="Known cases: read_wav"):
        _small_report(only=["no_such_case"])


def test_missing_api_is_reported_as_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    original = operation_benchmark._cases

    def with_missing_api(frame: Any, wav_path: Path) -> dict[str, Any]:
        cases = original(frame, wav_path)
        cases["stft"] = lambda: frame.no_such_operation()
        return cases

    monkeypatch.setattr(operation_benchmark, "_cases", with_missing_api)
    report = _small_report(only=["stft"])

    assert report["cases"]["stft"]["status"] == "unavailable"


def test_compare_prints_ratio_per_case(tmp_path: Path) -> None:
    base = _small_report(only=["fft", "welch"])
    candidate = json.loads(json.dumps(base))
    candidate["cases"]["fft"]["median_seconds"] = base["cases"]["fft"]["median_seconds"] * 2
    del candidate["cases"]["welch"]
    base_path = tmp_path / "base.json"
    candidate_path = tmp_path / "candidate.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    table = operation_benchmark.compare([base_path], [candidate_path])

    assert "| fft |" in table and "| 2.00 |" in table
    assert "| welch |" in table and "unavailable" in table


def test_compare_refuses_reports_with_different_inputs(tmp_path: Path) -> None:
    base = _small_report(only=["fft"])
    candidate = json.loads(json.dumps(base))
    candidate["input"]["channels"] = 2
    paths = []
    for name, report in (("base.json", base), ("candidate.json", candidate)):
        path = tmp_path / name
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)

    with pytest.raises(SystemExit, match="different input settings"):
        operation_benchmark.compare([paths[0]], [paths[1]])
