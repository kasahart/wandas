from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import wandas
from scripts.run_documentation_gate import (
    NUMERICAL_CONTRACT_TESTS,
    PREREQUISITE_MARKERS,
    GateConfigurationError,
    Stage,
    build_stages,
    detect_profile,
    run_stages,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_markers(root: Path, names: set[str]) -> None:
    for name, markers in PREREQUISITE_MARKERS.items():
        if name not in names:
            continue
        for marker in markers:
            path = root / marker
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()


def test_profile_is_standalone_without_predecessors(tmp_path: Path) -> None:
    assert detect_profile(tmp_path) == "standalone"


def test_profile_requires_the_complete_canonical_checker_cohort(tmp_path: Path) -> None:
    one_name = next(iter(PREREQUISITE_MARKERS))
    _install_markers(tmp_path, {one_name})

    with pytest.raises(GateConfigurationError, match="partial documentation-gate prerequisite cohort"):
        detect_profile(tmp_path)

    with pytest.raises(GateConfigurationError, match="requires prerequisite checkers"):
        detect_profile(tmp_path / "empty", require_final=True)


def test_profile_rejects_one_marker_from_a_multi_file_checker(tmp_path: Path) -> None:
    markers = next(markers for markers in PREREQUISITE_MARKERS.values() if len(markers) > 1)
    marker = tmp_path / markers[0]
    marker.parent.mkdir(parents=True)
    marker.touch()

    with pytest.raises(GateConfigurationError, match="partial documentation-gate prerequisite cohort"):
        detect_profile(tmp_path)


def test_profile_is_final_only_when_every_predecessor_is_present(tmp_path: Path) -> None:
    _install_markers(tmp_path, set(PREREQUISITE_MARKERS))

    assert detect_profile(tmp_path, require_final=True) == "final"


def test_final_plan_uses_canonical_checkers_and_one_ordered_site_pipeline() -> None:
    stages = build_stages(REPO_ROOT, REPO_ROOT / "docs/site", profile="final")
    names = [stage.name for stage in stages]
    commands = [" ".join(stage.command) for stage in stages]

    assert names.index("strict MkDocs build") < names.index("execute/export 00_why_wandas.py")
    assert names.index("execute/export 08_metadata_driven_dataset_search.py") < names.index(
        "finalize learning application URLs and canonical metadata"
    )
    assert names.index("finalize learning application URLs and canonical metadata") < names.index(
        "crawl completed generated site"
    )
    assert sum(name.startswith("execute/export ") for name in names) == 9
    assert any("scripts/check_public_docstrings.py" in command for command in commands)
    assert any("scripts/finalize_learning_html.py" in command for command in commands)
    assert any("scripts/check_docs_site.py" in command for command in commands)
    assert all(any(test in command for command in commands) for test in NUMERICAL_CONTRACT_TESTS)


@pytest.mark.parametrize(
    "failing_stage",
    [
        "broken source/body/generated link",
        "undefined learning example",
        "canonical public API drift",
        "FFT/Welch/IFFT numerical mismatch",
    ],
)
def test_each_consistency_failure_stops_the_gate(failing_stage: str, tmp_path: Path) -> None:
    executed: list[str] = []
    stages = (
        Stage("precondition", ("true",)),
        Stage(failing_stage, ("deliberate-mutation", failing_stage)),
        Stage("deployment", ("must-not-run",)),
    )

    def runner(
        command: tuple[str, ...],
        *,
        cwd: Path,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == tmp_path
        assert text is True
        assert check is False
        executed.append(command[0])
        return subprocess.CompletedProcess(command, 1 if command[0] == "deliberate-mutation" else 0)

    with pytest.raises(RuntimeError, match=failing_stage):
        run_stages(stages, repo_root=tmp_path, runner=runner)

    assert executed == ["true", "deliberate-mutation"]


def test_ci_and_deploy_use_the_single_gate_with_different_safety_profiles() -> None:
    ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    deploy = (REPO_ROOT / ".github/workflows/deploy-docs.yml").read_text(encoding="utf-8")
    command = "python scripts/run_documentation_gate.py"

    assert ci.count(command) == 1
    assert deploy.count(command) == 1
    assert "--require-final --site-only" in deploy
    assert "mkdocs build" not in deploy
    assert "marimo export" not in deploy
    assert deploy.index(command) < deploy.index("uses: peaceiris/actions-gh-pages")


def test_deliberately_broken_markdown_link_fails_strict_build(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "index.md").write_text("[missing](missing.md)", encoding="utf-8")
    config = tmp_path / "mkdocs.yml"
    config.write_text("site_name: fixture\ndocs_dir: docs\nnav:\n  - Home: index.md\n", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "-f", str(config)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "missing.md" in completed.stdout + completed.stderr


def test_deliberately_undefined_learning_name_fails_marimo_export(tmp_path: Path) -> None:
    app = tmp_path / "undefined_learning.py"
    app.write_text(
        """\
import marimo

__generated_with = "0.23.9"
app = marimo.App()

@app.cell
def _(undefined_learning_name):
    rendered = undefined_learning_name
    return (rendered,)

if __name__ == "__main__":
    app.run()
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            str(app),
            "-o",
            str(tmp_path / "undefined.html"),
            "--no-include-code",
            "-f",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "undefined_learning_name" in completed.stdout + completed.stderr


def test_deliberate_public_api_drift_fails_the_inventory_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests import test_init

    monkeypatch.setattr(wandas, "__all__", [*wandas.__all__, "deliberate_drift"])

    with pytest.raises(AssertionError):
        test_init.test_top_level_all_is_curated_primary_api()


def test_deliberate_fft_numerical_mismatch_fails_reference_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.processing.test_spectral_operations import TestFFTOperation
    from wandas.processing.spectral import FFT

    def wrong_fft(self: FFT, data: np.ndarray) -> np.ndarray:
        n_fft = self.n_fft or data.shape[-1]
        return np.zeros((data.shape[0], n_fft // 2 + 1), dtype=np.complex128)

    monkeypatch.setattr(FFT, "_process", wrong_fft)

    with pytest.raises(AssertionError):
        TestFFTOperation().test_fft_amplitude_matches_numpy_rfft_reference()
