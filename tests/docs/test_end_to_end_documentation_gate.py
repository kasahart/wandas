from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import wandas
from scripts.run_documentation_gate import (
    FINALIZATION_SENTINEL,
    LEARNING_APP_NAMES,
    LEARNING_FIXTURES,
    NUMERICAL_CONTRACT_TESTS,
    PREREQUISITE_MARKERS,
    SOURCE_CONTRACT_TESTS,
    GateConfigurationError,
    Stage,
    build_stages,
    ci_requires_final,
    detect_profile,
    learning_apps,
    mkdocs_site_url,
    prepare_learning_workspace,
    remove_stale_learning_exports,
    require_worktree_finalization_sentinel,
    run_stages,
    validate_mode,
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


def test_profile_accepts_complete_predecessors_but_requires_all_for_deploy(
    tmp_path: Path,
) -> None:
    one_name = next(iter(PREREQUISITE_MARKERS))
    _install_markers(tmp_path, {one_name})

    assert detect_profile(tmp_path) == "integration"

    with pytest.raises(GateConfigurationError, match="requires prerequisite checkers"):
        detect_profile(tmp_path, require_final=True)


def test_profile_matrix_classifies_every_complete_checker_subset(
    tmp_path: Path,
) -> None:
    names = tuple(PREREQUISITE_MARKERS)
    for mask in range(1 << len(names)):
        root = tmp_path / f"subset-{mask}"
        selected = {name for index, name in enumerate(names) if mask & (1 << index)}
        _install_markers(root, selected)

        expected = "standalone" if not selected else "final" if len(selected) == len(names) else "integration"
        assert detect_profile(root) == expected

        validate_mode(expected, site_only=False)
        if expected == "final":
            validate_mode(expected, site_only=True)
        else:
            with pytest.raises(
                GateConfigurationError,
                match=f"requires final profile; detected {expected}",
            ):
                validate_mode(expected, site_only=True)

        if expected == "final":
            assert detect_profile(root, require_final=True) == "final"
        else:
            with pytest.raises(
                GateConfigurationError,
                match="requires prerequisite checkers",
            ):
                detect_profile(root, require_final=True)


def test_profile_rejects_one_marker_from_a_multi_file_checker(tmp_path: Path) -> None:
    markers = next(markers for markers in PREREQUISITE_MARKERS.values() if len(markers) > 1)
    marker = tmp_path / markers[0]
    marker.parent.mkdir(parents=True)
    marker.touch()

    with pytest.raises(GateConfigurationError, match="partially installed documentation-gate checker"):
        detect_profile(tmp_path)


def test_profile_rejects_partial_checker_among_complete_predecessors(
    tmp_path: Path,
) -> None:
    multi_name, multi_markers = next(
        (name, markers) for name, markers in PREREQUISITE_MARKERS.items() if len(markers) > 1
    )
    complete_name = next(name for name in PREREQUISITE_MARKERS if name != multi_name)
    _install_markers(tmp_path, {complete_name})
    marker = tmp_path / multi_markers[0]
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()

    with pytest.raises(
        GateConfigurationError,
        match=f"{multi_name}=partial",
    ):
        detect_profile(tmp_path)


def test_profile_is_final_only_when_every_predecessor_is_present(tmp_path: Path) -> None:
    _install_markers(tmp_path, set(PREREQUISITE_MARKERS))

    assert detect_profile(tmp_path, require_final=True) == "final"


@pytest.mark.parametrize(
    ("state", "event_name", "base_sha", "paths", "expected"),
    [
        ("pre-final PR", "pull_request", "pre-final-base", set(), False),
        ("closing PR", "pull_request", "closing-base", {("HEAD", FINALIZATION_SENTINEL)}, False),
        (
            "post-final checker deletion PR",
            "pull_request",
            "finalized-base",
            {
                ("finalized-base", FINALIZATION_SENTINEL),
                ("HEAD", FINALIZATION_SENTINEL),
            },
            True,
        ),
        ("finalized main push", "push", None, {("HEAD", FINALIZATION_SENTINEL)}, True),
    ],
)
def test_ci_finalization_state_transition_matrix(
    tmp_path: Path,
    state: str,
    event_name: str,
    base_sha: str | None,
    paths: set[tuple[str, Path]],
    expected: bool,
) -> None:
    def path_exists(repo_root: Path, ref: str, path: Path) -> bool:
        assert repo_root == tmp_path, state
        return (ref, path) in paths

    assert (
        ci_requires_final(
            tmp_path,
            event_name=event_name,
            base_sha=base_sha,
            path_exists=path_exists,
        )
        is expected
    )


def test_finalized_main_push_rejects_removed_sentinel(tmp_path: Path) -> None:
    with pytest.raises(GateConfigurationError, match="finalized main push is missing"):
        ci_requires_final(
            tmp_path,
            event_name="push",
            base_sha=None,
            path_exists=lambda _root, _ref, _path: False,
        )


def test_finalized_pull_request_rejects_removed_sentinel(tmp_path: Path) -> None:
    with pytest.raises(GateConfigurationError, match="finalized pull request is missing"):
        ci_requires_final(
            tmp_path,
            event_name="pull_request",
            base_sha="finalized-base",
            path_exists=lambda _root, ref, _path: ref == "finalized-base",
        )


def test_final_deployment_rejects_removed_worktree_sentinel(tmp_path: Path) -> None:
    with pytest.raises(GateConfigurationError, match="final documentation gate is missing"):
        require_worktree_finalization_sentinel(tmp_path)

    sentinel = tmp_path / FINALIZATION_SENTINEL
    sentinel.parent.mkdir(parents=True)
    sentinel.touch()
    require_worktree_finalization_sentinel(tmp_path)


def test_ci_policy_rejects_missing_or_unexpected_event_context(tmp_path: Path) -> None:
    with pytest.raises(GateConfigurationError, match="requires WANDAS_DOCS_BASE_SHA"):
        ci_requires_final(
            tmp_path,
            event_name="pull_request",
            base_sha=None,
        )
    with pytest.raises(GateConfigurationError, match="does not support event"):
        ci_requires_final(
            tmp_path,
            event_name="workflow_dispatch",
            base_sha=None,
        )


def test_learning_inventory_rejects_additional_numbered_apps(tmp_path: Path) -> None:
    learning_root = tmp_path / "learning-path"
    learning_root.mkdir()
    for name in LEARNING_APP_NAMES:
        (learning_root / name).touch()
    (learning_root / "09_new_topic.py").touch()

    with pytest.raises(GateConfigurationError, match="09_new_topic.py"):
        learning_apps(tmp_path)


def test_learning_workspace_uses_checked_in_fixtures_outside_repository(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    source_root = repo_root / "learning-path"
    source_root.mkdir(parents=True)
    for name in LEARNING_FIXTURES:
        (source_root / name).write_bytes(f"fixture:{name}".encode())
    workspace = tmp_path / "isolated"

    prepare_learning_workspace(repo_root, workspace)

    assert not workspace.is_relative_to(repo_root)
    assert {path.name: path.read_bytes() for path in workspace.iterdir()} == {
        name: f"fixture:{name}".encode() for name in LEARNING_FIXTURES
    }


def test_learning_workspace_rejects_repository_paths_and_stale_content(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    source_root = repo_root / "learning-path"
    source_root.mkdir(parents=True)
    for name in LEARNING_FIXTURES:
        (source_root / name).write_bytes(b"fixture")

    with pytest.raises(GateConfigurationError, match="outside the repository"):
        prepare_learning_workspace(repo_root, repo_root / "generated")

    stale_workspace = tmp_path / "stale"
    stale_workspace.mkdir()
    (stale_workspace / "output").mkdir()
    with pytest.raises(GateConfigurationError, match="must start empty"):
        prepare_learning_workspace(repo_root, stale_workspace)


@pytest.mark.parametrize("relative_site_dir", ["docs", "docs/src", "docs/src/generated"])
def test_site_output_rejects_source_overlap_before_removing_stale_exports(
    tmp_path: Path,
    relative_site_dir: str,
) -> None:
    repo_root = tmp_path / "repo"
    site_dir = repo_root / relative_site_dir
    stale_export = site_dir / "learning-path/stale.html"
    stale_export.parent.mkdir(parents=True)
    stale_export.touch()

    with pytest.raises(GateConfigurationError, match="must be disjoint"):
        remove_stale_learning_exports(repo_root, site_dir)

    assert stale_export.is_file()


def test_site_output_removes_only_stale_exports_from_safe_destination(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    site_dir = repo_root / "docs/site"
    stale_exports = site_dir / "learning-path"
    stale_exports.mkdir(parents=True)
    (stale_exports / "stale.html").touch()

    assert remove_stale_learning_exports(repo_root, site_dir) == site_dir.resolve()
    assert not stale_exports.exists()


def test_final_plan_uses_canonical_checkers_and_one_ordered_site_pipeline(
    tmp_path: Path,
) -> None:
    learning_workspace = tmp_path / "isolated-learning"
    stages = build_stages(
        REPO_ROOT,
        tmp_path / "site",
        profile="final",
        learning_workspace=learning_workspace,
    )
    names = [stage.name for stage in stages]
    commands = [" ".join(stage.command) for stage in stages]
    export_stages = [stage for stage in stages if stage.name.startswith("execute/export ")]

    assert names.index("strict MkDocs build") < names.index("execute/export 00_why_wandas.py")
    assert names.index("execute/export 08_metadata_driven_dataset_search.py") < names.index(
        "finalize learning application URLs and canonical metadata"
    )
    assert names.index("finalize learning application URLs and canonical metadata") < names.index(
        "crawl completed generated site"
    )
    assert sum(name.startswith("execute/export ") for name in names) == 9
    assert all(stage.cwd == learning_workspace for stage in export_stages)
    assert all(str(REPO_ROOT / "learning-path") in " ".join(stage.command) for stage in export_stages)
    assert any("scripts/check_public_docstrings.py" in command for command in commands)
    assert any("scripts/finalize_learning_html.py" in command for command in commands)
    assert any("scripts/check_docs_site.py" in command for command in commands)
    assert all(any(test in command for command in commands) for test in SOURCE_CONTRACT_TESTS)


def test_source_contract_inventory_includes_dependency_metadata() -> None:
    assert "tests/test_optional_dependencies.py" in SOURCE_CONTRACT_TESTS
    assert set(NUMERICAL_CONTRACT_TESTS).issubset(SOURCE_CONTRACT_TESTS)


def test_final_site_only_plan_retains_every_final_site_checker(
    tmp_path: Path,
) -> None:
    stages = build_stages(
        REPO_ROOT,
        tmp_path / "site",
        profile="final",
        learning_workspace=tmp_path / "isolated-learning",
        site_only=True,
    )
    commands = [" ".join(stage.command) for stage in stages]

    assert not any("-m pytest" in command for command in commands)
    assert any("scripts/check_public_docstrings.py" in command for command in commands)
    assert any("scripts/finalize_learning_html.py" in command for command in commands)
    assert any("scripts/check_docs_site.py" in command for command in commands)


def test_final_plan_uses_mkdocs_site_url_as_the_single_canonical_origin(
    tmp_path: Path,
) -> None:
    learning_root = tmp_path / "learning-path"
    learning_root.mkdir()
    for name in LEARNING_APP_NAMES:
        (learning_root / name).touch()
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "mkdocs.yml").write_text(
        "site_name: fixture\nsite_url: https://docs.example.test/project/\n",
        encoding="utf-8",
    )

    stages = build_stages(
        tmp_path,
        docs_root / "site",
        profile="final",
        learning_workspace=tmp_path.parent / "isolated-learning",
    )
    final_commands = {
        stage.name: stage.command
        for stage in stages
        if stage.name
        in {
            "finalize learning application URLs and canonical metadata",
            "crawl completed generated site",
        }
    }

    assert mkdocs_site_url(tmp_path) == "https://docs.example.test/project/"
    assert all(command[-1] == "https://docs.example.test/project/" for command in final_commands.values())


@pytest.mark.parametrize(
    "config",
    [
        "site_name: missing\n",
        "site_url: https://one.example/\nsite_url: https://two.example/\n",
        "site_url: /relative/site/\n",
        "site_url: https://example.test/site/?preview=true\n",
    ],
)
def test_mkdocs_site_url_rejects_missing_ambiguous_or_noncanonical_values(
    tmp_path: Path,
    config: str,
) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "mkdocs.yml").write_text(config, encoding="utf-8")

    with pytest.raises(GateConfigurationError):
        mkdocs_site_url(tmp_path)


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
    assert "--ci-policy" in ci
    assert "WANDAS_DOCS_BASE_SHA: ${{ github.event.pull_request.base.sha }}" in ci
    assert "fetch-depth: 0" in ci
    assert FINALIZATION_SENTINEL.as_posix() in ci
    assert "--require-final" in deploy
    assert "--site-only" not in deploy
    assert "--group docs --group test" in deploy
    assert "mkdocs build" not in deploy
    assert "marimo export" not in deploy
    assert deploy.index(command) < deploy.index("uses: peaceiris/actions-gh-pages")


def test_stage_specific_working_directory_is_respected(tmp_path: Path) -> None:
    isolated = tmp_path / "isolated"
    seen: list[Path] = []

    def runner(
        command: tuple[str, ...],
        *,
        cwd: Path,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        seen.append(cwd)
        return subprocess.CompletedProcess(command, 0)

    run_stages(
        (Stage("isolated learning export", ("true",), cwd=isolated),),
        repo_root=tmp_path,
        runner=runner,
    )

    assert seen == [isolated]


def test_deliberately_broken_markdown_link_fails_strict_build(tmp_path: Path) -> None:
    pytest.importorskip("mkdocs", reason="broken-link mutation runs in the dedicated docs job")
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
