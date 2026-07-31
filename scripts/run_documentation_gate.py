"""Run the documentation consistency contract once, in deployment order.

The gate has three intentional profiles:

* ``standalone`` uses checks already present on the Issue #375 audit baseline.
* ``integration`` accepts fully installed predecessor checkers while the closing
  cohort is merging, but still rejects a half-installed multi-file checker.
* ``final`` is selected only when the complete prerequisite checker cohort is
  present and is mandatory before deployment.

The final profile delegates each domain to its canonical predecessor
implementation. This module owns only orchestration and fail-fast propagation.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
FINALIZATION_SENTINEL = Path(".github/documentation-gate-finalized")

PREREQUISITE_MARKERS = {
    "#365 spectral numerical contracts": (Path("tests/docs/test_spectral_numerical_contracts.py"),),
    "#373 docstring governance": (
        Path("scripts/check_public_docstrings.py"),
        Path("tests/docs/test_public_docstrings.py"),
    ),
    "#369 public API inventory": (
        Path("wandas/_public_api.py"),
        Path("tests/docs/test_public_api_inventory.py"),
    ),
    "#372 generated-site contract": (
        Path("scripts/check_docs_site.py"),
        Path("scripts/finalize_learning_html.py"),
        Path("tests/docs/test_docs_site.py"),
    ),
    "#367 executable learning path": (Path("tests/docs/test_learning_path_executable_contracts.py"),),
}

NUMERICAL_CONTRACT_TESTS = (
    "tests/processing/test_spectral_operations.py",
    "tests/frames/test_spectral_frame.py",
)
SOURCE_CONTRACT_TESTS = (
    "tests/docs",
    "tests/test_init.py",
    "tests/test_optional_dependencies.py",
    *NUMERICAL_CONTRACT_TESTS,
)

LEARNING_APP_NAMES = (
    "00_why_wandas.py",
    "01_getting_started.py",
    "02_working_with_data.py",
    "03_signal_processing_basics.py",
    "04_advanced_processing.py",
    "05_custom_functions.py",
    "06_reusable_pipeline_recipes.py",
    "07_per_channel_calibration.py",
    "08_metadata_driven_dataset_search.py",
)
LEARNING_FIXTURES = ("sample_audio.wav", "sensor_data.csv")


@dataclass(frozen=True)
class Stage:
    """One fail-fast command in the documentation gate."""

    name: str
    command: tuple[str, ...]
    cwd: Path | None = None


class GateConfigurationError(RuntimeError):
    """Raised when prerequisite checkers are only partly integrated."""


def git_path_exists(repo_root: Path, ref: str, path: Path) -> bool:
    """Return whether *path* exists at a verified Git commit *ref*."""
    verified = subprocess.run(
        ("git", "rev-parse", "--verify", f"{ref}^{{commit}}"),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if verified.returncode:
        raise GateConfigurationError(f"cannot resolve documentation-gate baseline commit: {ref}")
    found = subprocess.run(
        ("git", "cat-file", "-e", f"{ref}:{path.as_posix()}"),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return found.returncode == 0


def ci_requires_final(
    repo_root: Path,
    *,
    event_name: str,
    base_sha: str | None,
    path_exists: Callable[[Path, str, Path], bool] = git_path_exists,
) -> bool:
    """Apply the irreversible CI transition recorded by the finalization sentinel."""
    if event_name == "pull_request":
        if not base_sha:
            raise GateConfigurationError("pull-request documentation CI requires WANDAS_DOCS_BASE_SHA")
        return path_exists(repo_root, base_sha, FINALIZATION_SENTINEL)
    if event_name == "push":
        if not path_exists(repo_root, "HEAD", FINALIZATION_SENTINEL):
            raise GateConfigurationError(f"finalized main push is missing {FINALIZATION_SENTINEL}")
        return True
    raise GateConfigurationError(f"documentation CI policy does not support event {event_name!r}")


def detect_profile(repo_root: Path, *, require_final: bool = False) -> str:
    """Return the safe profile without allowing a half-installed checker."""
    complete = {
        name: all((repo_root / marker).is_file() for marker in markers)
        for name, markers in PREREQUISITE_MARKERS.items()
    }
    partial = {
        name: any((repo_root / marker).is_file() for marker in markers) and not complete[name]
        for name, markers in PREREQUISITE_MARKERS.items()
    }
    if any(partial.values()):
        details = ", ".join(
            f"{name}={'complete' if complete[name] else 'partial' if partial[name] else 'absent'}"
            for name in PREREQUISITE_MARKERS
        )
        raise GateConfigurationError(f"partially installed documentation-gate checker: {details}")
    if all(complete.values()):
        return "final"
    if require_final:
        missing = ", ".join(name for name, found in complete.items() if not found)
        raise GateConfigurationError(f"final documentation gate requires prerequisite checkers: {missing}")
    return "integration" if any(complete.values()) else "standalone"


def validate_mode(profile: str, *, site_only: bool) -> None:
    """Reject source-test-skipping mode unless every final checker is present."""
    if site_only and profile != "final":
        raise GateConfigurationError(f"site-only documentation gate requires final profile; detected {profile}")


def mkdocs_site_url(repo_root: Path) -> str:
    """Read and validate the one canonical site origin from the MkDocs config."""
    config = repo_root / "docs/mkdocs.yml"
    matches = [
        line.partition(":")[2].strip()
        for line in config.read_text(encoding="utf-8").splitlines()
        if line.startswith("site_url:")
    ]
    if len(matches) != 1:
        raise GateConfigurationError(f"expected exactly one top-level site_url in {config}, found {len(matches)}")

    site_url = matches[0]
    if len(site_url) >= 2 and site_url[0] == site_url[-1] and site_url[0] in {'"', "'"}:
        site_url = site_url[1:-1]
    parsed = urlparse(site_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise GateConfigurationError(
            f"MkDocs site_url must be an absolute HTTP(S) origin without query or fragment: {site_url!r}"
        )
    return site_url.rstrip("/") + "/"


def learning_apps(repo_root: Path) -> tuple[Path, ...]:
    """Return the exact numbered learning application inventory or fail."""
    learning_root = repo_root / "learning-path"
    apps = tuple(sorted(path for path in learning_root.glob("*.py") if re.fullmatch(r"\d+_.+\.py", path.name)))
    found = tuple(path.name for path in apps)
    if found != LEARNING_APP_NAMES:
        raise GateConfigurationError(f"expected learning applications {LEARNING_APP_NAMES!r}, found {found!r}")
    return apps


def prepare_learning_workspace(repo_root: Path, workspace: Path) -> None:
    """Copy checked-in learning fixtures into an isolated execution directory."""
    repo_root = repo_root.resolve()
    workspace = workspace.resolve()
    if workspace == repo_root or workspace.is_relative_to(repo_root):
        raise GateConfigurationError(f"learning workspace must be outside the repository: {workspace}")
    source_root = repo_root / "learning-path"
    workspace.mkdir(parents=True, exist_ok=True)
    if any(workspace.iterdir()):
        raise GateConfigurationError(f"learning workspace must start empty: {workspace}")
    for name in LEARNING_FIXTURES:
        source = source_root / name
        if not source.is_file():
            raise GateConfigurationError(f"missing checked-in learning fixture: {source}")
        shutil.copy2(source, workspace / name)


def build_stages(
    repo_root: Path,
    site_dir: Path,
    *,
    profile: str,
    learning_workspace: Path,
    site_only: bool = False,
) -> tuple[Stage, ...]:
    """Build the ordered, single-job documentation validation plan."""
    python = sys.executable
    apps = learning_apps(repo_root)
    stages: list[Stage] = []

    if not site_only:
        stages.append(
            Stage(
                "documentation, API, and numerical contract tests",
                (
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    *SOURCE_CONTRACT_TESTS,
                ),
            )
        )

    if profile == "final":
        stages.append(
            Stage(
                "public docstring parser contract",
                (python, "scripts/check_public_docstrings.py"),
            )
        )

    stages.append(
        Stage(
            "all learning applications: static/reactive check",
            (python, "-m", "marimo", "check", "--strict", *(str(path.relative_to(repo_root)) for path in apps)),
        )
    )
    stages.append(
        Stage(
            "strict MkDocs build",
            (
                python,
                "-m",
                "mkdocs",
                "build",
                "--strict",
                "--clean",
                "-f",
                "docs/mkdocs.yml",
                "-d",
                str(site_dir),
            ),
        )
    )
    stages.append(
        Stage(
            "prepare clean learning export directory",
            (
                python,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path({str(learning_output := site_dir / 'learning-path')!r}).mkdir(parents=True, exist_ok=True)"
                ),
            ),
        )
    )

    for app in apps:
        output = learning_output / app.with_suffix(".html").name
        stages.append(
            Stage(
                f"execute/export {app.name}",
                (
                    python,
                    "-m",
                    "marimo",
                    "export",
                    "html",
                    str(app.resolve()),
                    "-o",
                    str(output),
                    "--no-include-code",
                    "-f",
                ),
                cwd=learning_workspace,
            )
        )

    if profile == "final":
        site_url = mkdocs_site_url(repo_root)
        stages.extend(
            (
                Stage(
                    "finalize learning application URLs and canonical metadata",
                    (
                        python,
                        "scripts/finalize_learning_html.py",
                        str(site_dir),
                        "--site-url",
                        site_url,
                    ),
                ),
                Stage(
                    "crawl completed generated site",
                    (
                        python,
                        "scripts/check_docs_site.py",
                        str(site_dir),
                        "--source-dir",
                        "docs/src",
                        "--site-url",
                        site_url,
                    ),
                ),
            )
        )
    return tuple(stages)


def run_stages(
    stages: Sequence[Stage],
    *,
    repo_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Run *stages* in order and stop at the first non-zero command."""
    for stage in stages:
        print(f"::group::{stage.name}", flush=True)
        completed = runner(
            stage.command,
            cwd=stage.cwd or repo_root,
            text=True,
            check=False,
        )
        print("::endgroup::", flush=True)
        if completed.returncode:
            rendered = " ".join(stage.command)
            raise RuntimeError(f"documentation gate failed at {stage.name!r}: {rendered}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", type=Path, default=Path("docs/site"))
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--require-final",
        action="store_true",
        help="require the complete prerequisite checker cohort (used by deployment)",
    )
    profile_group.add_argument(
        "--ci-policy",
        action="store_true",
        help="require final according to the permanent base/main finalization sentinel",
    )
    parser.add_argument(
        "--site-only",
        action="store_true",
        help="skip pytest while retaining parser, learning, strict build, finalize, and crawl stages",
    )
    args = parser.parse_args()

    try:
        require_final = args.require_final
        if args.ci_policy:
            require_final = ci_requires_final(
                REPO_ROOT,
                event_name=os.environ.get("GITHUB_EVENT_NAME", ""),
                base_sha=os.environ.get("WANDAS_DOCS_BASE_SHA"),
            )
        profile = detect_profile(REPO_ROOT, require_final=require_final)
        validate_mode(profile, site_only=args.site_only)
        site_dir = args.site_dir if args.site_dir.is_absolute() else REPO_ROOT / args.site_dir
        site_dir = site_dir.resolve()
        docs_root = (REPO_ROOT / "docs").resolve()
        try:
            site_dir.relative_to(docs_root)
        except ValueError as exc:
            raise GateConfigurationError(f"site directory must be under {docs_root}: {site_dir}") from exc

        # MkDocs --clean owns the site directory. Remove only the learning
        # subdirectory so stale exports cannot satisfy the exact-nine check.
        learning_output = site_dir / "learning-path"
        if learning_output.exists():
            shutil.rmtree(learning_output)

        with tempfile.TemporaryDirectory(prefix="wandas-learning-gate-") as temporary_directory:
            learning_workspace = Path(temporary_directory)
            prepare_learning_workspace(REPO_ROOT, learning_workspace)
            stages = build_stages(
                REPO_ROOT,
                site_dir,
                profile=profile,
                learning_workspace=learning_workspace,
                site_only=args.site_only,
            )
            print(f"Documentation consistency profile: {profile}", flush=True)
            run_stages(stages, repo_root=REPO_ROOT)
    except (GateConfigurationError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Documentation consistency gate passed ({profile})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
