"""Run the documentation consistency contract once, in deployment order.

The gate has two intentional profiles:

* ``standalone`` uses checks already present on the Issue #375 audit baseline.
* ``final`` is selected only when the complete prerequisite checker cohort is
  present. A partial cohort is an error, so integration can never silently skip
  a predecessor check.

The final profile delegates each domain to its canonical predecessor
implementation. This module owns only orchestration and fail-fast propagation.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_URL = "https://kasahart.github.io/wandas/"

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


@dataclass(frozen=True)
class Stage:
    """One fail-fast command in the documentation gate."""

    name: str
    command: tuple[str, ...]


class GateConfigurationError(RuntimeError):
    """Raised when prerequisite checkers are only partly integrated."""


def detect_profile(repo_root: Path, *, require_final: bool = False) -> str:
    """Return ``standalone`` or ``final`` without allowing a partial cohort."""
    complete = {
        name: all((repo_root / marker).is_file() for marker in markers)
        for name, markers in PREREQUISITE_MARKERS.items()
    }
    any_marker = any((repo_root / marker).is_file() for markers in PREREQUISITE_MARKERS.values() for marker in markers)
    if all(complete.values()):
        return "final"
    if any_marker:
        details = ", ".join(f"{name}={'complete' if found else 'incomplete'}" for name, found in complete.items())
        raise GateConfigurationError(f"partial documentation-gate prerequisite cohort: {details}")
    if require_final:
        required = ", ".join(PREREQUISITE_MARKERS)
        raise GateConfigurationError(f"final documentation gate requires prerequisite checkers: {required}")
    return "standalone"


def learning_apps(repo_root: Path) -> tuple[Path, ...]:
    """Return all nine versioned learning applications or fail."""
    apps = tuple(sorted((repo_root / "learning-path").glob("0[0-8]_*.py")))
    prefixes = [path.name[:2] for path in apps]
    expected = [f"{index:02d}" for index in range(9)]
    if prefixes != expected:
        raise GateConfigurationError(f"expected learning applications {expected!r}, found {prefixes!r}")
    return apps


def build_stages(
    repo_root: Path,
    site_dir: Path,
    *,
    profile: str,
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
                    "tests/docs",
                    "tests/test_init.py",
                    *NUMERICAL_CONTRACT_TESTS,
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
                    str(app.relative_to(repo_root)),
                    "-o",
                    str(output),
                    "--no-include-code",
                    "-f",
                ),
            )
        )

    if profile == "final":
        stages.extend(
            (
                Stage(
                    "finalize learning application URLs and canonical metadata",
                    (
                        python,
                        "scripts/finalize_learning_html.py",
                        str(site_dir),
                        "--site-url",
                        SITE_URL,
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
                        SITE_URL,
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
        completed = runner(stage.command, cwd=repo_root, text=True, check=False)
        print("::endgroup::", flush=True)
        if completed.returncode:
            rendered = " ".join(stage.command)
            raise RuntimeError(f"documentation gate failed at {stage.name!r}: {rendered}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", type=Path, default=Path("docs/site"))
    parser.add_argument(
        "--require-final",
        action="store_true",
        help="require the complete prerequisite checker cohort (used by deployment)",
    )
    parser.add_argument(
        "--site-only",
        action="store_true",
        help="skip pytest while retaining parser, learning, strict build, finalize, and crawl stages",
    )
    args = parser.parse_args()

    try:
        profile = detect_profile(REPO_ROOT, require_final=args.require_final)
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

        stages = build_stages(REPO_ROOT, site_dir, profile=profile, site_only=args.site_only)
        print(f"Documentation consistency profile: {profile}", flush=True)
        run_stages(stages, repo_root=REPO_ROOT)
    except (GateConfigurationError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Documentation consistency gate passed ({profile})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
