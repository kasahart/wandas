"""Classify changed paths for the pull-request CI workflow.

The classifier deliberately owns only routing policy.  It has no GitHub API
dependency, so the same rules can be exercised locally and in the repository's
contract tests.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

CHECKS = ("native", "lint", "docs", "wheel", "pyodide")
_ROOT_CONFIG_FILES = frozenset(
    {
        ".pre-commit-config.yaml",
        "MANIFEST.in",
        "hatch.toml",
        "mkdocs.yml",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "tox.ini",
        "uv.lock",
    }
)
_PACKAGING_SCRIPT_NAMES = frozenset({"build.py", "package.py", "test_installation.py"})
_PYODIDE_PATH_PREFIXES = (
    "scripts/pyodide/",
    "tests/pyodide/",
    "docs/src/how-to/pyodide",
)


def _normalize(path: str) -> str:
    """Return a repository-relative POSIX path for a changed-file entry."""

    normalized = path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _under(path: str, prefix: str) -> bool:
    return path == prefix.rstrip("/") or path.startswith(prefix)


def _is_build_configuration(path: str) -> bool:
    if path in _ROOT_CONFIG_FILES:
        return True
    return path.startswith(("build/", "packaging/")) or path.startswith(("setup.", "hatch."))


def _is_python_source(path: str) -> bool:
    return path.endswith((".py", ".pyi")) and path.startswith(("wandas/", "tests/", "scripts/", "learning-path/"))


def _is_test_related_script(path: str) -> bool:
    if not path.startswith("scripts/"):
        return False
    name = path.removeprefix("scripts/")
    return name.startswith(("test_", "run_")) or name.startswith("ci_")


def _is_pyodide_path(path: str) -> bool:
    return path.startswith(_PYODIDE_PATH_PREFIXES) or path in {
        "scripts/test_pyodide.sh",
        "scripts/run_pyodide_tests.py",
        "scripts/run_pyodide_tests.mjs",
    }


def _is_known_path(path: str) -> bool:
    return (
        path in {"README.md", "README.ja.md"}
        or path in _ROOT_CONFIG_FILES
        or path.startswith(
            (
                ".github/workflows/",
                "docs/",
                "learning-path/",
                "scripts/",
                "tests/",
                "wandas/",
            )
        )
    )


def classify_paths(paths: Iterable[str]) -> dict[str, bool]:
    """Return the CI jobs selected by ``paths``.

    Unknown paths select every job.  This is intentional: a routing miss must
    increase validation rather than silently reduce it.
    """

    normalized = tuple(path for raw_path in paths if (path := _normalize(raw_path)))
    if not normalized:
        return {**{check: True for check in CHECKS}, "unknown": True}

    selected = {check: False for check in CHECKS}
    unknown = False

    for path in normalized:
        if not _is_known_path(path):
            unknown = True
            continue

        is_workflow = _under(path, ".github/workflows/")
        is_build_config = _is_build_configuration(path)
        is_source = _is_python_source(path)
        is_test_script = _is_test_related_script(path)
        is_pyodide = _is_pyodide_path(path)

        selected["native"] |= (
            _under(path, "wandas/") or _under(path, "tests/") or is_build_config or is_workflow or is_test_script
        )
        selected["lint"] |= is_source or is_build_config or is_workflow
        selected["docs"] |= (
            _under(path, "docs/")
            or _under(path, "learning-path/")
            or _under(path, "wandas/")
            or path in {"README.md", "README.ja.md"}
            or path == "scripts/check_public_docstrings.py"
            or is_build_config
            or is_workflow
        )
        selected["wheel"] |= (
            _under(path, "wandas/")
            or is_build_config
            or is_workflow
            or path.removeprefix("scripts/") in _PACKAGING_SCRIPT_NAMES
        )
        selected["pyodide"] |= is_pyodide or _under(path, "wandas/") or is_build_config or is_workflow

    if unknown:
        selected = {check: True for check in CHECKS}

    selected["unknown"] = unknown
    return selected


def _write_github_outputs(output_path: Path, decision: dict[str, bool]) -> None:
    with output_path.open("a", encoding="utf-8") as output:
        for key, value in decision.items():
            output.write(f"{key}={'true' if value else 'false'}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("paths", nargs="*", help="changed paths; defaults to newline-delimited stdin")
    args = parser.parse_args(argv)

    paths = args.paths or sys.stdin.read().splitlines()
    decision = classify_paths(paths)
    if args.github_output is not None:
        _write_github_outputs(args.github_output, decision)
    for key, value in decision.items():
        print(f"{key}={'true' if value else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
