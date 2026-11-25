"""Keep the test names cited in the component test specification real."""

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = "docs/src/contributing/component-test-specification.md"
TEST_LINK = re.compile(r"\]\(https://github\.com/kasahart/wandas/blob/main/(tests/[^)]+\.py)\)([^\[]*)")
RULE = f"Rule: each 'Existing assertion' cell names a test that exists in the linked file (see {SPEC_PATH})."


def _defined_tests(path: Path) -> set[str]:
    """Return the names pytest collects: top-level ``test_*`` functions and ``test_*`` methods of ``Test*`` classes.

    Each method is listed as ``method`` and as ``Class.method``, so a cited helper name fails the check.
    """
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name.startswith("test_"):
                    names.update({member.name, f"{node.name}.{member.name}"})
    return names


def _cited_tests() -> list[tuple[str, str, list[str]]]:
    """Return (row ID, test file, cited names) for every table row that links a test."""
    citations = []
    for line in (REPO_ROOT / SPEC_PATH).read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or "/blob/main/tests/" not in line:
            continue
        row_id = line.split("|")[1].strip()
        for test_file, after_link in TEST_LINK.findall(line):
            citations.append((row_id, test_file, re.findall(r"`([^`]+)`", after_link.split("|")[0])))
    return citations


def test_component_test_specification_cites_existing_tests() -> None:
    citations = _cited_tests()
    assert citations, f"No test links were parsed from {SPEC_PATH}. {RULE} Fix: update the parser in this test."

    problems = []
    defined_by_file: dict[str, set[str]] = {}
    for row_id, test_file, names in citations:
        path = REPO_ROOT / test_file
        if not path.is_file():
            problems.append(f"row {row_id} links {test_file}, which does not exist")
            continue
        if not names:
            problems.append(f"row {row_id} links {test_file} but names no test in backticks")
            continue
        defined = defined_by_file.setdefault(test_file, _defined_tests(path))
        problems.extend(
            f"row {row_id} cites {test_file}::{name}, which does not exist" for name in names if name not in defined
        )

    assert not problems, (
        f"{SPEC_PATH} cites tests that do not exist:\n  "
        + "\n  ".join(problems)
        + f"\n{RULE}\nFix: change the cell to the test that now asserts the row (use Class.method for a method), "
        "or mark the row Partial or Unverified when no test asserts it."
    )
