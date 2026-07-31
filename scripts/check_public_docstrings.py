"""Validate structured sections in public Wandas docstrings.

Mkdocstrings delegates Python docstring parsing to Griffe. The site deliberately
uses Griffe's ``auto`` parser because the established public API contains complete
Google- and NumPy-style docstrings. This checker keeps that migration boundary
strict: one docstring may use either style, but not both, and every declared
recognized section must produce its expected Griffe structured-section kind.
"""

from __future__ import annotations

import ast
import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from inspect import cleandoc
from pathlib import Path

from griffe import Docstring, Parser

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SOURCE_ROOT = REPO_ROOT / "wandas"
MKDOCS_CONFIG = REPO_ROOT / "docs" / "mkdocs.yml"

_GOOGLE_SECTION_KIND = {
    "Args": "parameters",
    "Arguments": "parameters",
    "Params": "parameters",
    "Parameters": "parameters",
    "Keyword Args": "other parameters",
    "Keyword Arguments": "other parameters",
    "Other Args": "other parameters",
    "Other Arguments": "other parameters",
    "Other Params": "other parameters",
    "Other Parameters": "other parameters",
    "Type Args": "type parameters",
    "Type Arguments": "type parameters",
    "Type Params": "type parameters",
    "Type Parameters": "type parameters",
    "Returns": "returns",
    "Raises": "raises",
    "Exceptions": "raises",
    "Yields": "yields",
    "Receives": "receives",
    "Examples": "examples",
    "Attributes": "attributes",
    "Functions": "functions",
    "Methods": "functions",
    "Classes": "classes",
    "Type Aliases": "type aliases",
    "Modules": "modules",
    "Warns": "warns",
    "Warnings": "warns",
    # Griffe renders recognized Google-style free-form sections as admonitions.
    "Example": "admonition",
    "Note": "admonition",
    "Notes": "admonition",
    "References": "admonition",
    "See Also": "admonition",
    "Deprecated": "admonition",
}
_NUMPY_SECTION_KIND = {
    "Deprecated": "deprecated",
    "Parameters": "parameters",
    "Other Parameters": "other parameters",
    "Type Parameters": "type parameters",
    "Returns": "returns",
    "Yields": "yields",
    "Receives": "receives",
    "Raises": "raises",
    "Warns": "warns",
    "Examples": "examples",
    "Attributes": "attributes",
    "Functions": "functions",
    "Methods": "functions",
    "Classes": "classes",
    "Type Aliases": "type aliases",
    "Modules": "modules",
    # Numpydoc treats these markup-oriented sections as named admonitions.
    "Warnings": "admonition",
    "Notes": "admonition",
    "References": "admonition",
    "See Also": "admonition",
}
_GOOGLE_SECTION = re.compile(
    r"^("
    + "|".join(
        re.escape(header) for header in sorted(_GOOGLE_SECTION_KIND, key=lambda header: len(header), reverse=True)
    )
    + r"):\s*$"
)


@dataclass(frozen=True)
class PublicDocstring:
    """One documented public definition found without importing the package."""

    path: Path
    line: int
    qualified_name: str
    value: str

    @property
    def location(self) -> str:
        """Return a repository-relative diagnostic location."""
        try:
            display_path = self.path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            display_path = self.path.as_posix()
        return f"{display_path}:{self.line} ({self.qualified_name})"


@dataclass(frozen=True)
class AuditResult:
    """Repository-wide parser audit outcome."""

    checked_docstrings: int
    google_docstrings: int
    numpy_docstrings: int
    structured_sections: int
    errors: tuple[str, ...]


def _public_definitions(
    body: Iterable[ast.stmt],
    *,
    module_name: str,
    owner: tuple[str, ...] = (),
) -> Iterable[tuple[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, str]]:
    """Yield public module/class definitions while excluding local functions."""
    for node in body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_public_constructor = bool(owner) and node.name == "__init__"
        if node.name.startswith("_") and not is_public_constructor:
            continue
        qualified_name = ".".join((module_name, *owner, node.name))
        yield node, qualified_name
        if isinstance(node, ast.ClassDef):
            yield from _public_definitions(
                node.body,
                module_name=module_name,
                owner=(*owner, node.name),
            )


def public_docstrings(source_root: Path = PUBLIC_SOURCE_ROOT) -> tuple[PublicDocstring, ...]:
    """Collect docstrings for public definitions without importing optional modules."""
    found: list[PublicDocstring] = []
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root)
        module_parts = (source_root.name, *relative.with_suffix("").parts)
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        module_name = ".".join(module_parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node, qualified_name in _public_definitions(tree.body, module_name=module_name):
            value = ast.get_docstring(node, clean=False)
            if value:
                found.append(PublicDocstring(path, node.lineno, qualified_name, value))
    return tuple(found)


def _declared_sections(value: str) -> tuple[Counter[str], set[str], list[str]]:
    """Return expected parsed kinds plus the Google/NumPy declarations in use."""
    expected: Counter[str] = Counter()
    styles: set[str] = set()
    headers: list[str] = []
    lines = cleandoc(value).splitlines()
    for index, raw_line in enumerate(lines):
        line = raw_line.rstrip()
        google_match = _GOOGLE_SECTION.match(line)
        if google_match:
            header = google_match.group(1)
            expected[_GOOGLE_SECTION_KIND[header]] += 1
            headers.append(header)
            styles.add("google")
            continue

        header = line
        if raw_line != raw_line.lstrip() or header not in _NUMPY_SECTION_KIND or index + 1 >= len(lines):
            continue
        raw_underline = lines[index + 1]
        underline = raw_underline.strip()
        if raw_underline == raw_underline.lstrip() and underline and set(underline) == {"-"}:
            expected[_NUMPY_SECTION_KIND[header]] += 1
            headers.append(header)
            styles.add("numpy")
    return expected, styles, headers


def configured_docstring_style(config_path: Path = MKDOCS_CONFIG) -> str | None:
    """Read the mkdocstrings parser setting without evaluating Python YAML tags."""
    match = re.search(
        r"^\s*docstring_style:\s*([A-Za-z_]+)\s*$",
        config_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def audit_public_docstrings(source_root: Path = PUBLIC_SOURCE_ROOT) -> AuditResult:
    """Audit every structured public docstring against Griffe's configured parser."""
    logging.getLogger("griffe").setLevel(logging.ERROR)
    errors: list[str] = []
    checked = 0
    google = 0
    numpy = 0
    section_count = 0

    if configured_docstring_style() != Parser.auto.value:
        errors.append("docs/mkdocs.yml must configure mkdocstrings Python docstring_style: auto")

    for public_docstring in public_docstrings(source_root):
        expected, styles, headers = _declared_sections(public_docstring.value)
        google += "google" in styles
        numpy += "numpy" in styles

        if len(styles) > 1:
            errors.append(
                f"{public_docstring.location}: mixes Google and NumPy structured sections "
                f"({', '.join(sorted(headers))})"
            )
            continue

        if not expected:
            continue
        checked += 1
        section_count += sum(expected.values())

        parsed = Counter(section.kind.value for section in Docstring(public_docstring.value).parse(Parser.auto))
        missing = expected - parsed
        if missing:
            errors.append(
                f"{public_docstring.location}: Griffe auto did not parse "
                f"{', '.join(sorted(missing.elements()))}; keep a blank line before Google sections "
                "or a valid underline below NumPy section headings"
            )

    if google == 0 or numpy == 0:
        errors.append("public docstring audit must exercise both established Google and NumPy styles")

    return AuditResult(checked, google, numpy, section_count, tuple(errors))


def main() -> int:
    """Run the repository audit and print a compact CI summary."""
    result = audit_public_docstrings()
    if result.errors:
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Public docstrings valid: "
        f"{result.checked_docstrings} docstrings, "
        f"{result.structured_sections} structured section kinds "
        f"({result.google_docstrings} Google, {result.numpy_docstrings} NumPy)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
