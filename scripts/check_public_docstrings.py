"""Validate structured sections in public Wandas docstrings.

Mkdocstrings delegates Python docstring parsing to Griffe. The site deliberately
uses Griffe's ``auto`` parser because the established public API contains complete
Google- and NumPy-style docstrings. This checker keeps that migration boundary
strict: one docstring may use either style, but not both, and every declared
recognized section must produce its expected Griffe structured-section kind.
The audited surface is public modules, classes, functions, and methods. The raw
declaration grammar ignores Griffe-style fenced code, normalizes recognized
headings the same way as Griffe, treats unknown headings as prose, and rejects
Sphinx field lists explicitly because the repository permits only Google/NumPy.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from inspect import cleandoc
from pathlib import Path

from griffe import Docstring, DocstringSection, Parser

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
_GOOGLE_SECTION_KIND_CASEFOLD = {header.casefold(): kind for header, kind in _GOOGLE_SECTION_KIND.items()}
_GOOGLE_CANONICAL_HEADER = {header.casefold(): header for header in _GOOGLE_SECTION_KIND}
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
    + r"):(?:\s+\S.*)?\s*$",
    flags=re.IGNORECASE,
)
_NUMPY_SECTION_KIND_CASEFOLD = {header.casefold(): kind for header, kind in _NUMPY_SECTION_KIND.items()}
_NUMPY_CANONICAL_HEADER = {header.casefold(): header for header in _NUMPY_SECTION_KIND}
_SPHINX_FIELD = re.compile(
    r"^:(?:param|parameter|arg|argument|key|keyword|type|var|ivar|cvar|vartype|"
    r"returns?|rtype|raises?|except|exception)(?:\s+\w+)*:(?:\s+.*)?$",
    flags=re.IGNORECASE,
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

    audited_docstrings: int
    checked_docstrings: int
    google_docstrings: int
    numpy_docstrings: int
    structured_sections: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class DeclaredSection:
    """One recognized raw heading and its expected parsed identity."""

    style: str
    header: str
    kind: str
    line: int

    @property
    def identity(self) -> tuple[str, str | None]:
        """Return the identity retained by Griffe for ordered matching."""
        title = self.header.casefold() if self.kind == "admonition" else None
        return self.kind, title


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
    """Collect public module/definition docstrings without importing modules."""
    found: list[PublicDocstring] = []
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root)
        is_public_module = all(
            not part.startswith("_") for part in relative.with_suffix("").parts if part != "__init__"
        )
        if not is_public_module:
            continue
        module_parts = (source_root.name, *relative.with_suffix("").parts)
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        module_name = ".".join(module_parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_value = ast.get_docstring(tree, clean=False)
        if module_value:
            docstring_node = tree.body[0]
            found.append(PublicDocstring(path, docstring_node.lineno, module_name, module_value))
        for node, qualified_name in _public_definitions(tree.body, module_name=module_name):
            value = ast.get_docstring(node, clean=False)
            if value:
                found.append(PublicDocstring(path, node.lineno, qualified_name, value))
    return tuple(found)


def _declared_sections(value: str) -> tuple[tuple[DeclaredSection, ...], set[str], list[str]]:
    """Lex top-level declarations and return their expected Griffe kinds/styles."""
    declared: list[DeclaredSection] = []
    styles: set[str] = set()
    sphinx_fields: list[str] = []
    lines = cleandoc(value).splitlines()
    in_fenced_code = False
    for index, raw_line in enumerate(lines):
        # Griffe's Google and NumPy parsers toggle fenced-code state on any
        # indentation followed by three backticks. Keep the declaration lexer
        # on that same boundary so literal examples cannot become sections.
        if raw_line.lstrip(" ").startswith("```"):
            in_fenced_code = not in_fenced_code
            continue
        if in_fenced_code:
            continue

        line = raw_line.rstrip()
        google_match = _GOOGLE_SECTION.match(line)
        if google_match:
            normalized_header = google_match.group(1).casefold()
            header = _GOOGLE_CANONICAL_HEADER[normalized_header]
            declared.append(
                DeclaredSection("google", header, _GOOGLE_SECTION_KIND_CASEFOLD[normalized_header], index + 1)
            )
            styles.add("google")
            continue

        if _SPHINX_FIELD.match(line):
            sphinx_fields.append(line.split(":", 2)[1].split()[0].casefold())
            styles.add("sphinx")
            continue

        normalized_header = raw_line.casefold()
        if (
            raw_line != raw_line.lstrip()
            or normalized_header not in _NUMPY_SECTION_KIND_CASEFOLD
            or index + 1 >= len(lines)
        ):
            continue
        raw_underline = lines[index + 1]
        # This is Griffe's ``_is_dash_line`` rule: the underline must be
        # non-empty after whitespace removal and consist only of hyphens.
        if raw_underline.strip() and not raw_underline.replace("-", "").strip():
            header = _NUMPY_CANONICAL_HEADER[normalized_header]
            declared.append(
                DeclaredSection("numpy", header, _NUMPY_SECTION_KIND_CASEFOLD[normalized_header], index + 1)
            )
            styles.add("numpy")
    return tuple(declared), styles, sphinx_fields


def _parsed_section_identities(
    sections: Iterable[DocstringSection],
    *,
    style: str,
) -> tuple[tuple[str, str | None], ...]:
    """Return ordered identities for parsed sections recognized by the grammar."""
    section_kinds = _GOOGLE_SECTION_KIND_CASEFOLD if style == "google" else _NUMPY_SECTION_KIND_CASEFOLD
    recognized_kinds = set(section_kinds.values())
    identities: list[tuple[str, str | None]] = []
    for section in sections:
        kind = section.kind.value
        if kind not in recognized_kinds:
            continue
        if kind != "admonition":
            identities.append((kind, None))
            continue
        title = section.title
        if title is None:
            continue
        normalized_title = title.casefold()
        if section_kinds.get(normalized_title) == "admonition":
            identities.append((kind, normalized_title))
    return tuple(identities)


def _missing_declarations(
    declared: tuple[DeclaredSection, ...],
    parsed: tuple[tuple[str, str | None], ...],
) -> tuple[DeclaredSection, ...]:
    """Match declarations to parsed identities in source order."""
    missing: list[DeclaredSection] = []
    parsed_index = 0
    for declaration in declared:
        if parsed_index < len(parsed) and parsed[parsed_index] == declaration.identity:
            parsed_index += 1
        else:
            missing.append(declaration)
    return tuple(missing)


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

    docstrings = public_docstrings(source_root)
    for public_docstring in docstrings:
        declared, styles, sphinx_fields = _declared_sections(public_docstring.value)
        headers = [section.header for section in declared]
        google += "google" in styles
        numpy += "numpy" in styles

        if "sphinx" in styles:
            errors.append(
                f"{public_docstring.location}: uses unsupported Sphinx field-list sections "
                f"({', '.join(sorted(sphinx_fields))}); use one complete Google or NumPy style"
            )
            continue

        if len(styles) > 1:
            errors.append(
                f"{public_docstring.location}: mixes Google and NumPy structured sections "
                f"({', '.join(sorted(headers))})"
            )
            continue

        if not declared:
            continue
        checked += 1
        section_count += len(declared)

        style = next(iter(styles))
        parsed = _parsed_section_identities(Docstring(public_docstring.value).parse(Parser.auto), style=style)
        missing = _missing_declarations(declared, parsed)
        if missing:
            errors.append(
                f"{public_docstring.location}: Griffe auto did not parse "
                f"{', '.join(f'{section.header} (docstring line {section.line})' for section in missing)}; "
                "keep a blank line before Google sections "
                "or a valid underline below NumPy section headings"
            )

    if google == 0 or numpy == 0:
        errors.append("public docstring audit must exercise both established Google and NumPy styles")

    return AuditResult(len(docstrings), checked, google, numpy, section_count, tuple(errors))


def main() -> int:
    """Run the repository audit and print a compact CI summary."""
    result = audit_public_docstrings()
    if result.errors:
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Public docstrings valid: "
        f"{result.audited_docstrings} audited, "
        f"{result.checked_docstrings} with structured sections, "
        f"{result.structured_sections} structured section kinds "
        f"({result.google_docstrings} Google, {result.numpy_docstrings} NumPy)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
