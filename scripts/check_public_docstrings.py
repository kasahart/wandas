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
    r"^(?P<header>"
    + "|".join(
        re.escape(header) for header in sorted(_GOOGLE_SECTION_KIND, key=lambda header: len(header), reverse=True)
    )
    + r"):(?:\s+(?P<title>\S.*))?\s*$",
    flags=re.IGNORECASE,
)
_NUMPY_SECTION_KIND_CASEFOLD = {header.casefold(): kind for header, kind in _NUMPY_SECTION_KIND.items()}
_NUMPY_CANONICAL_HEADER = {header.casefold(): header for header in _NUMPY_SECTION_KIND}
_SPHINX_FIELD = re.compile(
    r"^:(?:param|parameter|arg|argument|key|keyword|type|var|ivar|cvar|vartype|"
    r"returns?|rtype|raises?|except|exception)(?:\s+\w+)*:(?:\s+.*)?$",
    flags=re.IGNORECASE,
)
SectionIdentity = tuple[str, str | None, str | None]

# Every docstring-bearing dunder in a public class must be classified here.
# Data-model and ecosystem protocols are audited; construction-time invariant
# hooks are intentionally internal. An unknown dunder is surfaced as scope drift.
_AUDITED_DUNDER_METHODS = {
    "__init__": "public constructor",
    "__len__": "Python container protocol",
    "__iter__": "Python iteration protocol",
    "__getitem__": "Python indexing protocol",
    "__setitem__": "Python indexing protocol",
    "__array__": "NumPy array protocol",
    "__array_ufunc__": "NumPy ufunc protocol",
    "__add__": "Python numeric protocol",
    "__sub__": "Python numeric protocol",
    "__mul__": "Python numeric protocol",
    "__truediv__": "Python numeric protocol",
    "__pow__": "Python numeric protocol",
    "__radd__": "Python numeric protocol",
    "__rsub__": "Python numeric protocol",
    "__rmul__": "Python numeric protocol",
    "__rtruediv__": "Python numeric protocol",
    "__rpow__": "Python numeric protocol",
    "__sklearn_is_fitted__": "scikit-learn estimator protocol",
}
_INTERNAL_DUNDER_METHODS = {
    "__post_init__": "dataclass invariant hook",
    "__init_subclass__": "subclass invariant hook",
}


def _admonition_annotation(style: str, header: str) -> str:
    """Return the normalized admonition kind emitted by Griffe."""
    annotation = header.casefold().replace(" ", "-")
    if style == "numpy" and annotation in {"notes", "warnings"}:
        annotation = annotation[:-1]
    return annotation


@dataclass(frozen=True)
class PublicDocstring:
    """One documented public definition found without importing the package."""

    path: Path
    line: int
    qualified_name: str
    value: str
    scope_error: str | None = None

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
    def identity(self) -> SectionIdentity:
        """Return the identity retained by Griffe for ordered matching."""
        if self.kind != "admonition":
            return self.kind, None, None
        return self.kind, _admonition_annotation(self.style, self.header), self.header.casefold()


def _public_definitions(
    body: Iterable[ast.stmt],
    *,
    module_name: str,
    owner: tuple[str, ...] = (),
) -> Iterable[tuple[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, str, str | None]]:
    """Yield public module/class definitions while excluding local functions."""
    for node in body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_dunder_method = bool(owner) and node.name.startswith("__") and node.name.endswith("__")
        scope_error = None
        if is_dunder_method:
            if node.name in _INTERNAL_DUNDER_METHODS:
                continue
            if node.name not in _AUDITED_DUNDER_METHODS:
                if ast.get_docstring(node):
                    scope_error = (
                        f"unclassified docstring-bearing dunder {node.name}; add it to the audited public-protocol "
                        "or explicit internal-hook registry"
                    )
                else:
                    continue
        elif node.name.startswith("_"):
            continue
        qualified_name = ".".join((module_name, *owner, node.name))
        yield node, qualified_name, scope_error
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
        for node, qualified_name, scope_error in _public_definitions(tree.body, module_name=module_name):
            value = ast.get_docstring(node, clean=False)
            if value:
                found.append(PublicDocstring(path, node.lineno, qualified_name, value, scope_error))
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
            has_next_line = index < len(lines) - 1
            has_next_lines = index < len(lines) - 2
            blank_line_below = has_next_line and not lines[index + 1].strip()
            blank_lines_below = has_next_lines and not lines[index + 2].strip()
            indented_line_below = has_next_line and not blank_line_below and lines[index + 1].startswith(" ")
            indented_lines_below = has_next_lines and not blank_lines_below and lines[index + 2].startswith(" ")
            if not (indented_line_below or indented_lines_below):
                continue
            normalized_header = google_match.group("header").casefold()
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
) -> tuple[SectionIdentity, ...]:
    """Return ordered identities for parsed sections recognized by the grammar."""
    section_kinds = _GOOGLE_SECTION_KIND_CASEFOLD if style == "google" else _NUMPY_SECTION_KIND_CASEFOLD
    recognized_kinds = set(section_kinds.values())
    canonical_headers = _GOOGLE_CANONICAL_HEADER.values() if style == "google" else _NUMPY_CANONICAL_HEADER.values()
    allowed_admonitions = {
        DeclaredSection(style, header, "admonition", 0).identity
        for header in canonical_headers
        if section_kinds[header.casefold()] == "admonition"
    }
    identities: list[SectionIdentity] = []
    for section in sections:
        kind = section.kind.value
        if kind not in recognized_kinds:
            continue
        if kind != "admonition":
            identities.append((kind, None, None))
            continue
        title = section.title
        if title is None:
            continue
        annotation = getattr(section.value, "kind", None)
        identity = kind, annotation, title.casefold()
        if identity in allowed_admonitions:
            identities.append(identity)
    return tuple(identities)


def _missing_declarations(
    declared: tuple[DeclaredSection, ...],
    parsed: tuple[SectionIdentity, ...],
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
        if public_docstring.scope_error:
            errors.append(f"{public_docstring.location}: {public_docstring.scope_error}")
            continue
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
