"""Validate structured sections in public Wandas docstrings.

Mkdocstrings delegates Python docstring parsing to Griffe. The site deliberately
uses Griffe's ``auto`` parser because the established public API contains complete
Google- and NumPy-style docstrings. This checker keeps that migration boundary
strict: one docstring may use either style, but not both, and ``auto`` must retain
the same structured section identities and order as Griffe's explicit parser for
that style. Plain text is not style evidence. The audited surface is public
modules, classes, functions, and methods. Fenced examples are masked without
changing line positions, and Sphinx field lists are rejected explicitly because
the repository permits only Google/NumPy.
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

_SPHINX_FIELD = re.compile(
    r"^:(?:param|parameter|arg|argument|key|keyword|type|var|ivar|cvar|vartype|"
    r"returns?|rtype|raises?|except|exception)(?:\s+\*{0,2}\w+)*:(?:\s+.*)?$",
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


def _mask_fenced_code(value: str) -> str:
    """Mask fenced examples while preserving every source line position."""
    masked: list[str] = []
    in_fenced_code = False
    for raw_line in cleandoc(value).splitlines():
        if raw_line.lstrip(" ").startswith("```"):
            in_fenced_code = not in_fenced_code
            masked.append("")
        elif in_fenced_code:
            masked.append("")
        else:
            masked.append(raw_line)
    return "\n".join(masked)


def _sphinx_fields(value: str) -> tuple[tuple[str, int], ...]:
    """Return unsupported Sphinx field names and stable docstring lines."""
    sphinx_fields: list[tuple[str, int]] = []
    for line_number, raw_line in enumerate(_mask_fenced_code(value).splitlines(), start=1):
        line = raw_line.rstrip()
        if _SPHINX_FIELD.fullmatch(line):
            field = line.split(":", 2)[1].split()[0].casefold()
            sphinx_fields.append((field, line_number))
    return tuple(sphinx_fields)


def _structured_section_identities(sections: Iterable[DocstringSection]) -> tuple[SectionIdentity, ...]:
    """Return ordered non-text identities used as parser-specific evidence."""
    identities: list[SectionIdentity] = []
    for section in sections:
        kind = section.kind.value
        if kind == "text":
            continue
        annotation = getattr(section.value, "kind", None) if kind == "admonition" else None
        title = section.title.casefold() if isinstance(section.title, str) else None
        identities.append((kind, annotation, title))
    return tuple(identities)


def _parse_identities(value: str, parser: Parser) -> tuple[SectionIdentity, ...]:
    """Parse one fence-masked source with a selected Griffe parser."""
    return _structured_section_identities(Docstring(value).parse(parser))


def _format_identities(identities: tuple[SectionIdentity, ...]) -> str:
    """Format section evidence for an actionable diagnostic."""
    formatted = []
    for kind, annotation, title in identities:
        details = "/".join(detail for detail in (annotation, title) if detail)
        formatted.append(f"{kind} ({details})" if details else kind)
    return ", ".join(formatted) or "none"


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
        masked_value = _mask_fenced_code(public_docstring.value)
        sphinx_fields = _sphinx_fields(public_docstring.value)
        if sphinx_fields:
            errors.append(
                f"{public_docstring.location}: uses unsupported Sphinx field-list sections "
                f"({', '.join(f'{field} at docstring line {line}' for field, line in sphinx_fields)}); "
                "use one complete Google or NumPy style"
            )
            continue

        google_identities = _parse_identities(masked_value, Parser.google)
        numpy_identities = _parse_identities(masked_value, Parser.numpy)
        google += bool(google_identities)
        numpy += bool(numpy_identities)

        if google_identities and numpy_identities:
            errors.append(
                f"{public_docstring.location}: mixes Google and NumPy structured evidence "
                f"(Google: {_format_identities(google_identities)}; "
                f"NumPy: {_format_identities(numpy_identities)})"
            )
            continue

        if not google_identities and not numpy_identities:
            continue
        checked += 1
        style = "Google" if google_identities else "NumPy"
        expected_identities = google_identities or numpy_identities
        section_count += len(expected_identities)
        # Mkdocstrings passes the original source to ``auto``. Its style
        # heuristic runs before the selected parser ignores fenced examples,
        # so auditing masked input here could hide a real rendering mismatch.
        auto_identities = _parse_identities(public_docstring.value, Parser.auto)
        if auto_identities != expected_identities:
            errors.append(
                f"{public_docstring.location}: Griffe auto structured identities "
                f"({_format_identities(auto_identities)}) do not match explicit {style} identities "
                f"({_format_identities(expected_identities)}); preserve section kind, title, and order"
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
