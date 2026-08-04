"""Shared metadata and translation helpers for the Learning Path.

The notebooks import this module at runtime, while the export and validation
scripts use it to resolve the same lesson manifest and locale paths.  The
module intentionally uses only the Python standard library so that opening a
lesson locally does not add a translation-specific dependency.
"""

from __future__ import annotations

import argparse
import ast
import json
import posixpath
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LEARNING_PATH_ROOT = REPO_ROOT / "learning-path"
MANIFEST_PATH = LEARNING_PATH_ROOT / "manifest.json"
CATALOG_ROOT = LEARNING_PATH_ROOT / "translations"
SUPPORTED_LOCALES = ("ja", "en")
DEFAULT_LOCALE = "ja"

# These keys are used by ``navigation_markdown`` rather than by a notebook's
# direct ``t("...")`` calls.  Including them in the validator's used-key set
# keeps unused-key detection honest without asking authors to duplicate
# navigation text in every notebook.
NAVIGATION_KEYS = frozenset(
    {
        "navigation.heading",
        "navigation.previous",
        "navigation.next",
        "navigation.japanese_only",
        "language.ja",
        "language.en",
    }
)


class LearningPathI18nError(ValueError):
    """Raised when Learning Path metadata or translations are invalid."""


@dataclass(frozen=True)
class Lesson:
    """A lesson entry from the central Learning Path manifest."""

    lesson_id: str
    source: str
    locales: tuple[str, ...]
    previous: str | None
    next: str | None
    catalog: str | None
    poc: bool

    @property
    def source_path(self) -> Path:
        return REPO_ROOT / self.source

    @property
    def catalog_path(self) -> Path | None:
        if self.catalog is None:
            return None
        return CATALOG_ROOT / self.catalog


@dataclass(frozen=True)
class TranslationCatalog:
    """A line-oriented JSON catalog for one lesson."""

    lesson_id: str
    messages: Mapping[str, Mapping[str, tuple[str, ...]]]
    locale: str = DEFAULT_LOCALE

    def text(self, key: str, **values: object) -> str:
        """Return a translated message, formatting only declared placeholders."""

        try:
            localized = self.messages[key]
        except KeyError as exc:
            raise LearningPathI18nError(f"Unknown translation key {key!r} for {self.lesson_id}") from exc

        locale = self.locale
        try:
            lines = localized[locale]
        except KeyError as exc:
            raise LearningPathI18nError(f"Missing locale {locale!r} for key {key!r}") from exc

        try:
            return "\n".join(line.format(**values) for line in lines)
        except KeyError as exc:
            raise LearningPathI18nError(
                f"Missing placeholder {exc.args[0]!r} while rendering {self.lesson_id}:{key}"
            ) from exc
        except ValueError as exc:
            raise LearningPathI18nError(f"Invalid placeholder syntax in {self.lesson_id}:{key}") from exc


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LearningPathI18nError(f"Missing Learning Path metadata file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LearningPathI18nError(f"Invalid JSON in {path}: {exc}") from exc


def load_manifest() -> tuple[Lesson, ...]:
    """Load and validate the central lesson manifest."""

    raw = _read_json(MANIFEST_PATH)
    if not isinstance(raw, dict) or not isinstance(raw.get("lessons"), list):
        raise LearningPathI18nError("manifest.json must contain a lessons array")

    manifest_locales = raw.get("locales")
    if not isinstance(manifest_locales, dict) or set(manifest_locales) != set(SUPPORTED_LOCALES):
        raise LearningPathI18nError("manifest.json locales must be exactly ja and en")
    if raw.get("default_locale") != DEFAULT_LOCALE:
        raise LearningPathI18nError(f"manifest.json default_locale must be {DEFAULT_LOCALE!r}")

    lessons: list[Lesson] = []
    ids: set[str] = set()
    for raw_lesson in raw["lessons"]:
        if not isinstance(raw_lesson, dict):
            raise LearningPathI18nError("Every manifest lesson must be an object")

        lesson_id = raw_lesson.get("id")
        source = raw_lesson.get("source")
        locales = raw_lesson.get("locales")
        if not isinstance(lesson_id, str) or not lesson_id:
            raise LearningPathI18nError("Every lesson needs a non-empty id")
        if lesson_id in ids:
            raise LearningPathI18nError(f"Duplicate lesson id: {lesson_id}")
        if not isinstance(source, str) or not source:
            raise LearningPathI18nError(f"Lesson {lesson_id} needs a source path")
        if not isinstance(locales, list) or not locales:
            raise LearningPathI18nError(f"Lesson {lesson_id} needs at least one locale")
        if any(locale not in SUPPORTED_LOCALES for locale in locales):
            raise LearningPathI18nError(f"Lesson {lesson_id} has an unknown locale: {locales}")
        if len(set(locales)) != len(locales):
            raise LearningPathI18nError(f"Lesson {lesson_id} repeats a locale")

        lesson = Lesson(
            lesson_id=lesson_id,
            source=source,
            locales=tuple(locales),
            previous=_optional_string(raw_lesson.get("previous"), f"{lesson_id}.previous"),
            next=_optional_string(raw_lesson.get("next"), f"{lesson_id}.next"),
            catalog=_optional_string(raw_lesson.get("catalog"), f"{lesson_id}.catalog"),
            poc=raw_lesson.get("poc", False) is True,
        )
        if not lesson.source_path.exists():
            raise LearningPathI18nError(f"Lesson source does not exist: {lesson.source}")
        if "en" in lesson.locales and lesson.catalog is None:
            raise LearningPathI18nError(f"English lesson {lesson_id} needs a translation catalog")
        lessons.append(lesson)
        ids.add(lesson_id)

    for lesson in lessons:
        for relation, target in (("previous", lesson.previous), ("next", lesson.next)):
            if target is not None and target not in ids:
                raise LearningPathI18nError(f"{lesson.lesson_id}.{relation} points to unknown lesson {target}")

    return tuple(lessons)


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise LearningPathI18nError(f"{field} must be a non-empty string or null")
    return value


def lesson_by_id(lesson_id: str) -> Lesson:
    """Return one manifest lesson or raise a useful error."""

    for lesson in load_manifest():
        if lesson.lesson_id == lesson_id:
            return lesson
    raise LearningPathI18nError(f"Unknown lesson id: {lesson_id}")


def locale_from_argv(argv: Sequence[str] | None = None) -> str:
    """Read the locale passed after marimo's ``--`` separator."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--locale", choices=SUPPORTED_LOCALES, default=DEFAULT_LOCALE)
    args, _unknown = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return str(args.locale)


def load_catalog(lesson_id: str, locale: str) -> TranslationCatalog:
    """Load a lesson catalog for ``locale`` without adding runtime dependencies."""

    if locale not in SUPPORTED_LOCALES:
        raise LearningPathI18nError(f"Unsupported locale {locale!r}; expected one of {SUPPORTED_LOCALES}")
    lesson = lesson_by_id(lesson_id)
    if locale not in lesson.locales:
        raise LearningPathI18nError(f"Lesson {lesson_id} is not available in locale {locale!r}")
    if lesson.catalog_path is None:
        raise LearningPathI18nError(f"Lesson {lesson_id} has no translation catalog")

    raw = _read_json(lesson.catalog_path)
    if not isinstance(raw, dict) or raw.get("lesson") != lesson_id or not isinstance(raw.get("messages"), dict):
        raise LearningPathI18nError(f"Catalog {lesson.catalog_path} has an invalid lesson or messages field")

    messages: dict[str, dict[str, tuple[str, ...]]] = {}
    for key, raw_locales in raw["messages"].items():
        if not isinstance(key, str) or not key:
            raise LearningPathI18nError(f"Catalog {lesson.catalog_path} has an invalid message key")
        if not isinstance(raw_locales, dict):
            raise LearningPathI18nError(f"Catalog key {key!r} must map to locale objects")
        if set(raw_locales) - set(SUPPORTED_LOCALES):
            unknown = sorted(set(raw_locales) - set(SUPPORTED_LOCALES))
            raise LearningPathI18nError(f"Catalog {lesson.catalog_path} has unknown locales: {unknown}")

        localized: dict[str, tuple[str, ...]] = {}
        for candidate_locale, raw_lines in raw_locales.items():
            if not isinstance(raw_lines, list) or not raw_lines or any(not isinstance(line, str) for line in raw_lines):
                raise LearningPathI18nError(
                    f"Catalog {lesson.catalog_path}:{key}:{candidate_locale} is not a string array"
                )
            if not "\n".join(raw_lines).strip():
                raise LearningPathI18nError(f"Catalog {lesson.catalog_path}:{key}:{candidate_locale} is empty")
            localized[candidate_locale] = tuple(raw_lines)
        messages[key] = localized

    catalog = TranslationCatalog(lesson_id=lesson_id, messages=messages, locale=locale)
    return catalog


def output_path(output_root: Path, lesson: Lesson, locale: str) -> Path:
    """Return the static HTML path for one lesson and locale."""

    if locale not in lesson.locales:
        raise LearningPathI18nError(f"Lesson {lesson.lesson_id} is not available in locale {locale!r}")
    relative = Path("learning-path") / f"{lesson.lesson_id}.html"
    if locale == "en":
        relative = Path("en") / relative
    return output_root / relative


def _site_relative_href(current_locale: str, target_locale: str, target_relative: str) -> str:
    current_dir = Path("en/learning-path" if current_locale == "en" else "learning-path")
    target_path = Path(target_relative)
    if target_locale == "en":
        target_path = Path("en") / target_path
    return posixpath.relpath(target_path.as_posix(), current_dir.as_posix())


def docs_relative_href(locale: str, target: str) -> str:
    """Return a link from a Learning Path page to the Japanese MkDocs site."""

    current_dir = Path("en/learning-path" if locale == "en" else "learning-path")
    return posixpath.relpath(Path(target).as_posix(), current_dir.as_posix())


def navigation_markdown(lesson_id: str, locale: str) -> str:
    """Build locale-aware navigation using only manifest metadata and catalog labels."""

    lesson = lesson_by_id(lesson_id)
    catalog = load_catalog(lesson_id, locale)
    lines = [
        f"## {catalog.text('navigation.heading')}",
        "",
        f"{catalog.text('language.ja')} | {catalog.text('language.en')}",
        "",
    ]

    language_links = []
    for target_locale in SUPPORTED_LOCALES:
        if target_locale not in lesson.locales:
            continue
        target = Path("learning-path") / f"{lesson.lesson_id}.html"
        href = _site_relative_href(locale, target_locale, target.as_posix())
        language_links.append(f"[{catalog.text(f'language.{target_locale}')}]({href})")
    lines[2] = " | ".join(language_links)

    for relation, label_key in (("previous", "navigation.previous"), ("next", "navigation.next")):
        target_id = getattr(lesson, relation)
        if target_id is None:
            continue
        target = lesson_by_id(target_id)
        target_locale = locale if locale in target.locales else DEFAULT_LOCALE
        suffix = "" if target_locale == locale else f" ({catalog.text('navigation.japanese_only')})"
        target_path = Path("learning-path") / f"{target.lesson_id}.html"
        href = _site_relative_href(locale, target_locale, target_path.as_posix())
        lines.append(f"**{catalog.text(label_key)}{suffix}**: [{target.lesson_id}]({href})")

    return "\n".join(lines)


def translation_keys_in_source(source_path: Path) -> set[str]:
    """Collect literal ``t("key")`` references from a notebook source."""

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "t":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
            raise LearningPathI18nError(f"Translation calls in {source_path} must use literal keys")
        keys.add(node.args[0].value)
    return keys


def _placeholder_names(value: str) -> set[str]:
    names: set[str] = set()
    try:
        parsed = Formatter().parse(value)
        for _literal, field_name, _format_spec, _conversion in parsed:
            if field_name is None:
                continue
            if not field_name.isidentifier():
                raise LearningPathI18nError(f"Only simple named placeholders are supported: {{{field_name}}}")
            names.add(field_name)
    except ValueError as exc:
        raise LearningPathI18nError(f"Invalid placeholder syntax in catalog value: {value!r}") from exc
    return names


def _validate_markdown(value: str, location: str) -> None:
    if value.count("```") % 2:
        raise LearningPathI18nError(f"Unbalanced Markdown code fence in {location}")
    for target in _markdown_targets(value):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if "{" in target or "}" in target:
            continue
        if not target.endswith((".html", "/")):
            raise LearningPathI18nError(f"Suspicious internal Markdown link in {location}: {target}")


def _markdown_targets(value: str) -> Iterable[str]:
    marker = "]("
    start = 0
    while True:
        position = value.find(marker, start)
        if position < 0:
            return
        end = value.find(")", position + len(marker))
        if end < 0:
            raise LearningPathI18nError(f"Unclosed Markdown link in catalog value: {value!r}")
        yield value[position + len(marker) : end]
        start = end + 1


def validate_catalogs() -> None:
    """Validate every catalog, source reference, and navigation link."""

    for lesson in load_manifest():
        if lesson.catalog_path is None:
            continue
        if not lesson.catalog_path.exists():
            raise LearningPathI18nError(f"Missing catalog for {lesson.lesson_id}: {lesson.catalog_path}")
        catalogs = {locale: load_catalog(lesson.lesson_id, locale) for locale in lesson.locales}
        key_sets = {locale: set(catalog.messages) for locale, catalog in catalogs.items()}
        if len({frozenset(keys) for keys in key_sets.values()}) != 1:
            raise LearningPathI18nError(f"Locale key sets differ for {lesson.lesson_id}: {key_sets}")

        used_keys = translation_keys_in_source(lesson.source_path) | NAVIGATION_KEYS
        catalog_keys = next(iter(key_sets.values()))
        missing = used_keys - catalog_keys
        unused = catalog_keys - used_keys
        if missing:
            raise LearningPathI18nError(f"Missing translation keys in {lesson.lesson_id}: {sorted(missing)}")
        if unused:
            raise LearningPathI18nError(f"Unused translation keys in {lesson.lesson_id}: {sorted(unused)}")

        for key in sorted(catalog_keys):
            for locale, catalog in catalogs.items():
                if set(catalog.messages[key]) != set(lesson.locales):
                    raise LearningPathI18nError(
                        f"Locale coverage differs for {lesson.lesson_id}:{key}; "
                        f"{locale} has {sorted(catalog.messages[key])}"
                    )
            localized_values = [catalogs[locale].messages[key][locale] for locale in lesson.locales]
            for locale, lines in ((locale, catalogs[locale].messages[key][locale]) for locale in lesson.locales):
                value = "\n".join(lines)
                _validate_markdown(value, f"{lesson.lesson_id}:{key}:{locale}")
            placeholder_sets = [frozenset(_placeholder_names("\n".join(value))) for value in localized_values]
            if len(set(placeholder_sets)) != 1:
                raise LearningPathI18nError(f"Placeholder sets differ for {lesson.lesson_id}:{key}")

        for locale in lesson.locales:
            navigation = navigation_markdown(lesson.lesson_id, locale)
            for target in _markdown_targets(navigation):
                if target.startswith("http"):
                    continue
                candidate = posixpath.normpath((output_path(Path(""), lesson, locale).parent / target).as_posix())
                if not any(
                    candidate == posixpath.normpath(output_path(Path(""), target_lesson, target_locale).as_posix())
                    for target_lesson in load_manifest()
                    for target_locale in target_lesson.locales
                ):
                    raise LearningPathI18nError(
                        f"Navigation link from {lesson.lesson_id}:{locale} does not target a manifest export: {target}"
                    )


def poc_lessons() -> tuple[Lesson, ...]:
    """Return the manifest entries required for the two-lesson PoC."""

    lessons = tuple(lesson for lesson in load_manifest() if lesson.poc)
    if not lessons:
        raise LearningPathI18nError("The manifest does not define any PoC lessons")
    return lessons
