"""Shared manifest, catalog, navigation, and export-plan helpers.

The Learning Path uses a deliberately small, standard-library-only layer.  The
lesson source remains the single executable source, while catalogs provide the
locale-specific prose rendered during a static export.
"""

from __future__ import annotations

import argparse
import ast
import json
import posixpath
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LEARNING_PATH_ROOT = REPO_ROOT / "learning-path"
MANIFEST_PATH = LEARNING_PATH_ROOT / "manifest.json"
CATALOG_ROOT = LEARNING_PATH_ROOT / "translations"
COMMON_CATALOG_PATH = CATALOG_ROOT / "common.json"
SUPPORTED_LOCALES = ("ja", "en")
DEFAULT_LOCALE = "ja"

_LESSON_SOURCE_RE = re.compile(r"learning-path/[0-9]{2}_[^/]+\.py\Z")
_PLACEHOLDER_RE = re.compile(r"\[\[([A-Za-z_][A-Za-z0-9_]*)\]\]")
_PLACEHOLDER_TOKEN_RE = re.compile(r"\[\[([^\[\]]*)\]\]")

# These keys are used by helpers rather than direct notebook ``t(...)`` calls.
# They belong to common.json and are deliberately never copied into lesson
# catalogs.
COMMON_KEYS = frozenset(
    {
        "navigation.heading",
        "navigation.previous",
        "navigation.next",
        "navigation.japanese_only",
        "language.ja",
        "language.en",
    }
)

_I18N_HELPER_NAMES = frozenset(
    {
        "docs_reference_links",
        "docs_relative_href",
        "language_switch_markdown",
        "load_catalog",
        "locale_from_argv",
        "navigation_markdown",
    }
)


class LearningPathI18nError(ValueError):
    """Raised when Learning Path metadata, translations, or plans are invalid."""


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
class ExportPlanItem:
    """One deterministic source/locale/output item in an export plan."""

    lesson: Lesson
    locale: str
    output_path: Path
    pass_locale: bool


@dataclass(frozen=True)
class TranslationCatalog:
    """A line-oriented JSON catalog merged with the common catalog."""

    lesson_id: str
    messages: Mapping[str, Mapping[str, tuple[str, ...]]]
    locale: str = DEFAULT_LOCALE

    def text(self, key: str, **values: object) -> str:
        """Render a message using only explicit ``[[name]]`` placeholders."""

        location = f"{self.lesson_id}:{key}:{self.locale}"
        try:
            localized = self.messages[key]
        except KeyError as exc:
            raise LearningPathI18nError(f"Unknown translation key at {location}") from exc

        try:
            lines = localized[self.locale]
        except KeyError as exc:
            raise LearningPathI18nError(f"Missing locale at {location}") from exc

        value = "\n".join(lines)
        names = _placeholder_names(value, location)
        provided = set(values)
        missing = names - provided
        unused = provided - names
        if missing:
            raise LearningPathI18nError(f"Missing placeholder(s) {sorted(missing)} at {location}")
        if unused:
            raise LearningPathI18nError(f"Unused value(s) {sorted(unused)} at {location}")
        return _PLACEHOLDER_RE.sub(lambda match: str(values[match.group(1)]), value)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LearningPathI18nError(f"Missing Learning Path metadata file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LearningPathI18nError(f"Invalid JSON in {path}: {exc}") from exc


def tracked_numbered_lesson_sources() -> tuple[str, ...]:
    """Return tracked numbered lesson sources without seeing ignored scratch files.

    The manifest contract is meaningful only in a Git checkout.  If Git is not
    available or the command fails, fail explicitly instead of silently falling
    back to ``Path.glob`` and treating an ignored scratch notebook as a public
    lesson.
    """

    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z", "--", "learning-path"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise LearningPathI18nError("Cannot verify tracked lessons: git is required for manifest validation") from exc
    if completed.returncode != 0:
        details = completed.stderr.strip() or "git ls-files failed"
        raise LearningPathI18nError(f"Cannot verify tracked lessons: {details}")

    return tuple(sorted(path for path in completed.stdout.split("\0") if _LESSON_SOURCE_RE.fullmatch(path)))


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise LearningPathI18nError(f"{field} must be a non-empty string or null")
    return value


def load_manifest() -> tuple[Lesson, ...]:
    """Load, validate, and derive navigation from the manifest order."""

    raw = _read_json(MANIFEST_PATH)
    if not isinstance(raw, dict) or not isinstance(raw.get("lessons"), list):
        raise LearningPathI18nError("manifest.json must contain a lessons array")

    manifest_locales = raw.get("locales")
    if not isinstance(manifest_locales, dict) or set(manifest_locales) != set(SUPPORTED_LOCALES):
        raise LearningPathI18nError("manifest.json locales must be exactly ja and en")
    if any(not isinstance(label, str) or not label for label in manifest_locales.values()):
        raise LearningPathI18nError("manifest.json locale labels must be non-empty strings")
    if raw.get("default_locale") != DEFAULT_LOCALE:
        raise LearningPathI18nError(f"manifest.json default_locale must be {DEFAULT_LOCALE!r}")

    lessons: list[Lesson] = []
    ids: set[str] = set()
    sources: set[str] = set()
    for raw_lesson in raw["lessons"]:
        if not isinstance(raw_lesson, dict):
            raise LearningPathI18nError("Every manifest lesson must be an object")
        if "previous" in raw_lesson or "next" in raw_lesson:
            raise LearningPathI18nError("previous/next are derived from manifest lesson order; remove those fields")

        lesson_id = raw_lesson.get("id")
        source = raw_lesson.get("source")
        locales = raw_lesson.get("locales")
        if not isinstance(lesson_id, str) or not lesson_id:
            raise LearningPathI18nError("Every lesson needs a non-empty id")
        if lesson_id in ids:
            raise LearningPathI18nError(f"Duplicate lesson id: {lesson_id}")
        if not isinstance(source, str) or not source or not _LESSON_SOURCE_RE.fullmatch(source):
            raise LearningPathI18nError(f"Lesson {lesson_id} needs a numbered learning-path source")
        if Path(source).stem != lesson_id:
            raise LearningPathI18nError(f"Lesson id/source stem mismatch: {lesson_id!r} != {Path(source).stem!r}")
        if source in sources:
            raise LearningPathI18nError(f"Duplicate lesson source: {source}")
        if not isinstance(locales, list) or not locales:
            raise LearningPathI18nError(f"Lesson {lesson_id} needs at least one locale")
        if any(locale not in SUPPORTED_LOCALES for locale in locales):
            raise LearningPathI18nError(f"Lesson {lesson_id} has an unknown locale: {locales}")
        if len(set(locales)) != len(locales):
            raise LearningPathI18nError(f"Lesson {lesson_id} repeats a locale")
        if DEFAULT_LOCALE not in locales:
            raise LearningPathI18nError(f"Lesson {lesson_id} must provide the default locale {DEFAULT_LOCALE!r}")

        catalog = _optional_string(raw_lesson.get("catalog"), f"{lesson_id}.catalog")
        if ("en" in locales) != (catalog is not None):
            raise LearningPathI18nError(f"Lesson {lesson_id} needs a catalog exactly when English is available")
        if catalog is not None and (Path(catalog).is_absolute() or Path(catalog).name != catalog):
            raise LearningPathI18nError(f"Lesson {lesson_id}.catalog must be a file name below translations/")

        poc = raw_lesson.get("poc", False)
        if not isinstance(poc, bool):
            raise LearningPathI18nError(f"Lesson {lesson_id}.poc must be boolean")
        lesson = Lesson(
            lesson_id=lesson_id,
            source=source,
            locales=tuple(locales),
            previous=None,
            next=None,
            catalog=catalog,
            poc=poc,
        )
        if not lesson.source_path.exists():
            raise LearningPathI18nError(f"Lesson source does not exist: {lesson.source}")
        lessons.append(lesson)
        ids.add(lesson_id)
        sources.add(source)

    tracked_sources = set(tracked_numbered_lesson_sources())
    if sources != tracked_sources:
        missing = sorted(tracked_sources - sources)
        extra = sorted(sources - tracked_sources)
        raise LearningPathI18nError(f"Manifest/tracked lesson mismatch; missing={missing}, untracked={extra}")

    derived = []
    for index, lesson in enumerate(lessons):
        derived.append(
            replace(
                lesson,
                previous=lessons[index - 1].lesson_id if index else None,
                next=lessons[index + 1].lesson_id if index + 1 < len(lessons) else None,
            )
        )
    lessons_tuple = tuple(derived)
    build_export_plan(lessons_tuple, Path())
    return lessons_tuple


def lesson_by_id(lesson_id: str) -> Lesson:
    """Return one manifest lesson or raise a useful error."""

    for lesson in load_manifest():
        if lesson.lesson_id == lesson_id:
            return lesson
    raise LearningPathI18nError(f"Unknown lesson id: {lesson_id}")


def translated_lessons(lessons: Iterable[Lesson] | None = None) -> tuple[Lesson, ...]:
    """Return manifest lessons that have an English translation catalog.

    ``poc_lessons`` intentionally remains a small export-selection helper for
    representative CI exports.  HTML validation should use this manifest
    driven predicate instead, so adding an English lesson automatically adds
    it to the detailed validation scope.
    """

    available_lessons = load_manifest() if lessons is None else tuple(lessons)
    return tuple(lesson for lesson in available_lessons if "en" in lesson.locales and lesson.catalog is not None)


def locale_from_argv(argv: Sequence[str] | None = None) -> str:
    """Read the locale passed after marimo's ``--`` separator."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--locale", choices=SUPPORTED_LOCALES, default=DEFAULT_LOCALE)
    args, _unknown = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return str(args.locale)


def _load_messages(path: Path, identity_key: str, identity: str) -> dict[str, dict[str, tuple[str, ...]]]:
    raw = _read_json(path)
    if not isinstance(raw, dict) or raw.get(identity_key) != identity or not isinstance(raw.get("messages"), dict):
        raise LearningPathI18nError(f"Catalog {path} has an invalid {identity_key} or messages field")

    messages: dict[str, dict[str, tuple[str, ...]]] = {}
    for key, raw_locales in raw["messages"].items():
        if not isinstance(key, str) or not key:
            raise LearningPathI18nError(f"Catalog {path} has an invalid message key")
        if not isinstance(raw_locales, dict):
            raise LearningPathI18nError(f"Catalog {path}:{key} must map to locale objects")
        unknown_locales = set(raw_locales) - set(SUPPORTED_LOCALES)
        if unknown_locales:
            raise LearningPathI18nError(f"Catalog {path}:{key} has unknown locales: {sorted(unknown_locales)}")

        localized: dict[str, tuple[str, ...]] = {}
        for locale, raw_lines in raw_locales.items():
            if not isinstance(raw_lines, list) or not raw_lines or any(not isinstance(line, str) for line in raw_lines):
                raise LearningPathI18nError(f"Catalog {path}:{key}:{locale} is not a non-empty string array")
            if not "\n".join(raw_lines).strip():
                raise LearningPathI18nError(f"Catalog {path}:{key}:{locale} is empty")
            localized[locale] = tuple(raw_lines)
        messages[key] = localized
    return messages


def _common_messages() -> dict[str, dict[str, tuple[str, ...]]]:
    return _load_messages(COMMON_CATALOG_PATH, "catalog", "common")


def load_catalog(lesson_id: str, locale: str) -> TranslationCatalog:
    """Load common and lesson-specific messages for one locale."""

    if locale not in SUPPORTED_LOCALES:
        raise LearningPathI18nError(f"Unsupported locale {locale!r}; expected one of {SUPPORTED_LOCALES}")
    lesson = lesson_by_id(lesson_id)
    if locale not in lesson.locales:
        raise LearningPathI18nError(f"Lesson {lesson_id} is not available in locale {locale!r}")
    if lesson.catalog_path is None:
        raise LearningPathI18nError(f"Lesson {lesson_id} has no translation catalog")

    common = _common_messages()
    lesson_messages = _load_messages(lesson.catalog_path, "lesson", lesson_id)
    collisions = sorted(set(common) & set(lesson_messages))
    if collisions:
        raise LearningPathI18nError(f"Common catalog keys are duplicated in {lesson_id}: {collisions}")
    merged = {**common, **lesson_messages}
    return TranslationCatalog(lesson_id=lesson_id, messages=merged, locale=locale)


def output_path(output_root: Path, lesson: Lesson, locale: str) -> Path:
    """Return the static HTML path for one lesson and locale."""

    if locale not in lesson.locales:
        raise LearningPathI18nError(f"Lesson {lesson.lesson_id} is not available in locale {locale!r}")
    relative = Path("learning-path") / f"{lesson.lesson_id}.html"
    if locale == "en":
        relative = Path("en") / relative
    return output_root / relative


def build_export_plan(
    lessons: Iterable[Lesson],
    output_root: Path,
    locale: str | None = None,
) -> tuple[ExportPlanItem, ...]:
    """Build a deterministic, duplicate-free export plan.

    When ``locale`` is supplied, each selected lesson contributes only that
    locale.  Without it, every locale declared by each lesson is retained.
    """

    if locale is not None and locale not in SUPPORTED_LOCALES:
        raise LearningPathI18nError(f"Unsupported locale {locale!r}; expected one of {SUPPORTED_LOCALES}")

    items: list[ExportPlanItem] = []
    seen_outputs: dict[str, tuple[str, str]] = {}
    for lesson in lessons:
        target_locales = (locale,) if locale is not None else lesson.locales
        for target_locale in target_locales:
            path = output_path(output_root, lesson, target_locale)
            key = path.as_posix()
            if key in seen_outputs:
                previous = seen_outputs[key]
                raise LearningPathI18nError(
                    f"Duplicate export path {key}: {previous[0]}:{previous[1]} and {lesson.lesson_id}:{target_locale}"
                )
            seen_outputs[key] = (lesson.lesson_id, target_locale)
            items.append(
                ExportPlanItem(
                    lesson=lesson,
                    locale=target_locale,
                    output_path=path,
                    pass_locale=lesson.catalog_path is not None,
                )
            )
    return tuple(items)


def _site_relative_href(current_locale: str, target_locale: str, target_relative: str) -> str:
    current_dir = Path("en/learning-path" if current_locale == "en" else "learning-path")
    target_path = Path(target_relative)
    if target_locale == "en":
        target_path = Path("en") / target_path
    return posixpath.relpath(target_path.as_posix(), current_dir.as_posix())


def docs_relative_href(locale: str, target: str) -> str:
    """Return a link from a Learning Path page to the Japanese MkDocs site."""

    if locale not in SUPPORTED_LOCALES:
        raise LearningPathI18nError(f"Unsupported locale {locale!r}; expected one of {SUPPORTED_LOCALES}")
    current_dir = Path("en/learning-path" if locale == "en" else "learning-path")
    return posixpath.relpath(Path(target).as_posix(), current_dir.as_posix())


def language_switch_markdown(lesson_id: str, locale: str) -> str:
    """Build the static language switch shown directly below a lesson title."""

    lesson = lesson_by_id(lesson_id)
    catalog = load_catalog(lesson_id, locale)
    links = []
    for target_locale in SUPPORTED_LOCALES:
        if target_locale not in lesson.locales:
            continue
        target = Path("learning-path") / f"{lesson.lesson_id}.html"
        href = _site_relative_href(locale, target_locale, target.as_posix())
        links.append(f"[{catalog.text(f'language.{target_locale}')}]({href})")
    return " | ".join(links)


def navigation_markdown(lesson_id: str, locale: str) -> str:
    """Build previous/next navigation from the manifest order."""

    lesson = lesson_by_id(lesson_id)
    catalog = load_catalog(lesson_id, locale)
    lines = [f"## {catalog.text('navigation.heading')}", ""]
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


def docs_reference_links(locale: str, catalog: TranslationCatalog) -> dict[str, str]:
    """Create 06's docs links and annotate English links to Japanese docs."""

    suffix = f" ({catalog.text('navigation.japanese_only')})" if locale == "en" else ""
    return {
        "how_to_link": (f"[RecipePlan how-to{suffix}]({docs_relative_href(locale, 'how-to/pipeline-recipes/')})"),
        "api_link": f"[Pipeline API reference{suffix}]({docs_relative_href(locale, 'api/pipeline/')})",
    }


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


def _placeholder_names(value: str, location: str) -> set[str]:
    if value.count("[[") != value.count("]]"):
        raise LearningPathI18nError(f"Unbalanced placeholder brackets at {location}")
    names: set[str] = set()
    for match in _PLACEHOLDER_TOKEN_RE.finditer(value):
        name = match.group(1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise LearningPathI18nError(f"Invalid placeholder {name!r} at {location}")
        names.add(name)
    return names


def _validate_markdown(value: str, location: str) -> None:
    if value.count("```") % 2:
        raise LearningPathI18nError(f"Unbalanced Markdown code fence in {location}")
    for target in _markdown_targets(value):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if "[[" in target or "]]" in target:
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


def _validate_message_collection(
    messages: Mapping[str, Mapping[str, tuple[str, ...]]],
    locales: Sequence[str],
    location_prefix: str,
) -> None:
    for key, localized in messages.items():
        if set(localized) != set(locales):
            raise LearningPathI18nError(
                f"Locale coverage differs for {location_prefix}:{key}; found {sorted(localized)}"
            )
        placeholder_sets = []
        for locale in locales:
            location = f"{location_prefix}:{key}:{locale}"
            value = "\n".join(localized[locale])
            _validate_markdown(value, location)
            placeholder_sets.append(frozenset(_placeholder_names(value, location)))
        if len(set(placeholder_sets)) != 1:
            raise LearningPathI18nError(f"Placeholder sets differ for {location_prefix}:{key}")


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return (*parent, node.attr) if parent else None
    return None


def _is_hidden_cell(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call) or _attribute_path(decorator.func) != ("app", "cell"):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "hide_code":
                return isinstance(keyword.value, ast.Constant) and keyword.value.value is True
        return False
    return False


def _reject_visible_i18n_ast(tree: ast.AST, location: str) -> None:
    """Reject actual i18n machinery while allowing ordinary identifiers.

    ``t``, ``locale``, and ``catalog`` are useful names in learning material,
    so their mere presence is not evidence of leaked implementation details.
    This check deliberately looks at Python syntax instead of searching source
    text.  The same function is used for source cells and exported cell
    metadata so the two validation paths cannot drift apart.
    """

    for child in ast.walk(tree):
        if isinstance(child, ast.Import):
            if any(alias.name.startswith("scripts.learning_path_i18n") for alias in child.names):
                raise LearningPathI18nError(f"i18n import appears in visible code {location}")
        elif isinstance(child, ast.ImportFrom):
            if child.module and child.module.startswith("scripts.learning_path_i18n"):
                raise LearningPathI18nError(f"i18n import appears in visible code {location}")
        elif isinstance(child, ast.Call):
            function_path = _attribute_path(child.func)
            if function_path and function_path[-1] in _I18N_HELPER_NAMES:
                raise LearningPathI18nError(f"i18n helper {function_path[-1]!r} is called in visible code {location}")
            if function_path == ("catalog", "text"):
                raise LearningPathI18nError(f"catalog.text() is called in visible code {location}")
            if (
                function_path == ("t",)
                and child.args
                and isinstance(child.args[0], ast.Constant)
                and isinstance(child.args[0].value, str)
            ):
                raise LearningPathI18nError(f"translation call appears in visible code {location}")
            for keyword in child.keywords:
                if keyword.arg not in {"label", "title", "xlabel", "ylabel"}:
                    continue
                if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    if not keyword.value.value.isascii():
                        raise LearningPathI18nError(f"localized plot label must be ASCII in {location}")


def validate_visible_code(code: str, location: str | Path) -> None:
    """Validate one learner-visible cell using the source AST contract."""

    try:
        tree = ast.parse(code, filename=str(location))
    except SyntaxError as exc:
        raise LearningPathI18nError(f"Invalid visible Python code in {location}: {exc}") from exc
    _reject_visible_i18n_ast(tree, str(location))


def validate_visible_source(source_path: Path) -> None:
    """Reject i18n implementation details in learner-visible cells."""

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not any(
            _attribute_path(decorator.func if isinstance(decorator, ast.Call) else decorator) == ("app", "cell")
            for decorator in node.decorator_list
        ):
            continue
        if _is_hidden_cell(node):
            continue
        _reject_visible_i18n_ast(node, f"{source_path}:{node.lineno}")


def validate_catalogs() -> None:
    """Validate manifest, common/lesson catalogs, sources, and navigation."""

    common = _common_messages()
    if set(common) != COMMON_KEYS:
        raise LearningPathI18nError(f"common.json keys must be exactly {sorted(COMMON_KEYS)}; found {sorted(common)}")
    _validate_message_collection(common, SUPPORTED_LOCALES, "common")

    for lesson in load_manifest():
        if lesson.catalog_path is None:
            continue
        lesson_messages = _load_messages(lesson.catalog_path, "lesson", lesson.lesson_id)
        collisions = sorted(set(common) & set(lesson_messages))
        if collisions:
            raise LearningPathI18nError(f"Common catalog keys are duplicated in {lesson.lesson_id}: {collisions}")
        _validate_message_collection(lesson_messages, lesson.locales, lesson.lesson_id)
        validate_visible_source(lesson.source_path)

        used_keys = translation_keys_in_source(lesson.source_path)
        lesson_keys = set(lesson_messages)
        missing = (used_keys - COMMON_KEYS) - lesson_keys
        unused = lesson_keys - used_keys
        if missing:
            raise LearningPathI18nError(f"Missing translation keys in {lesson.lesson_id}: {sorted(missing)}")
        if unused:
            raise LearningPathI18nError(f"Unused translation keys in {lesson.lesson_id}: {sorted(unused)}")

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


def poc_lessons(lessons: Iterable[Lesson] | None = None) -> tuple[Lesson, ...]:
    """Return the two manifest entries required for the PoC."""

    available_lessons = load_manifest() if lessons is None else tuple(lessons)
    poc = tuple(lesson for lesson in available_lessons if lesson.poc)
    expected = {"01_getting_started", "06_reusable_pipeline_recipes"}
    if {lesson.lesson_id for lesson in poc} != expected:
        raise LearningPathI18nError(f"PoC lessons must be exactly {sorted(expected)}")
    return poc
