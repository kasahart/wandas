"""Validate Learning Path catalogs, export plans, and static HTML."""

from __future__ import annotations

import argparse
import json
import posixpath
import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

try:
    from .learning_path_i18n import (
        LearningPathI18nError,
        Lesson,
        _markdown_targets,
        build_export_plan,
        docs_reference_links,
        language_switch_markdown,
        load_catalog,
        load_manifest,
        navigation_markdown,
        output_path,
        poc_lessons,
        translated_lessons,
        validate_catalogs,
        validate_visible_code,
    )
except ImportError:  # Running this file directly keeps the existing python scripts/foo.py workflow.
    from learning_path_i18n import (
        LearningPathI18nError,
        Lesson,
        _markdown_targets,
        build_export_plan,
        docs_reference_links,
        language_switch_markdown,
        load_catalog,
        load_manifest,
        navigation_markdown,
        output_path,
        poc_lessons,
        translated_lessons,
        validate_catalogs,
        validate_visible_code,
    )

# These are deliberately kept as PoC-specific regression checks. The general
# validator compares every translated lesson's exported visible cells.
_POC_CODE_MARKERS = {
    "01_getting_started": ("wd.generate_sin", "combined_signal.fft().plot"),
    "06_reusable_pipeline_recipes": ("RecipePlan.from_frame", "loaded_recipe.apply"),
}


def _normalized_site_target(current: Path, href: str) -> str:
    return posixpath.normpath((current.parent / href).as_posix())


def _serialized_text(value: str) -> tuple[str, str]:
    """Return the literal and JSON-escaped forms used by a marimo export."""

    return value, json.dumps(value, ensure_ascii=True)[1:-1]


def _first_position(html: str, value: str) -> int:
    positions = [html.find(candidate) for candidate in _serialized_text(value)]
    positions = [position for position in positions if position >= 0]
    return min(positions, default=-1)


def _first_href_position(html: str, hrefs: set[str]) -> int:
    positions = [html.find(href) for href in hrefs]
    positions = [position for position in positions if position >= 0]
    return min(positions, default=-1)


def _exported_notebook_cells(html: str, page: Path) -> list[dict[str, object]]:
    """Read marimo's exported cell metadata without snapshotting the HTML."""

    marker = '"notebook": '
    marker_position = html.find(marker)
    if marker_position < 0:
        raise LearningPathI18nError(f"Exported marimo notebook metadata is missing: {page}")
    try:
        notebook, _ = json.JSONDecoder().raw_decode(html[marker_position + len(marker) :])
    except json.JSONDecodeError as exc:
        raise LearningPathI18nError(f"Exported marimo notebook metadata is invalid: {page}") from exc
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise LearningPathI18nError(f"Exported marimo notebook cells are missing: {page}")
    cells = notebook["cells"]
    if not all(isinstance(cell, dict) for cell in cells):
        raise LearningPathI18nError(f"Exported marimo notebook contains an invalid cell: {page}")
    return cells


def _visible_exported_code(html: str, page: Path) -> tuple[str, ...]:
    """Return exported visible cell source, excluding hidden implementation cells."""

    visible_code: list[str] = []
    for cell in _exported_notebook_cells(html, page):
        code = cell.get("code")
        config = cell.get("config")
        if not isinstance(code, str) or not isinstance(config, Mapping):
            raise LearningPathI18nError(f"Exported marimo cell metadata is invalid: {page}")
        config_mapping = cast(Mapping[str, object], config)
        if config_mapping.get("hide_code") is True:
            continue
        visible_code.append(code)
    return tuple(visible_code)


def _validate_visible_exported_code(html: str, page: Path) -> None:
    """Ensure exported visible cells do not expose the translation machinery."""

    for code in _visible_exported_code(html, page):
        validate_visible_code(code, page)


_PYTHON_ERROR_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"(?<!except )\bModuleNotFoundError:\s+\S"),
    re.compile(r"(?<!except )\bImportError:\s+\S"),
)


def _contains_python_error_text(html: str) -> bool:
    """Return whether HTML contains an actual traceback/error line.

    A lesson may legitimately show ``except ImportError:`` as source code;
    that is not an execution failure.  Error markers therefore require a
    traceback heading or an exception name followed by an error message.
    """

    return any(pattern.search(html) for pattern in _PYTHON_ERROR_PATTERNS)


def _validate_shared_visible_code(
    japanese_html: str,
    japanese_page: Path,
    english_html: str,
    english_page: Path,
) -> None:
    """Ensure ja/en exports preserve the same visible source-cell sequence."""

    japanese_code = _visible_exported_code(japanese_html, japanese_page)
    english_code = _visible_exported_code(english_html, english_page)
    if japanese_code == english_code:
        return

    for index in range(max(len(japanese_code), len(english_code))):
        ja_cell = japanese_code[index] if index < len(japanese_code) else "<missing>"
        en_cell = english_code[index] if index < len(english_code) else "<missing>"
        if ja_cell != en_cell:
            raise LearningPathI18nError(
                f"Visible cell code differs between locale exports at cell {index}: {japanese_page} != {english_page}"
            )


def _validate_poc_code_markers(lesson_id: str, html: str, page: Path) -> None:
    """Run legacy fixed markers only for the two PoC lessons."""

    for marker in _POC_CODE_MARKERS.get(lesson_id, ()):
        if marker not in html:
            raise LearningPathI18nError(f"Missing shared code marker {marker!r} in {page}")


def _validate_exported_layout(html: str, page: Path) -> None:
    """Check the title/switch/navigation cell order without a full snapshot."""

    cells = _exported_notebook_cells(html, page)
    marker_indices: dict[str, list[int]] = {"title": [], "switch": [], "navigation": []}
    for index, cell in enumerate(cells):
        code = cell.get("code")
        if not isinstance(code, str):
            continue
        if "t('title')" in code or 't("title")' in code:
            marker_indices["title"].append(index)
        if "language_switch_markdown(" in code:
            marker_indices["switch"].append(index)
        if "navigation_markdown(" in code:
            marker_indices["navigation"].append(index)

    if any(len(indices) != 1 for indices in marker_indices.values()):
        raise LearningPathI18nError(f"Exported title/switch/navigation cells are ambiguous: {page}")
    title_index = marker_indices["title"][0]
    switch_index = marker_indices["switch"][0]
    navigation_index = marker_indices["navigation"][0]
    if switch_index != title_index + 1:
        raise LearningPathI18nError(f"Language switch is not immediately below the title cell: {page}")
    if navigation_index != len(cells) - 1:
        raise LearningPathI18nError(f"Navigation is not the final cell: {page}")

    for index in (title_index, switch_index, navigation_index):
        config = cells[index].get("config")
        if not isinstance(config, Mapping) or cast(Mapping[str, object], config).get("hide_code") is not True:
            raise LearningPathI18nError(f"Title, switch, and navigation cells must hide code: {page}")


def _translated_site_lessons(site_root: Path, manifest_lessons: tuple[Lesson, ...]) -> tuple[Lesson, ...]:
    """Select translated lessons represented by a full or lesson-sized site.

    CI passes a site containing every translated lesson.  Local lesson-focused
    checks often export one lesson, so a translated validation accepts a
    non-empty manifest subset and still requires both locales for each lesson
    represented by the site.
    """

    candidates = []
    for lesson in translated_lessons(manifest_lessons):
        if any(output_path(site_root, lesson, locale).exists() for locale in lesson.locales):
            candidates.append(lesson)
    if not candidates:
        raise LearningPathI18nError(f"No translated lesson export found below {site_root}")
    return tuple(candidates)


def validate_exported_site(
    site_root: Path,
    *,
    validate_all: bool = False,
    validate_translated: bool = False,
) -> None:
    """Check planned files and translated HTML without snapshots or pixel comparison."""

    if validate_all and validate_translated:
        raise LearningPathI18nError("--all and --translated are mutually exclusive site scopes")
    manifest_lessons = load_manifest()
    if validate_all:
        planned_lessons = manifest_lessons
        detailed_lessons = translated_lessons(manifest_lessons)
    elif validate_translated:
        detailed_lessons = _translated_site_lessons(site_root, manifest_lessons)
        planned_lessons = detailed_lessons
    else:
        planned_lessons = poc_lessons(manifest_lessons)
        detailed_lessons = planned_lessons
    lessons_by_id = {lesson.lesson_id: lesson for lesson in manifest_lessons}
    plan = build_export_plan(planned_lessons, site_root)
    for target in plan:
        if not target.output_path.exists():
            raise LearningPathI18nError(f"Missing planned export: {target.output_path}")

    for lesson in detailed_lessons:
        pages: dict[str, tuple[Path, str]] = {}
        for locale in lesson.locales:
            page = output_path(site_root, lesson, locale)
            html = page.read_text(encoding="utf-8")
            pages[locale] = (page, html)
            if _contains_python_error_text(html):
                raise LearningPathI18nError(f"Python error text found in exported page: {page}")
            _validate_visible_exported_code(html, page)
            _validate_exported_layout(html, page)
            _validate_poc_code_markers(lesson.lesson_id, html, page)

            catalog = load_catalog(lesson.lesson_id, locale)
            title = catalog.text("title")
            if _first_position(html, title) < 0:
                raise LearningPathI18nError(f"Missing {locale} title in {page}: {title}")

            switch = language_switch_markdown(lesson.lesson_id, locale)
            navigation = navigation_markdown(lesson.lesson_id, locale)
            expected_hrefs = set(_markdown_targets(switch)) | set(_markdown_targets(navigation))
            navigation_hrefs = set(_markdown_targets(navigation))
            summary = ""
            if lesson.lesson_id == "06_reusable_pipeline_recipes":
                summary = catalog.text("summary", **docs_reference_links(locale, catalog))
                expected_hrefs.update(_markdown_targets(summary))
            missing_hrefs = {href for href in expected_hrefs if href not in html}
            if missing_hrefs:
                raise LearningPathI18nError(f"Missing links in {page}: {sorted(missing_hrefs)}")

            switch_position = _first_href_position(html, set(_markdown_targets(switch)))
            navigation_position = _first_href_position(html, navigation_hrefs)
            title_position = _first_position(html, title)
            if switch_position < 0 or title_position < 0 or not title_position < switch_position:
                raise LearningPathI18nError(f"Language switch is not below the title in {page}")
            if navigation_hrefs and (navigation_position < 0 or navigation_position <= switch_position):
                raise LearningPathI18nError(f"Previous/next navigation is not below the language switch in {page}")

            if not summary and "summary" in catalog.messages:
                summary = "\n".join(catalog.messages["summary"][locale])
            summary_heading = summary.splitlines()[0] if summary else ""
            navigation_heading = navigation.splitlines()[0]
            if summary_heading == navigation_heading:
                raise LearningPathI18nError(f"Summary and navigation headings are duplicated in {page}")
            if summary_heading and _first_position(html, summary_heading.lstrip("# ")) < 0:
                raise LearningPathI18nError(f"Missing summary heading in {page}")
            if _first_position(html, navigation_heading.lstrip("# ")) < 0:
                raise LearningPathI18nError(f"Missing navigation heading in {page}")

            if locale == "en":
                for relation in ("previous", "next"):
                    target_id = getattr(lesson, relation)
                    if target_id is None or "en" in lessons_by_id[target_id].locales:
                        continue
                    japanese_only = catalog.text("navigation.japanese_only")
                    if _first_position(html, japanese_only) < 0:
                        raise LearningPathI18nError(
                            f"Missing Japanese-only fallback label in {page} for {relation} link"
                        )
                if (
                    lesson.lesson_id == "06_reusable_pipeline_recipes"
                    and _first_position(html, catalog.text("navigation.japanese_only")) < 0
                ):
                    raise LearningPathI18nError(f"Missing Japanese-only docs-link note in {page}")

            exported_hrefs = re.findall(r'href=\\?"([^"\\\s]*learning-path[^"\\\s]*)\\?"', html)
            for href in exported_hrefs:
                target = _normalized_site_target(page.relative_to(site_root), href)
                if "en/learning-path" in target and not (site_root / target).exists():
                    raise LearningPathI18nError(f"English link points to a missing page: {page} -> {href}")

        if "ja" in pages and "en" in pages:
            japanese_page, japanese_html = pages["ja"]
            english_page, english_html = pages["en"]
            _validate_shared_visible_code(japanese_html, japanese_page, english_html, english_page)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, help="validate exported HTML below this directory")
    parser.add_argument("--all", action="store_true", help="validate every manifest export, not only the PoC pages")
    parser.add_argument(
        "--translated",
        action="store_true",
        help="validate all translated lesson exports present below the site directory",
    )
    args = parser.parse_args(argv)
    if args.all and args.translated:
        parser.error("--all and --translated are mutually exclusive")
    validate_catalogs()
    print("Learning Path manifest and translation catalogs are valid.")
    if args.site is not None:
        validate_exported_site(args.site.resolve(), validate_all=args.all, validate_translated=args.translated)
        scope = "all planned" if args.all else "translated" if args.translated else "PoC"
        print(f"Learning Path {scope} HTML is valid below {args.site.resolve()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
