"""Validate Learning Path translation catalogs and PoC HTML exports."""

from __future__ import annotations

import argparse
import json
import posixpath
import re
from pathlib import Path

try:
    from .learning_path_i18n import (
        LearningPathI18nError,
        _markdown_targets,
        docs_relative_href,
        lesson_by_id,
        load_catalog,
        navigation_markdown,
        output_path,
        poc_lessons,
        validate_catalogs,
    )
except ImportError:  # Running this file directly keeps the existing `python scripts/foo.py` workflow.
    from learning_path_i18n import (
        LearningPathI18nError,
        _markdown_targets,
        docs_relative_href,
        lesson_by_id,
        load_catalog,
        navigation_markdown,
        output_path,
        poc_lessons,
        validate_catalogs,
    )

_CODE_MARKERS = {
    "01_getting_started": ("wd.generate_sin", "combined_signal.fft().plot"),
    "06_reusable_pipeline_recipes": ("RecipePlan.from_frame", "loaded_recipe.apply"),
}


def _normalized_site_target(current: Path, href: str) -> str:
    return posixpath.normpath((current.parent / href).as_posix())


def _serialized_text(value: str) -> tuple[str, str]:
    """Return the literal and JSON-escaped forms used by a marimo export."""

    return value, json.dumps(value, ensure_ascii=True)[1:-1]


def validate_exported_site(site_root: Path) -> None:
    """Check the generated PoC HTML without using snapshots or pixel comparison."""

    for lesson in poc_lessons():
        for locale in lesson.locales:
            page = output_path(site_root, lesson, locale)
            if not page.exists():
                raise LearningPathI18nError(f"Missing exported page: {page}")
            html = page.read_text(encoding="utf-8")
            if "Traceback (most recent call last)" in html or "ModuleNotFoundError" in html:
                raise LearningPathI18nError(f"Python error text found in exported page: {page}")

            title = load_catalog(lesson.lesson_id, locale).text("title")
            if not any(candidate in html for candidate in _serialized_text(title)):
                raise LearningPathI18nError(f"Missing {locale} title in {page}: {title}")
            for marker in _CODE_MARKERS[lesson.lesson_id]:
                if marker not in html:
                    raise LearningPathI18nError(f"Missing shared code marker {marker!r} in {page}")

            expected_hrefs = set(_markdown_targets(navigation_markdown(lesson.lesson_id, locale)))
            if locale == "en":
                for relation in ("previous", "next"):
                    target_id = getattr(lesson, relation)
                    if target_id is None or "en" in lesson_by_id(target_id).locales:
                        continue
                    japanese_only = load_catalog(lesson.lesson_id, locale).text("navigation.japanese_only")
                    if not any(candidate in html for candidate in _serialized_text(japanese_only)):
                        raise LearningPathI18nError(
                            f"Missing Japanese-only fallback label in {page} for {relation} link"
                        )
            if lesson.lesson_id == "06_reusable_pipeline_recipes":
                summary = load_catalog(lesson.lesson_id, locale).text(
                    "summary",
                    how_to_href=docs_relative_href(locale, "how-to/pipeline-recipes/"),
                    api_href=docs_relative_href(locale, "api/pipeline/"),
                )
                expected_hrefs.update(_markdown_targets(summary))
            missing_hrefs = {href for href in expected_hrefs if href not in html}
            if missing_hrefs:
                raise LearningPathI18nError(f"Missing navigation links in {page}: {sorted(missing_hrefs)}")

            exported_hrefs = re.findall(r'href=\\?"([^"\\\s]*learning-path[^"\\\s]*)\\?"', html)
            for href in exported_hrefs:
                target = _normalized_site_target(page.relative_to(site_root), href)
                if "en/learning-path" in target:
                    target_path = site_root / target
                    if not target_path.exists():
                        raise LearningPathI18nError(f"English link points to a missing page: {page} -> {href}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, help="validate exported PoC HTML below this directory")
    args = parser.parse_args(argv)
    validate_catalogs()
    print("Learning Path translation catalogs are valid.")
    if args.site is not None:
        validate_exported_site(args.site.resolve())
        print(f"Learning Path PoC HTML is valid below {args.site.resolve()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
