import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.export_learning_path import _parse_args, export_plan, marimo_export_command
from scripts.learning_path_i18n import (
    COMMON_CATALOG_PATH,
    COMMON_KEYS,
    LearningPathI18nError,
    TranslationCatalog,
    build_export_plan,
    load_catalog,
    load_manifest,
    locale_from_argv,
    tracked_numbered_lesson_sources,
    validate_catalogs,
)
from scripts.validate_learning_path_i18n import validate_exported_site

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_learning_path_manifest_matches_tracked_sources(tmp_path: Path) -> None:
    lessons = load_manifest()
    manifest_sources = {lesson.source for lesson in lessons}
    tracked_sources = set(tracked_numbered_lesson_sources())

    assert manifest_sources == tracked_sources
    assert len({lesson.lesson_id for lesson in lessons}) == len(lessons)
    assert len(manifest_sources) == len(lessons)
    assert all(lesson.lesson_id == Path(lesson.source).stem for lesson in lessons)
    assert all(lesson.source_path.exists() for lesson in lessons)
    manifest = json.loads((REPO_ROOT / "learning-path/manifest.json").read_text(encoding="utf-8"))
    assert all("previous" not in lesson and "next" not in lesson for lesson in manifest["lessons"])

    for index, lesson in enumerate(lessons):
        assert lesson.previous == (lessons[index - 1].lesson_id if index else None)
        assert lesson.next == (lessons[index + 1].lesson_id if index + 1 < len(lessons) else None)

    plan = build_export_plan(lessons, tmp_path)
    output_paths = [target.output_path for target in plan]
    assert len(output_paths) == len(set(output_paths))
    assert [target.lesson.lesson_id for target in plan if target.locale == "ja"] == [
        lesson.lesson_id for lesson in lessons
    ]


def test_learning_path_translation_catalog_contract() -> None:
    validate_catalogs()

    common = json.loads(COMMON_CATALOG_PATH.read_text(encoding="utf-8"))
    assert set(common["messages"]) == COMMON_KEYS
    for lesson in load_manifest():
        if lesson.catalog_path is None:
            continue
        catalog = json.loads(lesson.catalog_path.read_text(encoding="utf-8"))
        assert not COMMON_KEYS.intersection(catalog["messages"])

    for lesson in load_manifest():
        source_path = lesson.source_path
        assert not source_path.with_name(f"{source_path.stem}.ja.py").exists()
        assert not source_path.with_name(f"{source_path.stem}.en.py").exists()

    with pytest.raises(SystemExit):
        locale_from_argv(["--locale", "fr"])
    with pytest.raises(LearningPathI18nError, match="Unsupported locale"):
        load_catalog("01_getting_started", "fr")


def test_translation_catalog_uses_explicit_placeholders_and_literal_braces() -> None:
    catalog = TranslationCatalog(
        "placeholder_test",
        {
            "literal": {"ja": ('config = {"sampling_rate": 48000}',)},
            "message": {"ja": ("Hello [[name]]",)},
            "invalid": {"ja": ("Value [[name.attr]]",)},
        },
    )

    assert catalog.text("literal") == 'config = {"sampling_rate": 48000}'
    assert catalog.text("message", name="Wandas") == "Hello Wandas"
    with pytest.raises(LearningPathI18nError, match="placeholder_test:message:ja"):
        catalog.text("message")
    with pytest.raises(LearningPathI18nError, match="placeholder_test:message:ja"):
        catalog.text("message", name="Wandas", unused="value")
    with pytest.raises(LearningPathI18nError, match="placeholder_test:invalid:ja"):
        catalog.text("invalid", name="Wandas")
    with pytest.raises(LearningPathI18nError, match="placeholder_test:missing:ja"):
        catalog.text("missing")


def test_learning_path_export_plan_is_deterministic_and_preserves_legacy_commands() -> None:
    args = _parse_args(["--all", "--dry-run"])
    plan = export_plan(args)
    assert plan == export_plan(args)

    lessons = load_manifest()
    expected = [(lesson.lesson_id, locale) for lesson in lessons for locale in lesson.locales]
    assert [(target.lesson.lesson_id, target.locale) for target in plan] == expected
    assert len({target.output_path for target in plan}) == len(plan)

    for target in plan:
        command = marimo_export_command(target)
        if target.lesson.catalog_path is None:
            assert "--" not in command
            assert target.locale == "ja"
        else:
            assert command[-3:] == ("--", "--locale", target.locale)

    poc_plan = export_plan(_parse_args(["--poc", "--jobs", "2"]))
    assert [(target.lesson.lesson_id, target.locale) for target in poc_plan] == [
        ("01_getting_started", "ja"),
        ("01_getting_started", "en"),
        ("06_reusable_pipeline_recipes", "ja"),
        ("06_reusable_pipeline_recipes", "en"),
    ]


def test_learning_path_poc_exports_are_static_and_cross_linked(tmp_path: Path) -> None:
    site_root = tmp_path / "site"
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/export_learning_path.py",
            "--poc",
            "--output",
            str(site_root),
            "--jobs",
            "2",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    try:
        validate_exported_site(site_root)
    except LearningPathI18nError as exc:
        pytest.fail(str(exc))
