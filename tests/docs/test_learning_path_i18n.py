import ast
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

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
    translated_lessons,
    validate_catalogs,
)
from scripts.validate_learning_path_i18n import (
    _exported_notebook_cells,
    _validate_shared_visible_code,
    validate_exported_site,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _app_cell_decorator(decorator: ast.expr) -> ast.Attribute | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name):
        return None
    if target.value.id != "app" or target.attr != "cell":
        return None
    return target


def _source_cell_code(source_path: Path) -> tuple[tuple[str, bool], ...]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    cells: list[tuple[str, bool]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(_app_cell_decorator(decorator) is not None for decorator in node.decorator_list):
            continue
        hidden = any(
            isinstance(decorator, ast.Call)
            and any(
                keyword.arg == "hide_code" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                for keyword in decorator.keywords
            )
            for decorator in node.decorator_list
        )
        cells.append((ast.unparse(node), hidden))
    return tuple(cells)


def _exported_cell_code(cell: dict[str, object]) -> tuple[str, bool]:
    code = cell.get("code")
    config = cell.get("config")
    assert isinstance(code, str)
    assert isinstance(config, Mapping)
    config_mapping = cast(Mapping[str, object], config)
    return code, config_mapping.get("hide_code") is True


def _exported_notebook_html(*cells: dict[str, object]) -> str:
    return f'<script>"notebook": {json.dumps({"cells": list(cells)})}</script>'


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


def test_translated_lessons_are_manifest_driven() -> None:
    lessons = load_manifest()

    assert [lesson.lesson_id for lesson in translated_lessons(lessons)] == [
        lesson.lesson_id for lesson in lessons if "en" in lesson.locales and lesson.catalog is not None
    ]
    assert all("en" in lesson.locales and lesson.catalog is not None for lesson in translated_lessons(lessons))


def test_shared_visible_code_ignores_hidden_locale_implementation() -> None:
    japanese = _exported_notebook_html(
        {"code": "locale = 'ja'", "config": {"hide_code": True}},
        {"code": "signal = wd.generate_sin()", "config": {"hide_code": False}},
    )
    english = _exported_notebook_html(
        {"code": "locale = 'en'", "config": {"hide_code": True}},
        {"code": "signal = wd.generate_sin()", "config": {"hide_code": False}},
    )

    _validate_shared_visible_code(japanese, Path("ja.html"), english, Path("en.html"))

    changed_english = _exported_notebook_html(
        {"code": "locale = 'en'", "config": {"hide_code": True}},
        {"code": "signal = wd.generate_cos()", "config": {"hide_code": False}},
    )
    with pytest.raises(LearningPathI18nError, match="Visible cell code differs"):
        _validate_shared_visible_code(japanese, Path("ja.html"), changed_english, Path("en.html"))


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

    en_plan = export_plan(_parse_args(["--lesson", "01_getting_started", "--locale", "en", "--dry-run"]))
    assert [(target.lesson.lesson_id, target.locale) for target in en_plan] == [("01_getting_started", "en")]

    ja_plan = export_plan(_parse_args(["--lesson", "01_getting_started", "--locale", "ja", "--dry-run"]))
    assert [(target.lesson.lesson_id, target.locale) for target in ja_plan] == [("01_getting_started", "ja")]

    all_en_plan = export_plan(_parse_args(["--all", "--locale", "en", "--dry-run"]))
    assert [(target.lesson.lesson_id, target.locale) for target in all_en_plan] == [
        (lesson.lesson_id, "en") for lesson in translated_lessons(lessons)
    ]

    all_ja_plan = export_plan(_parse_args(["--all", "--locale", "ja", "--dry-run"]))
    assert [(target.lesson.lesson_id, target.locale) for target in all_ja_plan] == [
        (lesson.lesson_id, "ja") for lesson in lessons
    ]

    legacy_lesson = next(lesson for lesson in lessons if "en" not in lesson.locales)
    with pytest.raises(LearningPathI18nError, match="No selected lesson is available"):
        export_plan(_parse_args(["--lesson", legacy_lesson.lesson_id, "--locale", "en", "--dry-run"]))

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


def test_learning_path_06_public_imports_are_visible_and_i18n_setup_is_hidden() -> None:
    source_cells = _source_cell_code(REPO_ROOT / "learning-path/06_reusable_pipeline_recipes.py")
    visible_source = "\n".join(code for code, hidden in source_cells if not hidden)
    hidden_source = "\n".join(code for code, hidden in source_cells if hidden)

    for statement in (
        "import json",
        "import numpy as np",
        "import wandas as wd",
        "from wandas import pipeline as pipeline_api",
    ):
        assert statement in visible_source
    assert "from scripts.learning_path_i18n import" in hidden_source
    assert "scripts.learning_path_i18n" not in visible_source
    assert "load_catalog" not in visible_source
    assert "locale_from_argv" not in visible_source
    assert re.search(r"\bt\s*\(\s*['\"]", visible_source) is None


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

    for locale in ("ja", "en"):
        page = (
            site_root
            / ("en/learning-path" if locale == "en" else "learning-path")
            / ("06_reusable_pipeline_recipes.html")
        )
        cells = [_exported_cell_code(cell) for cell in _exported_notebook_cells(page.read_text(encoding="utf-8"), page)]
        visible_html = "\n".join(code for code, hidden in cells if not hidden)
        hidden_html = "\n".join(code for code, hidden in cells if hidden)
        for statement in (
            "import json",
            "import numpy as np",
            "import wandas as wd",
            "from wandas import pipeline as pipeline_api",
        ):
            assert statement in visible_html
        assert "from scripts.learning_path_i18n import" in hidden_html
        assert "scripts.learning_path_i18n" not in visible_html
        assert "load_catalog" not in visible_html
        assert "locale_from_argv" not in visible_html
        assert re.search(r"\bt\s*\(\s*['\"]", visible_html) is None
