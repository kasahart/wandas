import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.learning_path_i18n import (
    LearningPathI18nError,
    locale_from_argv,
    poc_lessons,
    validate_catalogs,
)
from scripts.validate_learning_path_i18n import validate_exported_site

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_learning_path_translation_catalog_contract() -> None:
    validate_catalogs()

    source_paths = [lesson.source_path for lesson in poc_lessons()]
    assert len(source_paths) == len(set(source_paths))
    for source_path in source_paths:
        assert not source_path.with_name(f"{source_path.stem}.ja.py").exists()
        assert not source_path.with_name(f"{source_path.stem}.en.py").exists()

    with pytest.raises(SystemExit):
        locale_from_argv(["--locale", "fr"])


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
