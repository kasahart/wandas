"""Export the Learning Path through one manifest-driven command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

try:
    from .learning_path_i18n import REPO_ROOT, LearningPathI18nError, Lesson, load_manifest, output_path, poc_lessons
except ImportError:  # Running this file directly keeps the existing `python scripts/foo.py` workflow.
    from learning_path_i18n import REPO_ROOT, LearningPathI18nError, Lesson, load_manifest, output_path, poc_lessons


@dataclass(frozen=True)
class ExportTarget:
    lesson: Lesson
    locale: str
    output_root: Path

    @property
    def output_path(self) -> Path:
        return output_path(self.output_root, self.lesson, self.locale)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="export every lesson in every available locale")
    selection.add_argument("--poc", action="store_true", help="export the two manifest entries marked for the PoC")
    selection.add_argument("--lesson", help="export one lesson by manifest id")
    parser.add_argument("--locale", choices=("ja", "en"), help="export only this locale")
    parser.add_argument("--output", type=Path, default=Path("docs/site"), help="site output directory")
    parser.add_argument("--jobs", type=int, default=2, help="number of marimo exports to run concurrently")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    return args


def _select_lessons(args: argparse.Namespace) -> tuple[Lesson, ...]:
    lessons = load_manifest()
    if args.poc:
        selected = poc_lessons()
    elif args.lesson:
        selected = tuple(lesson for lesson in lessons if lesson.lesson_id == args.lesson)
        if not selected:
            raise LearningPathI18nError(f"Unknown lesson id: {args.lesson}")
    else:
        selected = lessons

    if args.locale is not None:
        selected = tuple(lesson for lesson in selected if args.locale in lesson.locales)
        if not selected:
            raise LearningPathI18nError(f"No selected lesson is available in locale {args.locale!r}")
    return selected


def _targets(args: argparse.Namespace) -> tuple[ExportTarget, ...]:
    output_root = args.output.resolve()
    targets = []
    for lesson in _select_lessons(args):
        locales = (args.locale,) if args.locale is not None else lesson.locales
        for locale in locales:
            targets.append(ExportTarget(lesson=lesson, locale=locale, output_root=output_root))
    return tuple(targets)


def _export(target: ExportTarget) -> None:
    target.output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "marimo",
        "export",
        "html",
        str(target.lesson.source_path),
        "-o",
        str(target.output_path),
        "-f",
        "--",
        "--locale",
        target.locale,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        details = (completed.stdout + completed.stderr).strip()
        raise RuntimeError(
            f"marimo export failed for {target.lesson.lesson_id} ({target.locale}) "
            f"with exit code {completed.returncode}:\n{details}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    targets = _targets(args)
    print(f"Exporting {len(targets)} Learning Path page(s) to {args.output.resolve()}")

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(targets))) as executor:
        futures = {executor.submit(_export, target): target for target in targets}
        for future in as_completed(futures):
            target = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - preserve lesson and locale context for CI.
                failures.append(str(exc))
            else:
                print(f"  {target.lesson.lesson_id} [{target.locale}] -> {target.output_path}")

    if failures:
        print("Learning Path export failed:", file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
