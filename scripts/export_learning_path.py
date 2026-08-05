"""Plan and export Learning Path HTML through one manifest-driven command."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from .learning_path_i18n import (
        REPO_ROOT,
        SUPPORTED_LOCALES,
        ExportPlanItem,
        LearningPathI18nError,
        Lesson,
        build_export_plan,
        load_manifest,
        poc_lessons,
        translated_lessons,
    )
except ImportError:  # Running this file directly keeps the existing python scripts/foo.py workflow.
    from learning_path_i18n import (
        REPO_ROOT,
        SUPPORTED_LOCALES,
        ExportPlanItem,
        LearningPathI18nError,
        Lesson,
        build_export_plan,
        load_manifest,
        poc_lessons,
        translated_lessons,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="export every lesson in every available locale")
    selection.add_argument("--poc", action="store_true", help="export the two manifest entries marked for the PoC")
    selection.add_argument(
        "--translated",
        action="store_true",
        help="export every lesson with an English translation catalog in Japanese and English",
    )
    selection.add_argument("--lesson", help="export one lesson by manifest id")
    parser.add_argument("--locale", choices=SUPPORTED_LOCALES, help="export only this locale")
    parser.add_argument("--output", type=Path, default=Path("docs/site"), help="site output directory")
    parser.add_argument("--jobs", type=int, default=1, help="number of marimo exports to run concurrently")
    parser.add_argument("--dry-run", action="store_true", help="print the deterministic plan without exporting")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    return args


def select_lessons(args: argparse.Namespace) -> tuple[Lesson, ...]:
    """Select lessons without changing their manifest order."""

    lessons = load_manifest()
    if args.poc:
        selected = poc_lessons()
    elif args.translated:
        selected = translated_lessons(lessons)
        if not selected:
            raise LearningPathI18nError("No translated lessons are available")
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


def export_plan(args: argparse.Namespace) -> tuple[ExportPlanItem, ...]:
    """Return a deterministic export plan for parsed CLI arguments."""

    return build_export_plan(select_lessons(args), args.output.resolve(), locale=args.locale)


def marimo_export_command(target: ExportPlanItem) -> tuple[str, ...]:
    """Build the exact public marimo command for one plan item."""

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
    ]
    if target.pass_locale:
        command.extend(("--", "--locale", target.locale))
    return tuple(command)


def _export(target: ExportPlanItem) -> None:
    target.output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        marimo_export_command(target),
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
    targets = export_plan(args)
    print(f"Exporting {len(targets)} Learning Path page(s) to {args.output.resolve()}")

    if args.dry_run:
        for target in targets:
            print(f"  {target.lesson.lesson_id} [{target.locale}] -> {target.output_path}")
            print(f"    {shlex.join(marimo_export_command(target))}")
        return 0

    if not targets:
        return 0

    futures = {}
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(targets))) as executor:
        for target in targets:
            futures[target] = executor.submit(_export, target)

        failures: list[str] = []
        for target in targets:
            try:
                futures[target].result()
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
