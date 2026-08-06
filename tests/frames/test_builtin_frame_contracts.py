"""Completeness contracts for explicit built-in Frame boundaries."""

from __future__ import annotations

import inspect
from collections import Counter
from typing import Any

import wandas.frames as public_frames
from tests.builtin_frame_cases import BUILTIN_FRAME_CASES
from wandas.core.base_frame import BaseFrame
from wandas.io import wdf_frames


def _public_concrete_frame_types() -> set[type[BaseFrame[Any]]]:
    return {
        value
        for name in public_frames.__all__
        if inspect.isclass(value := getattr(public_frames, name))
        and issubclass(value, BaseFrame)
        and not inspect.isabstract(value)
    }


def _type_diff(
    expected: set[type[BaseFrame[Any]]],
    actual: set[type[BaseFrame[Any]]],
) -> tuple[list[str], list[str]]:
    return (
        sorted(frame_type.__name__ for frame_type in expected - actual),
        sorted(frame_type.__name__ for frame_type in actual - expected),
    )


def _qualified_type_name(frame_type: type[object]) -> str:
    return f"{frame_type.__module__}.{frame_type.__qualname__}"


def test_shared_builtin_frame_cases_have_unique_exact_types_and_ids() -> None:
    type_counts = Counter(case.frame_type for case in BUILTIN_FRAME_CASES)
    id_counts = Counter(case.id for case in BUILTIN_FRAME_CASES)
    duplicate_types = sorted(
        f"{_qualified_type_name(frame_type)} ({count} cases)" for frame_type, count in type_counts.items() if count > 1
    )
    duplicate_ids = sorted(f"{case_id!r} ({count} cases)" for case_id, count in id_counts.items() if count > 1)

    assert not duplicate_types and not duplicate_ids, (
        f"Duplicate shared built-in Frame cases:\nexact types={duplicate_types}\ncase IDs={duplicate_ids}"
    )


def test_shared_builtin_frame_factories_return_declared_exact_types() -> None:
    mismatches = []
    for case in BUILTIN_FRAME_CASES:
        actual_type = type(case.factory())
        if actual_type is not case.frame_type:
            mismatches.append(
                f"{case.id}: declared={_qualified_type_name(case.frame_type)}, "
                f"returned={_qualified_type_name(actual_type)}"
            )

    assert not mismatches, "Built-in Frame factory type mismatch:\n" + "\n".join(mismatches)


def test_wdf_codec_registry_has_unique_exact_frame_types() -> None:
    type_counts = Counter(codec.frame_type for codec in wdf_frames._codecs())
    duplicate_types = sorted(
        f"{_qualified_type_name(frame_type)} ({count} codecs)" for frame_type, count in type_counts.items() if count > 1
    )

    assert not duplicate_types, "Duplicate WDF codecs for exact Frame types:\n" + "\n".join(duplicate_types)


def test_public_frames_shared_cases_and_wdf_codecs_cover_identical_exact_types() -> None:
    public_types = _public_concrete_frame_types()
    shared_case_types = {case.frame_type for case in BUILTIN_FRAME_CASES}
    wdf_codec_types = {codec.frame_type for codec in wdf_frames._codecs()}

    failures = []
    for boundary, actual in (("shared Frame cases", shared_case_types), ("WDF codecs", wdf_codec_types)):
        missing, unexpected = _type_diff(public_types, actual)
        if missing or unexpected:
            failures.append(f"{boundary}: missing={missing}, unexpected={unexpected}")

    assert not failures, "Built-in Frame type drift:\n" + "\n".join(failures)
