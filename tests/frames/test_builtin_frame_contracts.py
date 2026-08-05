"""Completeness contracts for explicit built-in Frame boundaries."""

from __future__ import annotations

import inspect
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
