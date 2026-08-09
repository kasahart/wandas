"""Completeness contracts for built-in Recipe declaration owners."""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Iterable
from typing import Any

import wandas.frames as public_frames
from wandas.core.base_frame import BaseFrame
from wandas.pipeline.builtins import builtin_recipe_operations
from wandas.pipeline.registry import RecipeOperation

_EXPECTED_BUILTIN_RECIPE_ORDER = (
    ("wandas.frame.with_source_time_offset", 1),
    ("wandas.channel.rename_channels", 1),
    ("wandas.frame.get_channel", 1),
    ("wandas.frame.index", 1),
    ("wandas.frame.astype", 1),
    ("wandas.operator.add", 1),
    ("wandas.operator.subtract", 1),
    ("wandas.operator.multiply", 1),
    ("wandas.operator.divide", 1),
    ("wandas.operator.power", 1),
    ("wandas.operator.reverse_add", 1),
    ("wandas.operator.reverse_subtract", 1),
    ("wandas.operator.reverse_multiply", 1),
    ("wandas.operator.reverse_divide", 1),
    ("wandas.operator.reverse_power", 1),
    ("wandas.custom.apply", 1),
    ("wandas.audio.highpass_filter", 1),
    ("wandas.audio.lowpass_filter", 1),
    ("wandas.audio.bandpass_filter", 1),
    ("wandas.audio.normalize", 1),
    ("wandas.audio.remove_dc", 1),
    ("wandas.audio.a_weighting", 1),
    ("wandas.audio.abs", 1),
    ("wandas.audio.power", 1),
    ("wandas.audio.sum", 1),
    ("wandas.audio.mean", 1),
    ("wandas.audio.trim", 1),
    ("wandas.frame.time_slice", 1),
    ("wandas.audio.fix_length", 1),
    ("wandas.audio.rms_trend", 2),
    ("wandas.audio.rms_trend", 1),
    ("wandas.audio.sound_level", 2),
    ("wandas.audio.sound_level", 1),
    ("wandas.audio.channel_difference", 1),
    ("wandas.audio.resampling", 1),
    ("wandas.audio.hpss_harmonic", 1),
    ("wandas.audio.hpss_percussive", 1),
    ("wandas.audio.loudness_zwtv", 1),
    ("wandas.audio.roughness_dw", 1),
    ("wandas.audio.roughness_dw_spec", 1),
    ("wandas.audio.fade", 1),
    ("wandas.audio.sharpness_din", 1),
    ("wandas.audio.cepstrum", 2),
    ("wandas.audio.cepstrum", 1),
    ("wandas.audio.fft", 2),
    ("wandas.audio.fft", 1),
    ("wandas.audio.welch", 2),
    ("wandas.audio.welch", 1),
    ("wandas.audio.noct_spectrum", 1),
    ("wandas.audio.stft", 1),
    ("wandas.audio.coherence", 2),
    ("wandas.audio.coherence", 1),
    ("wandas.audio.csd", 2),
    ("wandas.audio.csd", 1),
    ("wandas.audio.transfer_function", 2),
    ("wandas.audio.transfer_function", 1),
    ("wandas.channel.with_calibration", 1),
    ("wandas.audio.mix", 1),
    ("wandas.channel.add_channel", 2),
    ("wandas.channel.concat_frame", 1),
    ("wandas.channel.remove_channel", 1),
    ("wandas.frame.select_pair", 1),
    ("wandas.cepstral.lifter", 1),
    ("wandas.cepstral.to_spectral_envelope", 1),
    ("wandas.cepstrogram.lifter", 1),
    ("wandas.cepstrogram.to_spectral_envelope", 1),
    ("wandas.spectral.ifft", 2),
    ("wandas.spectral.ifft", 1),
    ("wandas.spectral.noct_synthesis", 2),
    ("wandas.spectral.noct_synthesis", 1),
    ("wandas.spectrogram.cepstrum", 1),
    ("wandas.spectrogram.absolute", 1),
    ("wandas.spectrogram.get_frame_at", 1),
    ("wandas.spectrogram.to_channel_frame", 1),
)


def _direct_recipe_operations(owner: type[Any]) -> tuple[RecipeOperation, ...]:
    return tuple(
        definition
        for member in vars(owner).values()
        if isinstance((definition := getattr(member, "__wandas_recipe_operation__", None)), RecipeOperation)
    )


def _reachable_recipe_owners() -> tuple[type[Any], ...]:
    owners: list[type[Any]] = []
    seen: set[type[Any]] = set()
    for export_name in public_frames.__all__:
        frame_type = getattr(public_frames, export_name)
        if not inspect.isclass(frame_type) or not issubclass(frame_type, BaseFrame) or inspect.isabstract(frame_type):
            continue
        for owner in frame_type.__mro__:
            if owner not in seen and _direct_recipe_operations(owner):
                seen.add(owner)
                owners.append(owner)
    return tuple(owners)


def _operation_key(operation: RecipeOperation) -> tuple[str, int]:
    return operation.operation_id, operation.version


def _expanded(counter: Counter[tuple[str, int]]) -> list[str]:
    return [f"{operation_id}@v{version}" for operation_id, version in sorted(counter.elements())]


def _definitions(counter: Counter[RecipeOperation]) -> list[str]:
    return sorted(f"{operation.operation_id}@v{operation.version}" for operation in counter.elements())


def _owner_operations(owners: Iterable[type[Any]]) -> tuple[RecipeOperation, ...]:
    return tuple(operation for owner in owners for operation in _direct_recipe_operations(owner))


def test_builtin_recipe_operations_cover_public_frame_mro_owners_exactly_once() -> None:
    owners = _reachable_recipe_owners()
    reachable = _owner_operations(owners)
    actual = builtin_recipe_operations()
    reachable_definitions = Counter(reachable)
    actual_definitions = Counter(actual)
    missing = reachable_definitions - actual_definitions
    unexpected = actual_definitions - reachable_definitions
    actual_keys = Counter(map(_operation_key, actual))
    duplicate_definitions = Counter(
        {definition: count for definition, count in actual_definitions.items() if count > 1}
    )
    duplicate_keys = Counter({key: count for key, count in actual_keys.items() if count > 1})

    assert not (missing or unexpected or duplicate_definitions or duplicate_keys), (
        "Built-in Recipe declaration drift:\n"
        f"  missing={_definitions(missing)}\n"
        f"  unexpected={_definitions(unexpected)}\n"
        f"  duplicate definitions={_definitions(duplicate_definitions)}\n"
        f"  duplicate IDs={_expanded(duplicate_keys)}\n"
        f"  reachable owners={[owner.__module__ + '.' + owner.__qualname__ for owner in owners]}"
    )


def test_builtin_recipe_operation_order_matches_explicit_contract() -> None:
    actual = tuple(map(_operation_key, builtin_recipe_operations()))

    assert actual == _EXPECTED_BUILTIN_RECIPE_ORDER
