"""Built-in Recipe declarations collected without mutable registration."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from wandas.pipeline.registry import RecipeOperation


def _declared_operations(owner: type[Any]) -> Iterable[RecipeOperation]:
    """Yield Recipe declarations attached directly to one public API owner."""
    for member in vars(owner).values():
        definition = getattr(member, "__wandas_recipe_operation__", None)
        if isinstance(definition, RecipeOperation):
            yield definition


def builtin_recipe_operations() -> tuple[RecipeOperation, ...]:
    """Return all explicitly declared built-in public Frame operations."""
    from wandas.core.base_frame import BaseFrame
    from wandas.frames.cepstral import CepstralFrame
    from wandas.frames.channel import ChannelFrame
    from wandas.frames.mixins.channel_processing_mixin import ChannelProcessingMixin
    from wandas.frames.mixins.channel_transform_mixin import ChannelTransformMixin
    from wandas.frames.spectral import SpectralFrame
    from wandas.frames.spectrogram import SpectrogramFrame

    owners = (
        BaseFrame,
        ChannelProcessingMixin,
        ChannelTransformMixin,
        ChannelFrame,
        CepstralFrame,
        SpectralFrame,
        SpectrogramFrame,
    )
    return tuple(operation for owner in owners for operation in _declared_operations(owner))
