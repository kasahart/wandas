"""Frame classes for wandas."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wandas.frames.channel import ChannelFrame
from wandas.frames.roughness import RoughnessFrame

if TYPE_CHECKING:
    from wandas.frames.cepstral import CepstralFrame

__all__ = ["ChannelFrame", "CepstralFrame", "RoughnessFrame"]


def __getattr__(name: str) -> Any:
    if name == "CepstralFrame":
        from wandas.frames.cepstral import CepstralFrame

        return CepstralFrame
    raise AttributeError(f"module 'wandas.frames' has no attribute {name!r}")
