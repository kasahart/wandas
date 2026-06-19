"""Frame classes for wandas."""

from wandas.frames.channel import ChannelFrame
from wandas.frames.noct import NOctFrame
from wandas.frames.roughness import RoughnessFrame
from wandas.frames.spectral import SpectralFrame
from wandas.frames.spectrogram import SpectrogramFrame

__all__ = ["ChannelFrame", "SpectralFrame", "SpectrogramFrame", "NOctFrame", "RoughnessFrame"]
