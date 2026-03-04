from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from .wav_io import write_wav
from .wdf_io import load, save

if TYPE_CHECKING:
    from wandas.frames.channel import ChannelFrame


def read_wav(
    path: str | Path | bytes | bytearray | memoryview | BinaryIO,
    **kwargs: Any,
) -> ChannelFrame:
    """Read a WAV file and return a ChannelFrame.

    .. deprecated::
        Use :meth:`wandas.ChannelFrame.read_wav` instead.
        This alias is kept for backwards compatibility.
    """
    import warnings

    from wandas.frames.channel import ChannelFrame as _ChannelFrame

    warnings.warn(
        "wandas.io.read_wav is deprecated. Use ChannelFrame.read_wav instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _ChannelFrame.read_wav(path, **kwargs)


__all__ = ["write_wav", "load", "save", "read_wav"]
