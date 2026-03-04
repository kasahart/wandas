from .wav_io import write_wav
from .wdf_io import load, save


def read_wav(path: str, **kwargs):  # type: ignore[no-untyped-def]
    """Read a WAV file and return a ChannelFrame.

    .. deprecated::
        Use :meth:`wandas.ChannelFrame.read_wav` instead.
        This alias is kept for backwards compatibility.
    """
    import warnings

    from wandas.frames.channel import ChannelFrame

    warnings.warn(
        "wandas.io.read_wav is deprecated. Use ChannelFrame.read_wav instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ChannelFrame.read_wav(path, **kwargs)


__all__ = ["write_wav", "load", "save", "read_wav"]
