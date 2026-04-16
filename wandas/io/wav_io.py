# wandas/io/wav_io.py
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import wavfile as _scipy_wavfile

try:
    import soundfile as sf

    _SOUNDFILE_AVAILABLE = True
except OSError:
    _SOUNDFILE_AVAILABLE = False

if TYPE_CHECKING:
    from ..frames.channel import ChannelFrame

logger = logging.getLogger(__name__)


def write_wav(filename: str, target: "ChannelFrame", format: str | None = None) -> None:
    """
    Write a ChannelFrame object to a WAV file.

    Parameters
    ----------
    filename : str
        Path to the WAV file.
    target : ChannelFrame
        ChannelFrame object containing the data to write.
    format : str, optional
        File format. If None, determined from file extension.

    Raises
    ------
    ValueError
        If target is not a ChannelFrame object.
    """
    from wandas.frames.channel import ChannelFrame

    if not isinstance(target, ChannelFrame):
        raise ValueError("target must be a ChannelFrame object.")

    logger.debug(f"Saving audio data to file: {filename} (will compute now)")
    data = target.compute()
    data = data.T
    if data.shape[1] == 1:
        data = data.squeeze(axis=1)

    if not _SOUNDFILE_AVAILABLE:
        # Fallback: scipy.io.wavfile (available in Pyodide)
        # Only WAV format is supported; format parameter is ignored.
        logger.debug("soundfile unavailable, falling back to scipy.io.wavfile")
        _scipy_wavfile.write(str(filename), int(target.sampling_rate), data.astype(np.float32))
    elif np.issubdtype(data.dtype, np.floating) and np.max(np.abs(data)) <= 1:
        sf.write(  # ty: ignore[unresolved-attribute]
            str(filename),
            data,
            int(target.sampling_rate),
            subtype="FLOAT",
            format=format,
        )
    else:
        sf.write(str(filename), data, int(target.sampling_rate), format=format)  # ty: ignore[unresolved-attribute]
    logger.debug(f"Save complete: {filename}")
