# wandas/io/wav_io.py
import logging
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import numpy as np
import soundfile as sf

if TYPE_CHECKING:
    from ..frames.channel import ChannelFrame

logger = logging.getLogger(__name__)


def write_wav(filename: str | Path | BinaryIO, target: "ChannelFrame", format: str | None = None) -> None:
    """
    Write a ChannelFrame object to a WAV file.

    Floating-point samples use IEEE FLOAT regardless of amplitude, preserving
    values outside [-1, 1] without normalization or PCM clipping. Values are
    stored at float32 precision. Integer input retains the format's default
    subtype. Writing computes the calibrated samples synchronously.

    Args:
        filename: Path or writable binary stream. Caller-owned streams remain open.
        target: ChannelFrame. ChannelFrame object containing the data to write.
        format: Required for streams. Paths infer the format from their extension.

    Raises:
        ValueError: If target is not a ChannelFrame or a stream has no format.
    """
    from wandas.frames.channel import ChannelFrame

    if not isinstance(target, ChannelFrame):
        raise ValueError("target must be a ChannelFrame object.")

    destination = str(filename) if isinstance(filename, (str, Path)) else filename
    if not isinstance(filename, (str, Path)) and format is None:
        raise ValueError("format is required when writing to a binary stream")

    logger.debug(f"Saving audio data to file: {filename} (will compute now)")
    data = target._compute()
    data = data.T
    if data.shape[1] == 1:
        data = data.squeeze(axis=1)
    if np.issubdtype(data.dtype, np.floating):
        sf.write(
            destination,
            data,
            int(target.sampling_rate),
            subtype="FLOAT",
            format=format,
        )
    else:
        sf.write(destination, data, int(target.sampling_rate), format=format)
    logger.debug(f"Save complete: {filename}")
