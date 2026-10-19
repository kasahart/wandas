"""Shared helpers for assertions against private Frame storage contracts."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from wandas.core.base_frame import BaseFrame
from wandas.io.readers import SoundFileReader

# Audio suffixes are listed only when the loaded libsndfile decodes them.
DEFAULT_FORMATS = [
    extension
    for extension in [
        ".aif",
        ".aiff",
        ".au",
        ".caf",
        ".csv",
        ".flac",
        ".mp3",
        ".oga",
        ".ogg",
        ".opus",
        ".rf64",
        ".snd",
        ".w64",
        ".wav",
    ]
    if extension == ".csv" or extension in SoundFileReader.supported_extensions
]


def channel_first_values(frame: BaseFrame[Any]) -> NDArray[Any]:
    """Materialize calibrated Frame values with the internal channel axis intact."""
    values = frame._compute()
    assert isinstance(values, np.ndarray)
    return values
