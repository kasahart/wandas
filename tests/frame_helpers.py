"""Shared helpers for assertions against private Frame storage contracts."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from wandas.core.base_frame import BaseFrame


def channel_first_values(frame: BaseFrame[Any]) -> NDArray[Any]:
    """Materialize calibrated Frame values with the internal channel axis intact."""
    values = frame._compute()
    assert isinstance(values, np.ndarray)
    return values
