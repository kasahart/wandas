import numpy as _np

from wandas._public_api import public_exports as _public_exports
from wandas.utils.types import NDArrayReal as _NDArrayReal

__all__ = _public_exports(__name__)


def load_sample_signal(frequency: float = 5.0, sampling_rate: int = 100, duration: float = 1.0) -> _NDArrayReal:
    """
    Generate a sample sine wave signal.

    Parameters
    ----------
    frequency : float, default=5.0
        Frequency of the signal in Hz.
    sampling_rate : int, default=100
        Sampling rate in Hz.
    duration : float, default=1.0
        Duration of the signal in seconds.

    Returns
    -------
    NDArrayReal
        Signal data as a NumPy array.
    """
    num_samples = int(sampling_rate * duration)
    t = _np.arange(num_samples) / sampling_rate
    signal: _NDArrayReal = _np.sin(2 * _np.pi * frequency * t, dtype=_np.float64)
    return signal
