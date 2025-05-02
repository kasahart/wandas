import numpy as np

from wandas.utils.types import NDArrayReal


def load_sample_signal(
    frequency: float = 5.0, sampling_rate: int = 100, duration: float = 1.0
) -> NDArrayReal:
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
    t = np.arange(num_samples) / sampling_rate
    signal: NDArrayReal = np.sin(2 * np.pi * frequency * t, dtype=np.float64)
    return signal
