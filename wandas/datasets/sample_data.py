import numpy as np

from wandas.utils.types import NDArrayReal


def load_sample_signal(
    frequency: float = 5.0, sampling_rate: int = 100, duration: float = 1.0
) -> NDArrayReal:
    """
    Generates a sample sine wave signal.

    Parameters:
        frequency (float): Frequency of the signal in Hz.
        sampling_rate (int): Sampling rate in Hz.
        duration (float): Duration of the signal in seconds.

    Returns:
        numpy.ndarray: Signal data as a NumPy array.
    """
    num_samples = int(sampling_rate * duration)
    t = np.arange(num_samples) / sampling_rate
    signal: NDArrayReal = np.sin(2 * np.pi * frequency * t, dtype=np.float64)
    return signal
