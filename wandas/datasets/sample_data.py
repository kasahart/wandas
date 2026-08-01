import numpy as np

from wandas.utils.types import NDArrayReal

__all__ = ["load_sample_signal"]


def load_sample_signal(frequency: float = 5.0, sampling_rate: int = 100, duration: float = 1.0) -> NDArrayReal:
    """Generate a sample sine-wave signal.

    Args:
        frequency: Frequency of the signal in Hz.
        sampling_rate: Sampling rate in Hz.
        duration: Duration of the signal in seconds.

    Returns:
        Signal data as a NumPy array.
    """
    num_samples = int(sampling_rate * duration)
    t = np.arange(num_samples) / sampling_rate
    signal: NDArrayReal = np.sin(2 * np.pi * frequency * t, dtype=np.float64)
    return signal
