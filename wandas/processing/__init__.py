"""
Audio time series processing operations.

This module provides audio processing operations for time series data.
"""

from wandas.processing.base import (
    _OPERATION_REGISTRY,
    AudioOperation,
    create_operation,
    get_operation,
    register_operation,
)
from wandas.processing.effects import (
    AddWithSNR,
    HpssHarmonic,
    HpssPercussive,
)
from wandas.processing.filters import (
    AWeighting,
    HighPassFilter,
    LowPassFilter,
)
from wandas.processing.psychoacoustic import (
    LoudnessZwtv,
)
from wandas.processing.spectral import (
    CSD,
    FFT,
    IFFT,
    ISTFT,
    STFT,
    Coherence,
    NOctSpectrum,
    NOctSynthesis,
    TransferFunction,
    Welch,
)
from wandas.processing.stats import (
    ABS,
    ChannelDifference,
    Mean,
    Power,
    Sum,
)
from wandas.processing.temporal import (
    ReSampling,
    RmsTrend,
    Trim,
)

__all__ = [
    # Base
    "AudioOperation",
    "_OPERATION_REGISTRY",
    "create_operation",
    "get_operation",
    "register_operation",
    # Filters
    "AWeighting",
    "HighPassFilter",
    "LowPassFilter",
    # Spectral
    "CSD",
    "Coherence",
    "FFT",
    "IFFT",
    "ISTFT",
    "NOctSpectrum",
    "NOctSynthesis",
    "STFT",
    "TransferFunction",
    "Welch",
    # Temporal
    "ReSampling",
    "RmsTrend",
    "Trim",
    # Effects
    "AddWithSNR",
    "HpssHarmonic",
    "HpssPercussive",
    # Stats
    "ABS",
    "ChannelDifference",
    "Mean",
    "Power",
    "Sum",
    # Psychoacoustic
    "LoudnessZwtv",
]
