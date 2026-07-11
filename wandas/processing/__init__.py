"""
Audio time series processing operations.

This module provides audio processing operations for time series data.
"""

from typing import Any

from wandas.processing.base import (
    _OPERATION_MODULES,
    _OPERATION_REGISTRY,
    AudioOperation,
    create_operation,
    get_operation,
    register_lazy_operation,
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
    SoundLevel,
    Trim,
)

_LAZY_OPERATION_CLASSES = {
    # Spectral
    "CSD": ("csd", "wandas.processing.spectral"),
    "Coherence": ("coherence", "wandas.processing.spectral"),
    "FFT": ("fft", "wandas.processing.spectral"),
    "IFFT": ("ifft", "wandas.processing.spectral"),
    "ISTFT": ("istft", "wandas.processing.spectral"),
    "NOctSpectrum": ("noct_spectrum", "wandas.processing.spectral"),
    "NOctSynthesis": ("noct_synthesis", "wandas.processing.spectral"),
    "STFT": ("stft", "wandas.processing.spectral"),
    "TransferFunction": ("transfer_function", "wandas.processing.spectral"),
    "Welch": ("welch", "wandas.processing.spectral"),
    # Psychoacoustic
    "LoudnessZwst": ("loudness_zwst", "wandas.processing.psychoacoustic"),
    "LoudnessZwtv": ("loudness_zwtv", "wandas.processing.psychoacoustic"),
    "RoughnessDw": ("roughness_dw", "wandas.processing.psychoacoustic"),
    "RoughnessDwSpec": ("roughness_dw_spec", "wandas.processing.psychoacoustic"),
    "SharpnessDin": ("sharpness_din", "wandas.processing.psychoacoustic"),
    "SharpnessDinSt": ("sharpness_din_st", "wandas.processing.psychoacoustic"),
}

for _operation_name, _module_name in _LAZY_OPERATION_CLASSES.values():
    register_lazy_operation(_operation_name, _module_name)


def __getattr__(name: str) -> Any:
    lazy_operation = _LAZY_OPERATION_CLASSES.get(name)
    if lazy_operation is not None:
        operation_name, _ = lazy_operation
        operation_class = get_operation(operation_name)
        globals()[name] = operation_class
        return operation_class
    raise AttributeError(f"module 'wandas.processing' has no attribute {name!r}")


__all__ = [  # noqa: RUF022  # intentionally grouped by category
    # Base
    "AudioOperation",
    "_OPERATION_MODULES",
    "_OPERATION_REGISTRY",
    "create_operation",
    "get_operation",
    "register_lazy_operation",
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
    "SoundLevel",
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
    "LoudnessZwst",
    "LoudnessZwtv",
    "RoughnessDw",
    "RoughnessDwSpec",
    "SharpnessDin",
    "SharpnessDinSt",
]
