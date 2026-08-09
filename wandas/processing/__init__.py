"""
Audio time series processing operations.

This module provides audio processing operations for time series data.
"""

from typing import Any

from wandas.processing.base import (
    AudioOperation,
    ChannelIndependentAudioOperation,
    create_operation,
    get_operation,
    register_lazy_operation,
    register_operation,
)
from wandas.processing.calibration import apply_channel_factors
from wandas.processing.conversion import Astype
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
    # Cepstral
    "Cepstrum": ("cepstrum", "wandas.processing.cepstral", "Cepstrum"),
    "Lifter": ("lifter", "wandas.processing.cepstral", "Lifter"),
    "SpectralEnvelope": (
        "spectral_envelope",
        "wandas.processing.cepstral",
        "SpectralEnvelope",
    ),
    "SpectrogramCepstrum": (
        "spectrogram_cepstrum",
        "wandas.processing.cepstral",
        "SpectrogramCepstrum",
    ),
    # Spectral
    "CSD": ("csd", "wandas.processing.spectral", "CSD"),
    "Coherence": ("coherence", "wandas.processing.spectral", "Coherence"),
    "FFT": ("fft", "wandas.processing.spectral", "FFT"),
    "IFFT": ("ifft", "wandas.processing.spectral", "IFFT"),
    "ISTFT": ("istft", "wandas.processing.spectral", "ISTFT"),
    "NOctSpectrum": ("noct_spectrum", "wandas.processing.spectral", "NOctSpectrum"),
    "NOctSynthesis": ("noct_synthesis", "wandas.processing.spectral", "NOctSynthesis"),
    "STFT": ("stft", "wandas.processing.spectral", "STFT"),
    "TransferFunction": (
        "transfer_function",
        "wandas.processing.spectral",
        "TransferFunction",
    ),
    "Welch": ("welch", "wandas.processing.spectral", "Welch"),
    # Psychoacoustic
    "LoudnessZwst": ("loudness_zwst", "wandas.processing.psychoacoustic", "LoudnessZwst"),
    "LoudnessZwtv": ("loudness_zwtv", "wandas.processing.psychoacoustic", "LoudnessZwtv"),
    "RoughnessDw": ("roughness_dw", "wandas.processing.psychoacoustic", "RoughnessDw"),
    "RoughnessDwSpec": (
        "roughness_dw_spec",
        "wandas.processing.psychoacoustic",
        "RoughnessDwSpec",
    ),
    "SharpnessDin": ("sharpness_din", "wandas.processing.psychoacoustic", "SharpnessDin"),
    "SharpnessDinSt": ("sharpness_din_st", "wandas.processing.psychoacoustic", "SharpnessDinSt"),
}

# These Recipe replay implementations are intentionally private and are not
# public ``wandas.processing`` exports.  They still have stable operation names
# in the runtime registry so persisted Recipe meanings have an explicit provider.
_PRIVATE_LAZY_OPERATION_PROVIDERS = (
    ("_recipe_cepstrum_v1", "wandas.processing.cepstral", "_RecipeCepstrumV1"),
    ("_recipe_fft_v1", "wandas.processing.spectral", "_RecipeFFTV1"),
    ("_recipe_ifft_v1", "wandas.processing.spectral", "_RecipeIFFTV1"),
    ("_recipe_welch_v1", "wandas.processing.spectral", "_RecipeWelchV1"),
    ("_recipe_noct_synthesis_v1", "wandas.processing.spectral", "_RecipeNOctSynthesisV1"),
    ("_recipe_transfer_function_v1", "wandas.processing.spectral", "_RecipeTransferFunctionV1"),
)

for _operation_name, _module_name, _attribute_name in (
    *_LAZY_OPERATION_CLASSES.values(),
    *_PRIVATE_LAZY_OPERATION_PROVIDERS,
):
    register_lazy_operation(
        _operation_name,
        _module_name,
        attribute_name=_attribute_name,
    )


def __getattr__(name: str) -> Any:
    lazy_operation = _LAZY_OPERATION_CLASSES.get(name)
    if lazy_operation is not None:
        operation_name, _, _ = lazy_operation
        operation_class = get_operation(operation_name)
        globals()[name] = operation_class
        return operation_class
    raise AttributeError(f"module 'wandas.processing' has no attribute {name!r}")


__all__ = [  # noqa: RUF022  # intentionally grouped by category
    # Calibration
    "apply_channel_factors",
    # Base
    "AudioOperation",
    "ChannelIndependentAudioOperation",
    "create_operation",
    "get_operation",
    "register_lazy_operation",
    "register_operation",
    # Conversion
    "Astype",
    # Cepstral
    "Cepstrum",
    "Lifter",
    "SpectralEnvelope",
    "SpectrogramCepstrum",
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
