"""Canonical classification of Wandas package export surfaces.

This module is intentionally dependency-free. Package ``__init__`` modules import
only :func:`public_exports`, so the inventory cannot introduce import cycles or eager
optional-dependency loading.
"""

from types import MappingProxyType
from typing import Final, NamedTuple

STABLE_PUBLIC: Final = "stable public"
EXPERIMENTAL_PUBLIC: Final = "experimental public"
DEPRECATED_COMPATIBILITY: Final = "deprecated compatibility"
PRIVATE_INTERNAL: Final = "private/internal"

CLASSIFICATIONS: Final = frozenset(
    {
        STABLE_PUBLIC,
        EXPERIMENTAL_PUBLIC,
        DEPRECATED_COMPATIBILITY,
        PRIVATE_INTERNAL,
    }
)


class ApiSymbol(NamedTuple):
    """One explicitly classified name on a package-level module surface."""

    name: str
    classification: str
    in_all: bool
    documentation: str | None = None
    replacement: str | None = None
    support: str | None = None


_INVENTORY = {
    "wandas": (
        ApiSymbol("ChannelFrame", STABLE_PUBLIC, True, "docs/src/api/index.md"),
        ApiSymbol("ChannelCalibration", STABLE_PUBLIC, True, "docs/src/api/index.md"),
        ApiSymbol("CepstralFrame", STABLE_PUBLIC, True, "docs/src/api/index.md"),
        ApiSymbol("CepstrogramFrame", STABLE_PUBLIC, True, "docs/src/api/index.md"),
        ApiSymbol("SpectralFrame", STABLE_PUBLIC, True, "docs/src/api/index.md"),
        ApiSymbol("SpectrogramFrame", STABLE_PUBLIC, True, "docs/src/api/index.md"),
        ApiSymbol("NOctFrame", STABLE_PUBLIC, True, "docs/src/api/index.md"),
        ApiSymbol("ChannelFrameDataset", STABLE_PUBLIC, True, "docs/src/api/index.md"),
        ApiSymbol("read", STABLE_PUBLIC, True, "docs/src/api/index.md"),
        ApiSymbol("load", STABLE_PUBLIC, True, "docs/src/api/index.md"),
        ApiSymbol("from_numpy", STABLE_PUBLIC, True, "docs/src/api/index.md"),
        ApiSymbol("from_folder", STABLE_PUBLIC, True, "docs/src/api/index.md"),
        ApiSymbol("supported_formats", STABLE_PUBLIC, True, "docs/src/api/index.md"),
        ApiSymbol("read_wav", STABLE_PUBLIC, False, "docs/src/api/index.md"),
        ApiSymbol("read_csv", STABLE_PUBLIC, False, "docs/src/api/index.md"),
        ApiSymbol(
            "generate_sin",
            EXPERIMENTAL_PUBLIC,
            False,
            "docs/src/api/index.md",
        ),
        ApiSymbol(
            "setup_wandas_logging",
            EXPERIMENTAL_PUBLIC,
            False,
            "docs/src/api/index.md",
        ),
        ApiSymbol(
            "from_ndarray",
            DEPRECATED_COMPATIBILITY,
            False,
            "docs/src/api/index.md",
            "from_numpy",
            "Deprecated since 0.2.0; retained through 0.6.x and removable no earlier than 0.7.0.",
        ),
    ),
    "wandas.frames": (
        ApiSymbol("ChannelFrame", STABLE_PUBLIC, True, "docs/src/api/frames.md"),
        ApiSymbol("CepstralFrame", STABLE_PUBLIC, True, "docs/src/api/frames.md"),
        ApiSymbol("CepstrogramFrame", STABLE_PUBLIC, True, "docs/src/api/frames.md"),
        ApiSymbol("SpectralFrame", STABLE_PUBLIC, True, "docs/src/api/frames.md"),
        ApiSymbol("SpectrogramFrame", STABLE_PUBLIC, True, "docs/src/api/frames.md"),
        ApiSymbol("NOctFrame", STABLE_PUBLIC, True, "docs/src/api/frames.md"),
        ApiSymbol("RoughnessFrame", STABLE_PUBLIC, True, "docs/src/api/frames.md"),
    ),
    "wandas.frames.mixins": (
        ApiSymbol(
            "ChannelProcessingMixin",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/frames.md",
        ),
        ApiSymbol(
            "ChannelTransformMixin",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/frames.md",
        ),
        ApiSymbol(
            "SpectralPropertiesMixin",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/frames.md",
        ),
    ),
    "wandas.processing": (
        ApiSymbol(
            "AudioOperation",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol(
            "ChannelIndependentAudioOperation",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol(
            "create_operation",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol(
            "get_operation",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol(
            "register_operation",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol("Cepstrum", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol("Lifter", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol(
            "SpectralEnvelope",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol(
            "SpectrogramCepstrum",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol("AWeighting", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol(
            "HighPassFilter",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol(
            "LowPassFilter",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol("CSD", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol("Coherence", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol("FFT", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol("IFFT", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol("ISTFT", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol(
            "NOctSpectrum",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol(
            "NOctSynthesis",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol("STFT", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol(
            "TransferFunction",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol("Welch", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol("ReSampling", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol("RmsTrend", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol("SoundLevel", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol(
            "Trim",
            DEPRECATED_COMPATIBILITY,
            True,
            "docs/src/api/processing.md",
            "Frame.trim",
            "Deprecated in 0.6.2; retained through 0.6.x and removable no earlier than 0.7.0.",
        ),
        ApiSymbol("AddWithSNR", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol(
            "HpssHarmonic",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol(
            "HpssPercussive",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol("ABS", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol(
            "ChannelDifference",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol("Mean", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol("Power", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol("Sum", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
        ApiSymbol(
            "LoudnessZwst",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol(
            "LoudnessZwtv",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol(
            "RoughnessDw",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol(
            "RoughnessDwSpec",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol(
            "SharpnessDin",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol(
            "SharpnessDinSt",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/processing.md",
        ),
        ApiSymbol("_OPERATION_MODULES", PRIVATE_INTERNAL, False),
        ApiSymbol("_OPERATION_REGISTRY", PRIVATE_INTERNAL, False),
        ApiSymbol("apply_channel_factors", PRIVATE_INTERNAL, False),
        ApiSymbol("register_lazy_operation", PRIVATE_INTERNAL, False),
    ),
    "wandas.utils": (
        ApiSymbol(
            "validate_sampling_rate",
            EXPERIMENTAL_PUBLIC,
            True,
            "docs/src/api/utils.md",
        ),
        ApiSymbol("accepted_kwargs", PRIVATE_INTERNAL, False),
        ApiSymbol("filter_kwargs", PRIVATE_INTERNAL, False),
        ApiSymbol("require_dependency", PRIVATE_INTERNAL, False),
        ApiSymbol("require_dependency_attr", PRIVATE_INTERNAL, False),
        ApiSymbol("require_optional_attr", PRIVATE_INTERNAL, False),
        ApiSymbol("require_optional_dependency", PRIVATE_INTERNAL, False),
    ),
    "wandas.datasets": (),
    "wandas.datasets.sample_data": (ApiSymbol("load_sample_signal", PRIVATE_INTERNAL, False),),
}

PUBLIC_API_INVENTORY: Final = MappingProxyType(_INVENTORY)


def public_exports(module_name: str) -> list[str]:
    """Return the canonical ordered ``__all__`` for *module_name*."""

    return [symbol.name for symbol in PUBLIC_API_INVENTORY[module_name] if symbol.in_all]
