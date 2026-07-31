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
SYMBOL_KINDS: Final = frozenset({"attribute", "class", "function", "mapping"})

# Closed set of governed package surfaces. Adding a surface requires updating this
# tuple and PUBLIC_API_INVENTORY together; the drift gate compares them exactly before
# importing any inventory-provided key.
TRACKED_PACKAGE_SURFACES: Final = (
    "wandas",
    "wandas.frames",
    "wandas.frames.mixins",
    "wandas.processing",
    "wandas.utils",
    "wandas.datasets",
    "wandas.datasets.sample_data",
)

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
    kind: str
    classification: str
    in_all: bool
    documentation: str | None = None
    replacement: str | None = None
    support: str | None = None


PUBLIC_API_INVENTORY: Final = MappingProxyType(
    {
        "wandas": (
            ApiSymbol("__getattr__", "function", PRIVATE_INTERNAL, False),
            ApiSymbol("TYPE_CHECKING", "attribute", PRIVATE_INTERNAL, False),
            ApiSymbol(
                "__version__",
                "attribute",
                STABLE_PUBLIC,
                False,
                "docs/src/api/index.md",
            ),
            ApiSymbol("ChannelFrame", "class", STABLE_PUBLIC, True, "docs/src/api/index.md"),
            ApiSymbol("ChannelCalibration", "class", STABLE_PUBLIC, True, "docs/src/api/index.md"),
            ApiSymbol("CepstralFrame", "class", STABLE_PUBLIC, True, "docs/src/api/index.md"),
            ApiSymbol("CepstrogramFrame", "class", STABLE_PUBLIC, True, "docs/src/api/index.md"),
            ApiSymbol("SpectralFrame", "class", STABLE_PUBLIC, True, "docs/src/api/index.md"),
            ApiSymbol("SpectrogramFrame", "class", STABLE_PUBLIC, True, "docs/src/api/index.md"),
            ApiSymbol("NOctFrame", "class", STABLE_PUBLIC, True, "docs/src/api/index.md"),
            ApiSymbol("ChannelFrameDataset", "class", STABLE_PUBLIC, True, "docs/src/api/index.md"),
            ApiSymbol("read", "function", STABLE_PUBLIC, True, "docs/src/api/index.md"),
            ApiSymbol("load", "function", STABLE_PUBLIC, True, "docs/src/api/index.md"),
            ApiSymbol("from_numpy", "function", STABLE_PUBLIC, True, "docs/src/api/index.md"),
            ApiSymbol("from_folder", "function", STABLE_PUBLIC, True, "docs/src/api/index.md"),
            ApiSymbol("supported_formats", "function", STABLE_PUBLIC, True, "docs/src/api/index.md"),
            ApiSymbol("read_wav", "function", STABLE_PUBLIC, False, "docs/src/api/index.md"),
            ApiSymbol("read_csv", "function", STABLE_PUBLIC, False, "docs/src/api/index.md"),
            ApiSymbol(
                "generate_sin",
                "function",
                STABLE_PUBLIC,
                True,
                "docs/src/api/index.md",
            ),
            ApiSymbol(
                "setup_wandas_logging",
                "function",
                EXPERIMENTAL_PUBLIC,
                False,
                "docs/src/api/index.md",
            ),
            ApiSymbol(
                "from_ndarray",
                "function",
                DEPRECATED_COMPATIBILITY,
                False,
                "docs/src/api/index.md",
                "from_numpy",
                "Deprecated since 0.2.0; retained through 0.6.x and removable no earlier than 0.7.0.",
            ),
        ),
        "wandas.frames": (
            ApiSymbol("ChannelFrame", "class", STABLE_PUBLIC, True, "docs/src/api/frames.md"),
            ApiSymbol("CepstralFrame", "class", STABLE_PUBLIC, True, "docs/src/api/frames.md"),
            ApiSymbol("CepstrogramFrame", "class", STABLE_PUBLIC, True, "docs/src/api/frames.md"),
            ApiSymbol("SpectralFrame", "class", STABLE_PUBLIC, True, "docs/src/api/frames.md"),
            ApiSymbol("SpectrogramFrame", "class", STABLE_PUBLIC, True, "docs/src/api/frames.md"),
            ApiSymbol("NOctFrame", "class", STABLE_PUBLIC, True, "docs/src/api/frames.md"),
            ApiSymbol("RoughnessFrame", "class", STABLE_PUBLIC, True, "docs/src/api/frames.md"),
        ),
        "wandas.frames.mixins": (
            ApiSymbol(
                "ChannelProcessingMixin",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/frames.md",
            ),
            ApiSymbol(
                "ChannelTransformMixin",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/frames.md",
            ),
            ApiSymbol(
                "SpectralPropertiesMixin",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/frames.md",
            ),
        ),
        "wandas.processing": (
            ApiSymbol(
                "AudioOperation",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol(
                "ChannelIndependentAudioOperation",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol(
                "create_operation",
                "function",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol(
                "get_operation",
                "function",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol(
                "register_operation",
                "function",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol("Cepstrum", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol("Lifter", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol(
                "SpectralEnvelope",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol(
                "SpectrogramCepstrum",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol("AWeighting", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol(
                "HighPassFilter",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol(
                "LowPassFilter",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol("CSD", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol("Coherence", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol("FFT", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol("IFFT", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol("ISTFT", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol(
                "NOctSpectrum",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol(
                "NOctSynthesis",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol("STFT", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol(
                "TransferFunction",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol("Welch", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol("ReSampling", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol("RmsTrend", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol("SoundLevel", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol(
                "Trim",
                "class",
                DEPRECATED_COMPATIBILITY,
                True,
                "docs/src/api/processing.md",
                "Frame.trim",
                "Deprecated in 0.6.2; retained through 0.7.x and removable no earlier than 0.8.0.",
            ),
            ApiSymbol("AddWithSNR", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol(
                "HpssHarmonic",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol(
                "HpssPercussive",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol("ABS", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol(
                "ChannelDifference",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol("Mean", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol("Power", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol("Sum", "class", EXPERIMENTAL_PUBLIC, True, "docs/src/api/processing.md"),
            ApiSymbol(
                "LoudnessZwst",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol(
                "LoudnessZwtv",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol(
                "RoughnessDw",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol(
                "RoughnessDwSpec",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol(
                "SharpnessDin",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol(
                "SharpnessDinSt",
                "class",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/processing.md",
            ),
            ApiSymbol("_OPERATION_MODULES", "mapping", PRIVATE_INTERNAL, False),
            ApiSymbol("_OPERATION_REGISTRY", "mapping", PRIVATE_INTERNAL, False),
            ApiSymbol("__getattr__", "function", PRIVATE_INTERNAL, False),
            ApiSymbol("apply_channel_factors", "function", PRIVATE_INTERNAL, False),
            ApiSymbol("register_lazy_operation", "function", PRIVATE_INTERNAL, False),
        ),
        "wandas.utils": (
            ApiSymbol(
                "validate_sampling_rate",
                "function",
                EXPERIMENTAL_PUBLIC,
                True,
                "docs/src/api/utils.md",
            ),
            ApiSymbol("accepted_kwargs", "function", PRIVATE_INTERNAL, False),
            ApiSymbol("filter_kwargs", "function", PRIVATE_INTERNAL, False),
            ApiSymbol("require_dependency", "function", PRIVATE_INTERNAL, False),
            ApiSymbol("require_dependency_attr", "function", PRIVATE_INTERNAL, False),
            ApiSymbol("require_optional_attr", "function", PRIVATE_INTERNAL, False),
            ApiSymbol("require_optional_dependency", "function", PRIVATE_INTERNAL, False),
        ),
        "wandas.datasets": (),
        "wandas.datasets.sample_data": (ApiSymbol("load_sample_signal", "function", PRIVATE_INTERNAL, False),),
    }
)


def public_exports(module_name: str) -> list[str]:
    """Return the canonical ordered ``__all__`` for *module_name*."""

    return [symbol.name for symbol in PUBLIC_API_INVENTORY[module_name] if symbol.in_all]
