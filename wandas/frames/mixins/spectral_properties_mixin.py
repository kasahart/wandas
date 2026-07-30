"""Mixin providing common spectral properties (magnitude, phase, power, dB, dBA).

These properties are shared between SpectralFrame (2D) and SpectrogramFrame (3D).
Broadcasting differences are handled via ``_data.ndim``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wandas.processing.weighting import a_weighting_db
from wandas.utils.types import NDArrayReal
from wandas.utils.util import ref_weighted_dB


class SpectralPropertiesMixin:
    """Shared magnitude, phase, squared-magnitude, and level properties.

    Host classes must provide ``data`` (computed array),
    ``_data`` (Dask array), ``_channel_metadata``, and ``freqs``.
    The operation that created the host defines the stored quantity and unit.
    """

    # -- read-only properties reused by SpectralFrame & SpectrogramFrame --

    @property
    def magnitude(self: Any) -> NDArrayReal:
        """Absolute magnitude of the stored spectral quantity."""
        result: NDArrayReal = np.abs(self.data)
        return result

    @property
    def phase(self: Any) -> NDArrayReal:
        """Phase angles in radians."""
        result: NDArrayReal = np.angle(self.data)
        return result

    @property
    def power(self: Any) -> NDArrayReal:
        """Squared magnitude, a compatibility property that is not a PSD."""
        mag: NDArrayReal = np.abs(self.data)
        result: NDArrayReal = mag**2
        return result

    @property
    def dB(self: Any) -> NDArrayReal:  # noqa: N802
        """Magnitude level: ``20 * log10(magnitude / channel_ref)``.

        For the canonical FFT, STFT, and Welch amplitude quantities, this is
        an amplitude level.
        """
        mag: NDArrayReal = np.abs(self.data)
        return ref_weighted_dB(mag, self._channel_metadata, self._data.ndim)

    @property
    def dBA(self: Any) -> NDArrayReal:  # noqa: N802
        """A-weighted amplitude level relative to each channel reference."""
        weighted: NDArrayReal = a_weighting_db(frequencies=self.freqs, min_db=None)
        if self._data.ndim == 3:
            # SpectrogramFrame: broadcast over time axis
            result: NDArrayReal = self.dB + weighted[:, np.newaxis]
            return result
        # SpectralFrame: weighted is already (freq,), broadcasts over (channels, freq)
        result = self.dB + weighted
        return result
