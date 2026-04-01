"""Mixin providing common spectral properties (magnitude, phase, power, dB, dBA).

These properties are shared between SpectralFrame (2D) and SpectrogramFrame (3D).
Broadcasting differences are handled via ``_data.ndim``.
"""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np

from wandas.utils.types import NDArrayReal


class SpectralPropertiesMixin:
    """Shared magnitude / phase / power / dB / dBA properties.

    Host classes must provide ``data`` (computed array),
    ``_data`` (Dask array), ``_channel_metadata``, and ``freqs``.
    """

    # -- read-only properties reused by SpectralFrame & SpectrogramFrame --

    @property
    def magnitude(self: Any) -> NDArrayReal:
        """Magnitude (absolute value) of the complex data."""
        result: NDArrayReal = np.abs(self.data)
        return result

    @property
    def phase(self: Any) -> NDArrayReal:
        """Phase angles in radians."""
        result: NDArrayReal = np.angle(self.data)
        return result

    @property
    def power(self: Any) -> NDArrayReal:
        """Power (squared magnitude)."""
        mag: NDArrayReal = np.abs(self.data)
        result: NDArrayReal = mag**2
        return result

    @property
    def dB(self: Any) -> NDArrayReal:  # noqa: N802
        """Decibel level relative to per-channel reference values."""
        mag: NDArrayReal = np.abs(self.data)
        ref = np.array([ch.ref for ch in self._channel_metadata])
        # ndim == 2  -> SpectralFrame  (channels, freq)       => ref[:, newaxis]
        # ndim == 3  -> SpectrogramFrame (channels, freq, time) => ref[:, newaxis, newaxis]
        extra_dims = self._data.ndim - 1  # number of trailing axes after channels
        ref_shape = ref.reshape((-1,) + (1,) * extra_dims)
        level: NDArrayReal = 20 * np.log10(np.maximum(mag / ref_shape, 1e-12))
        return level

    @property
    def dBA(self: Any) -> NDArrayReal:  # noqa: N802
        """A-weighted decibel level."""
        weighted: NDArrayReal = librosa.A_weighting(frequencies=self.freqs, min_db=None)
        if self._data.ndim == 3:
            # SpectrogramFrame: broadcast over time axis
            result: NDArrayReal = self.dB + weighted[:, np.newaxis]
            return result
        # SpectralFrame: weighted is already (freq,), broadcasts over (channels, freq)
        result = self.dB + weighted
        return result
