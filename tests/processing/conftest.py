"""Standard test fixtures for processing-layer tests.

Provides deterministic, analytically predictable signals as (DaskArray, sr) tuples
following the Grand Policy fixture naming convention (_dask suffix).
"""

import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.utils.dask_helpers import da_from_array

# ---------------------------------------------------------------------------
# Standard fixtures (per test-processing-policy.instructions.md)
# ---------------------------------------------------------------------------


@pytest.fixture
def sine_1khz_48k_dask() -> tuple[DaArray, int]:
    """1 kHz pure tone, 1 s, sr=48000.

    Standard signal for psychoacoustic tests.
    FFT peak at bin 1000 when n_fft=sr (1 Hz/bin).
    """
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = np.sin(2 * np.pi * 1000 * t).reshape(1, -1)
    return da_from_array(data, chunks=(1, -1)), sr


@pytest.fixture
def calibrated_sine_1khz_70dB_dask() -> tuple[DaArray, int]:
    """1 kHz pure tone at 70 dB SPL, 1 s, sr=48000.

    IEC-compliant psychoacoustic test signal.
    Amplitude = p_ref * 10^(70/20) * sqrt(2) where p_ref = 2e-5 Pa.
    """
    sr = 48000
    p_ref = 2e-5  # Pa, reference sound pressure
    amplitude = p_ref * 10 ** (70 / 20) * np.sqrt(2)
    t = np.linspace(0, 1, sr, endpoint=False)
    data = (amplitude * np.sin(2 * np.pi * 1000 * t)).reshape(1, -1)
    return da_from_array(data, chunks=(1, -1)), sr


@pytest.fixture
def dual_tone_16k_dask() -> tuple[DaArray, int]:
    """Composite tone: 50 Hz + 1000 Hz, 1 s, sr=16000.

    Used to verify filter pass/stop characteristics.
    Equivalent to composite_50hz_1khz_dask; aliased for policy compliance.
    """
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = (np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 1000 * t)).reshape(1, -1)
    return da_from_array(data, chunks=(1, -1)), sr


@pytest.fixture
def impulse_16k_dask() -> tuple[DaArray, int]:
    """Unit impulse, sr=16000, 1 s duration.

    Used to verify filter impulse responses.
    Single sample at t=0 with value 1.0, rest zeros.
    """
    sr = 16000
    data = np.zeros((1, sr))
    data[0, 0] = 1.0
    return da_from_array(data, chunks=(1, -1)), sr


# ---------------------------------------------------------------------------
# Pure-tone fixtures  (single frequency, FFT peak analytically predictable)
# ---------------------------------------------------------------------------


@pytest.fixture
def pure_sine_1khz_dask() -> tuple[DaArray, int]:
    """1 kHz pure sine, 1 s, sr=16000.

    FFT peak at bin index = freq * N / sr = 1000 * 16000 / 16000 = 1000.
    """
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = np.sin(2 * np.pi * 1000 * t).reshape(1, -1)
    return da_from_array(data, chunks=(1, -1)), sr


@pytest.fixture
def pure_sine_440hz_dask() -> tuple[DaArray, int]:
    """440 Hz pure sine, 1 s, sr=16000.

    Standard tuning reference. FFT peak at bin 440.
    """
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = np.sin(2 * np.pi * 440 * t).reshape(1, -1)
    return da_from_array(data, chunks=(1, -1)), sr


# ---------------------------------------------------------------------------
# Composite-tone fixtures  (multiple frequencies for filter pass/stop tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def composite_50hz_1khz_dask() -> tuple[DaArray, int]:
    """Composite signal: 50 Hz + 1000 Hz, 1 s, sr=16000.

    Used to verify HPF/LPF pass/stop characteristics.
    50 Hz is well below typical cutoffs; 1000 Hz is well above.
    """
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    low = np.sin(2 * np.pi * 50 * t)
    high = np.sin(2 * np.pi * 1000 * t)
    data = (low + high).reshape(1, -1)
    return da_from_array(data, chunks=(1, -1)), sr


@pytest.fixture
def composite_100_500_1500hz_dask() -> tuple[DaArray, int]:
    """Composite signal: 100 Hz + 500 Hz + 1500 Hz, 1 s, sr=16000.

    Used for bandpass filter tests.
    100 Hz = below band, 500 Hz = in band, 1500 Hz = above band.
    """
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    f100 = np.sin(2 * np.pi * 100 * t)
    f500 = np.sin(2 * np.pi * 500 * t)
    f1500 = np.sin(2 * np.pi * 1500 * t)
    data = (f100 + f500 + f1500).reshape(1, -1)
    return da_from_array(data, chunks=(1, -1)), sr


# ---------------------------------------------------------------------------
# Stereo fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stereo_sine_440_880hz_dask() -> tuple[DaArray, int]:
    """Stereo: ch0=440 Hz, ch1=880 Hz, 1 s, sr=16000.

    Verifies multi-channel processing independence.
    """
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    ch0 = np.sin(2 * np.pi * 440 * t)
    ch1 = np.sin(2 * np.pi * 880 * t)
    data = np.stack([ch0, ch1])
    return da_from_array(data, chunks=(1, -1)), sr


# ---------------------------------------------------------------------------
# Impulse fixture  (unit impulse for filter impulse response verification)
# ---------------------------------------------------------------------------


@pytest.fixture
def impulse_dask() -> tuple[DaArray, int]:
    """Unit impulse, sr=16000, 1 s duration.

    Used to verify filter impulse responses and A-weighting frequency response.
    """
    sr = 16000
    data = np.zeros((1, sr))
    data[0, 0] = 1.0
    return da_from_array(data, chunks=(1, -1)), sr


@pytest.fixture
def impulse_highsr_dask() -> tuple[DaArray, int]:
    """Unit impulse at high sample rate (300 kHz) for A-weighting tests.

    High sr needed to accurately capture A-weighting curve up to 20 kHz.
    """
    sr = 300000
    from scipy.signal import unit_impulse

    data = unit_impulse(sr).reshape(1, -1)
    return da_from_array(data, chunks=(1, -1)), sr


# ---------------------------------------------------------------------------
# DC offset fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def dc_offset_with_sine_dask() -> tuple[DaArray, int]:
    """Signal with DC offset (2.0) + 50 Hz sine, 1 s, sr=1000.

    Mean of signal should be ~2.0 before DC removal.
    """
    sr = 1000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = (2.0 + np.sin(2 * np.pi * 50 * t)).reshape(1, -1)
    return da_from_array(data, chunks=(1, -1)), sr


# ---------------------------------------------------------------------------
# Mixed harmonic+percussive fixture (for HPSS tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_harmonic_percussive_dask() -> tuple[DaArray, int]:
    """Mixed signal: 440 Hz + 880 Hz harmonics + periodic impulses, 1 s, sr=16000.

    Harmonic part: sin(440t) + 0.5*sin(880t)
    Percussive part: unit impulses every sr/8 samples.
    """
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    harmonic = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    percussive = np.zeros_like(t)
    impulse_locs = np.arange(0, sr, sr // 8)
    percussive[impulse_locs] = 1.0
    data = (harmonic + percussive).reshape(1, -1)
    return da_from_array(data, chunks=(1, -1)), sr
