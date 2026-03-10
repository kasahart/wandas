"""Tests verifying ImportError behavior when mosqito is not installed.

These tests are always run (not skipped) and verify that the correct
ImportError with install instructions is raised when mosqito is unavailable.
"""

import dask.array as da
import numpy as np
import pytest

import wandas.frames.noct as noct_module
import wandas.processing.psychoacoustic as psychoacoustic_module
import wandas.processing.spectral as spectral_module
from wandas.processing.psychoacoustic import (
    LoudnessZwst,
    LoudnessZwtv,
    RoughnessDw,
    RoughnessDwSpec,
    SharpnessDin,
    SharpnessDinSt,
)
from wandas.processing.spectral import NOctSpectrum, NOctSynthesis

_da_from_array = da.from_array  # type: ignore [unused-ignore]

_INSTALL_HINT = r'pip install "wandas\[analysis\]"'


@pytest.fixture()
def mosqito_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate mosqito not being installed by patching _MOSQITO_AVAILABLE to False."""
    monkeypatch.setattr(psychoacoustic_module, "_MOSQITO_AVAILABLE", False)
    monkeypatch.setattr(spectral_module, "_MOSQITO_AVAILABLE", False)
    monkeypatch.setattr(noct_module, "_MOSQITO_AVAILABLE", False)


class TestPsychoacousticUnavailable:
    """Verify ImportError with install hint when mosqito is unavailable."""

    def test_loudness_zwtv_raises(self, mosqito_unavailable: None) -> None:
        t = np.linspace(0, 0.1, int(48000 * 0.1))
        signal = np.array([np.sin(2 * np.pi * 1000 * t)])
        op = LoudnessZwtv(48000)
        dask_signal = _da_from_array(signal)
        with pytest.raises(ImportError, match=_INSTALL_HINT):
            op.process(dask_signal).compute()

    def test_loudness_zwst_raises(self, mosqito_unavailable: None) -> None:
        t = np.linspace(0, 0.1, int(48000 * 0.1))
        signal = np.array([np.sin(2 * np.pi * 1000 * t)])
        op = LoudnessZwst(48000)
        dask_signal = _da_from_array(signal)
        with pytest.raises(ImportError, match=_INSTALL_HINT):
            op.process(dask_signal).compute()

    def test_roughness_dw_raises(self, mosqito_unavailable: None) -> None:
        t = np.linspace(0, 1.0, int(44100 * 1.0))
        signal = np.array([np.sin(2 * np.pi * 1000 * t)])
        op = RoughnessDw(44100)
        dask_signal = _da_from_array(signal)
        with pytest.raises(ImportError, match=_INSTALL_HINT):
            op.process(dask_signal).compute()

    def test_roughness_dw_spec_raises(self, mosqito_unavailable: None) -> None:
        with pytest.raises(ImportError, match=_INSTALL_HINT):
            RoughnessDwSpec(44100)

    def test_sharpness_din_raises(self, mosqito_unavailable: None) -> None:
        t = np.linspace(0, 0.1, int(48000 * 0.1))
        signal = np.array([np.sin(2 * np.pi * 1000 * t)])
        op = SharpnessDin(48000)
        dask_signal = _da_from_array(signal)
        with pytest.raises(ImportError, match=_INSTALL_HINT):
            op.process(dask_signal).compute()

    def test_sharpness_din_st_raises(self, mosqito_unavailable: None) -> None:
        t = np.linspace(0, 0.1, int(48000 * 0.1))
        signal = np.array([np.sin(2 * np.pi * 1000 * t)])
        op = SharpnessDinSt(48000)
        dask_signal = _da_from_array(signal)
        with pytest.raises(ImportError, match=_INSTALL_HINT):
            op.process(dask_signal).compute()


class TestSpectralUnavailable:
    """Verify ImportError with install hint for N-octave operations when mosqito is unavailable."""

    def test_noct_spectrum_raises(self, mosqito_unavailable: None) -> None:
        with pytest.raises(ImportError, match=_INSTALL_HINT):
            NOctSpectrum(48000, fmin=20.0, fmax=20000.0)

    def test_noct_synthesis_raises(self, mosqito_unavailable: None) -> None:
        with pytest.raises(ImportError, match=_INSTALL_HINT):
            NOctSynthesis(48000, fmin=20.0, fmax=20000.0)


class TestNOctFrameUnavailable:
    """Verify ImportError with install hint for NOctFrame.freqs when mosqito is unavailable."""

    def test_noct_frame_freqs_raises(self, mosqito_unavailable: None) -> None:
        from wandas.frames.noct import NOctFrame

        # Build a minimal NOctFrame without hitting _center_freq at construction time
        data = _da_from_array(np.zeros((1, 5)))
        frame = NOctFrame(
            data=data,
            sampling_rate=48000,
            fmin=20.0,
            fmax=20000.0,
            n=3,
            G=10,
            fr=1000,
        )
        with pytest.raises(ImportError, match=_INSTALL_HINT):
            _ = frame.freqs
