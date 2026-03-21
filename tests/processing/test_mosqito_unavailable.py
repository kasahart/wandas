"""Tests verifying ImportError behavior when mosqito is not installed.

These tests are always run (not skipped) and verify that the correct
ImportError with install instructions is raised when mosqito is unavailable.
"""

import builtins
import importlib
from collections.abc import Callable
from types import ModuleType

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
_MODULES_UNDER_TEST = (
    spectral_module,
    psychoacoustic_module,
    noct_module,
)


def _reload_with_import_error(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    error_factory: Callable[[str], ImportError],
) -> ModuleType:
    """Reload a module while forcing mosqito imports to raise a custom error."""
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globalns: dict[str, object] | None = None,
        localns: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "mosqito" or name.startswith("mosqito."):
            raise error_factory(name)
        return original_import(name, globalns, localns, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return importlib.reload(module)


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


class TestMosqitoImportGuard:
    """Verify only missing mosqito imports are treated as optional."""

    @pytest.mark.parametrize("module", _MODULES_UNDER_TEST, ids=lambda module: module.__name__)
    def test_missing_top_level_mosqito_is_treated_as_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: ModuleType,
    ) -> None:
        try:
            with monkeypatch.context() as m:
                reloaded = _reload_with_import_error(
                    m,
                    module,
                    lambda _name: ModuleNotFoundError("No module named 'mosqito'", name="mosqito"),
                )
                assert reloaded._MOSQITO_AVAILABLE is False
                with pytest.raises(ImportError, match=_INSTALL_HINT):
                    reloaded._require_mosqito()
        finally:
            importlib.reload(module)

    @pytest.mark.parametrize("module", _MODULES_UNDER_TEST, ids=lambda module: module.__name__)
    def test_missing_mosqito_submodule_is_treated_as_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: ModuleType,
    ) -> None:
        missing_name = "mosqito.sound_level_meter.noct_spectrum._center_freq"

        try:
            with monkeypatch.context() as m:
                reloaded = _reload_with_import_error(
                    m,
                    module,
                    lambda _name: ModuleNotFoundError(
                        f"No module named '{missing_name}'",
                        name=missing_name,
                    ),
                )
                assert reloaded._MOSQITO_AVAILABLE is False
                with pytest.raises(ImportError, match=_INSTALL_HINT):
                    reloaded._require_mosqito()
        finally:
            importlib.reload(module)

    @pytest.mark.parametrize("module", _MODULES_UNDER_TEST, ids=lambda module: module.__name__)
    def test_unrelated_module_not_found_error_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: ModuleType,
    ) -> None:
        try:
            with monkeypatch.context() as m:
                with pytest.raises(ModuleNotFoundError, match="totally_unrelated_dependency"):
                    _reload_with_import_error(
                        m,
                        module,
                        lambda _name: ModuleNotFoundError(
                            "No module named 'totally_unrelated_dependency'",
                            name="totally_unrelated_dependency",
                        ),
                    )
        finally:
            importlib.reload(module)

    @pytest.mark.parametrize("module", _MODULES_UNDER_TEST, ids=lambda module: module.__name__)
    def test_plain_import_error_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: ModuleType,
    ) -> None:
        try:
            with monkeypatch.context() as m:
                with pytest.raises(ImportError, match="broken optional dependency import"):
                    _reload_with_import_error(
                        m,
                        module,
                        lambda _name: ImportError("broken optional dependency import"),
                    )
        finally:
            importlib.reload(module)
