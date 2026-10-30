"""SpectralFrame and SpectrogramFrame hold a selected subset of frequency bins."""

from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pytest

import wandas as wd
from wandas.frames.pairwise import CrossSpectralFrame
from wandas.frames.spectral import SpectralFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.pipeline import RecipePlan

_SR = 16_000
_N_FFT = 1024


def _signal() -> Any:
    return wd.generate_sin(freqs=[440, 1000], duration=0.5, sampling_rate=_SR)


class TestSpectralFrameSubset:
    def test_frequency_slice_keeps_selected_bins(self) -> None:
        spectrum = _signal().fft(n_fft=_N_FFT)

        sliced = spectrum[:, 10:20]

        assert type(sliced) is SpectralFrame
        assert sliced.frequency_indices == tuple(range(10, 20))
        assert not sliced.is_complete_spectrum
        np.testing.assert_array_equal(sliced.freqs, spectrum.freqs[10:20])
        np.testing.assert_array_equal(sliced.data, spectrum.data[..., 10:20])
        np.testing.assert_allclose(sliced.dB, spectrum.dB[..., 10:20])
        np.testing.assert_allclose(sliced.dBA, spectrum.dBA[..., 10:20])

    def test_complete_spectrum_reports_every_bin(self) -> None:
        spectrum = _signal().fft(n_fft=_N_FFT)

        assert spectrum.is_complete_spectrum
        assert spectrum.frequency_indices == tuple(range(_N_FFT // 2 + 1))
        assert spectrum[0].is_complete_spectrum

    def test_slice_of_slice_composes(self) -> None:
        spectrum = _signal().fft(n_fft=_N_FFT)

        sliced = spectrum[:, 10:100][:, 5:8]

        assert sliced.frequency_indices == (15, 16, 17)
        np.testing.assert_array_equal(sliced.freqs, spectrum.freqs[15:18])

    def test_stepped_slice_keeps_its_bins(self) -> None:
        spectrum = _signal().fft(n_fft=_N_FFT)

        sliced = spectrum[:, 0:9:4]

        assert sliced.frequency_indices == (0, 4, 8)
        np.testing.assert_array_equal(sliced.freqs, spectrum.freqs[0:9:4])

    def test_dataframe_index_uses_selected_frequencies(self) -> None:
        pytest.importorskip("pandas")
        sliced = _signal().fft(n_fft=_N_FFT)[:, 10:20]

        np.testing.assert_array_equal(sliced.to_dataframe().index.values, sliced.freqs)

    def test_scalar_arithmetic_keeps_bins(self) -> None:
        sliced = _signal().fft(n_fft=_N_FFT)[:, 10:20]

        scaled = sliced * 2.0

        assert scaled.frequency_indices == sliced.frequency_indices
        np.testing.assert_allclose(scaled.data, sliced.data * 2.0)

    def test_arithmetic_rejects_different_bins(self) -> None:
        spectrum = _signal().fft(n_fft=_N_FFT)

        with pytest.raises(ValueError, match="frequency bins mismatch"):
            spectrum[:, 10:20] + spectrum[:, 20:30]

    def test_arithmetic_rejects_same_bins_of_different_n_fft(self) -> None:
        # A 512-sample signal zero-padded to two FFT sizes keeps the same window length.
        signal = _signal()[:, :512]
        left = signal.fft(n_fft=1024)[:, 10:20]
        right = signal.fft(n_fft=2048)[:, 10:20]
        assert left.win_length == right.win_length

        with pytest.raises(ValueError, match="frequency bins mismatch"):
            left + right

    @pytest.mark.parametrize("method", ["ifft", "_ifft_recipe_v2"])
    def test_ifft_needs_complete_spectrum(self, method: str) -> None:
        sliced = _signal().fft(n_fft=_N_FFT)[:, 10:20]

        with pytest.raises(ValueError, match=r"ifft\(\) needs the complete spectrum"):
            getattr(sliced, method)()

    def test_noct_synthesis_needs_complete_spectrum(self) -> None:
        signal = wd.generate_sin(freqs=[1000], duration=0.5, sampling_rate=48_000)
        sliced = signal.fft(n_fft=4096)[:, 10:20]

        with pytest.raises(ValueError, match=r"noct_synthesis\(\) needs the complete spectrum"):
            sliced.noct_synthesis(fmin=100, fmax=4000)

    def test_constructor_validates_frequency_indices(self) -> None:
        data = da.ones((1, 3), dtype=complex)

        frame = SpectralFrame(data, _SR, n_fft=8, frequency_indices=[1, 2, 4])
        np.testing.assert_array_equal(frame.freqs, np.fft.rfftfreq(8, 1 / _SR)[[1, 2, 4]])

        with pytest.raises(ValueError, match="Expected: 2 bins for frequency_indices"):
            SpectralFrame(data, _SR, n_fft=8, frequency_indices=[1, 2])
        with pytest.raises(ValueError, match="outside the n_fft frequency axis"):
            SpectralFrame(data, _SR, n_fft=8, frequency_indices=[1, 2, 5])
        with pytest.raises(ValueError, match="duplicate"):
            SpectralFrame(data, _SR, n_fft=8, frequency_indices=[1, 1, 2])
        with pytest.raises(ValueError, match="Expected: 5 bins for n_fft=8"):
            SpectralFrame(data, _SR, n_fft=8)

    def test_wdf_round_trip_keeps_subset(self, tmp_path: Path) -> None:
        sliced = _signal().fft(n_fft=_N_FFT)[:, 10:20]
        path = tmp_path / "subset.wdf"

        sliced.save(path)
        loaded = wd.load(path)

        assert isinstance(loaded, SpectralFrame)
        assert loaded.frequency_indices == sliced.frequency_indices
        np.testing.assert_array_equal(loaded.freqs, sliced.freqs)
        np.testing.assert_array_equal(loaded.data, sliced.data)

    def test_recipe_replays_frequency_slice(self) -> None:
        sliced = _signal().fft(n_fft=_N_FFT)[:, 10:20]
        plan = RecipePlan.from_dict(RecipePlan.from_frame(sliced, input_names=("signal",)).to_dict())

        replayed = plan.apply({"signal": wd.generate_sin(freqs=[500], duration=0.5, sampling_rate=_SR)})

        assert isinstance(replayed, SpectralFrame)
        assert replayed.frequency_indices == tuple(range(10, 20))


class TestSpectrogramFrameSubset:
    def test_frequency_slice_keeps_selected_bins(self) -> None:
        spectrogram = _signal().stft(n_fft=512, hop_length=128)

        sliced = spectrogram[:, 5:40, :]

        assert type(sliced) is SpectrogramFrame
        assert sliced.frequency_indices == tuple(range(5, 40))
        assert sliced.n_freq_bins == 35
        np.testing.assert_array_equal(sliced.freqs, spectrogram.freqs[5:40])
        np.testing.assert_array_equal(sliced.data, spectrogram.data[..., 5:40, :])
        np.testing.assert_allclose(sliced.dBA, spectrogram.dBA[..., 5:40, :])

    def test_frequency_and_time_slice_together(self) -> None:
        spectrogram = _signal().stft(n_fft=512, hop_length=128)

        sliced = spectrogram[:, 5:40, 10:20]

        assert sliced.frequency_indices == tuple(range(5, 40))
        np.testing.assert_array_equal(sliced.frame_center_times, spectrogram.frame_center_times[:, 10:20])

    def test_arithmetic_rejects_different_bins(self) -> None:
        spectrogram = _signal().stft(n_fft=512, hop_length=128)

        with pytest.raises(ValueError, match="frequency bins mismatch"):
            spectrogram[:, 5:40, :] + spectrogram[:, 10:45, :]

    def test_arithmetic_rejects_same_bins_of_different_n_fft(self) -> None:
        signal = _signal()
        left = signal.stft(n_fft=512, hop_length=128)[:, 5:40, :]
        right = signal.stft(n_fft=1024, hop_length=128, win_length=512)[:, 5:40, :]

        with pytest.raises(ValueError, match="frequency bins mismatch"):
            left + right

    def test_arithmetic_with_same_bins(self) -> None:
        sliced = _signal().stft(n_fft=512, hop_length=128)[:, 5:40, :]

        summed = sliced + sliced

        assert summed.frequency_indices == sliced.frequency_indices
        np.testing.assert_allclose(summed.data, sliced.data * 2)

    def test_time_slice_keeps_complete_spectrum(self) -> None:
        spectrogram = _signal().stft(n_fft=512, hop_length=128)

        assert spectrogram[:, :, 10:20].is_complete_spectrum

    def test_get_frame_at_keeps_bins(self) -> None:
        spectrogram = _signal().stft(n_fft=512, hop_length=128)[:, 5:40, :]

        frame = spectrogram.get_frame_at(3)

        assert isinstance(frame, SpectralFrame)
        assert frame.frequency_indices == tuple(range(5, 40))
        np.testing.assert_array_equal(frame.freqs, spectrogram.freqs)

    def test_abs_keeps_bins(self) -> None:
        spectrogram = _signal().stft(n_fft=512, hop_length=128)[:, 5:40, :]

        assert spectrogram.abs().frequency_indices == tuple(range(5, 40))

    def test_csd_needs_complete_spectrum_and_its_result_slices(self) -> None:
        spectrogram = _signal().stft(n_fft=512, hop_length=128)

        with pytest.raises(ValueError, match=r"csd\(\) needs the complete spectrum"):
            spectrogram[:, 5:40, :].csd()

        sliced = spectrogram.csd()[:, 5:40]
        assert isinstance(sliced, CrossSpectralFrame)
        np.testing.assert_array_equal(sliced.freqs, spectrogram.freqs[5:40])

    @pytest.mark.parametrize("method", ["istft", "to_channel_frame", "cepstrum"])
    def test_inverse_and_cepstrum_need_complete_spectrum(self, method: str) -> None:
        sliced = _signal().stft(n_fft=512, hop_length=128)[:, 5:40, :]

        with pytest.raises(ValueError, match="needs the complete spectrum"):
            getattr(sliced, method)()

    def test_plot_uses_selected_frequencies(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        sliced = _signal().stft(n_fft=512, hop_length=128)[:, 5:40, :]
        _, ax = plt.subplots()
        try:
            sliced[0].plot(ax=ax)
            # Cells are centered on the bins, so the axis reaches half a bin beyond them.
            half_bin = _SR / 512 / 2
            low, high = ax.get_ylim()
            assert sliced.freqs[0] - half_bin - 1e-9 <= low < sliced.freqs[0]
            assert sliced.freqs[-1] < high <= sliced.freqs[-1] + half_bin + 1e-9
        finally:
            plt.close("all")

    def test_wdf_round_trip_keeps_subset(self, tmp_path: Path) -> None:
        sliced = _signal().stft(n_fft=512, hop_length=128)[:, 5:40, :]
        path = tmp_path / "subset-spectrogram.wdf"

        sliced.save(path)
        loaded = wd.load(path)

        assert isinstance(loaded, SpectrogramFrame)
        assert loaded.frequency_indices == sliced.frequency_indices
        np.testing.assert_array_equal(loaded.data, sliced.data)

    def test_complete_wdf_keeps_original_fields(self, tmp_path: Path) -> None:
        import json

        import xarray as xr

        path = tmp_path / "complete-spectrogram.wdf"
        _signal().stft(n_fft=512, hop_length=128).save(path)

        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            state = json.loads(dataset.attrs["constructor_json"])

        assert "frequency_indices" not in state
