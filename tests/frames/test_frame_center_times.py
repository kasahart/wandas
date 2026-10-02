"""Physical STFT centers are distinct from the stable zero-based local axis."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from scipy.signal import ShortTimeFFT, get_window

import wandas as wd
from wandas.pipeline import RecipePlan


@pytest.mark.parametrize(
    ("fft", "window_length", "hop", "window"),
    [(2048, 2048, 512, "hann"), (9, 7, 2, "boxcar"), (32, 16, 8, "hamming")],
)
def test_centers_match_independent_scipy_axis_without_computation(
    fft: int, window_length: int, hop: int, window: str
) -> None:
    source = wd.from_numpy(np.zeros((2, 4096)), sampling_rate=16000).with_source_time_offset([3.0, 5.0])
    expected = ShortTimeFFT(get_window(window, window_length), hop=hop, fs=16000, mfft=fft).t(4096)
    with patch.object(da.Array, "compute", side_effect=AssertionError("compute")):
        spectrum = source.stft(n_fft=fft, win_length=window_length, hop_length=hop, window=window)
        np.testing.assert_allclose(spectrum.frame_center_times, np.array([[3.0], [5.0]]) + expected)
        np.testing.assert_array_equal(spectrum.times, np.arange(len(expected)) * hop / 16000)
        np.testing.assert_array_equal(spectrum.source_times, np.array([[3.0], [5.0]]) + spectrum.times)
    # Return arrays cannot mutate authoritative state.
    spectrum.frame_center_times[:] = 999
    np.testing.assert_allclose(spectrum.frame_center_times, np.array([[3.0], [5.0]]) + expected)
    with pytest.raises(AttributeError):
        spectrum.frame_time_origin = 99


def test_trim_resample_channel_selection_and_time_slicing_preserve_centers() -> None:
    source = wd.from_numpy(np.zeros((2, 4000)), sampling_rate=8000).with_source_time_offset([10.0, 20.0])
    spectrum = source.trim(0.125, 0.375).resampling(16000).stft(n_fft=512, hop_length=128)
    expected = ShortTimeFFT(get_window("hann", 512), hop=128, fs=16000).t(4000)
    np.testing.assert_allclose(spectrum.frame_center_times, np.array([[10.125], [20.125]]) + expected)
    np.testing.assert_allclose(spectrum[1].frame_center_times, spectrum.frame_center_times[1:2])
    np.testing.assert_allclose(spectrum[:, :, 2:8].frame_center_times, spectrum.frame_center_times[:, 2:8])
    with pytest.raises(ValueError, match="continuous"):
        spectrum[:, :, ::2]


def test_centers_survive_same_domain_and_cepstral_transforms_and_recipe(tmp_path: Path) -> None:
    source = wd.from_numpy(np.arange(128.0), sampling_rate=8000)
    spectrum = source.with_source_time_offset(2.5).stft(n_fft=32, hop_length=8)
    plan = RecipePlan.from_dict(RecipePlan.from_frame(spectrum).to_dict())
    for result in [
        spectrum.abs(),
        spectrum.astype("complex64"),
        spectrum * 2,
        spectrum.with_label("copy"),
        spectrum.cache(),
        plan.apply({"input_0": source}),
        spectrum.cepstrum().lifter(0.001).to_spectral_envelope(),
    ]:
        np.testing.assert_array_equal(result.frame_center_times, spectrum.frame_center_times)
    for frame in [spectrum, spectrum.cepstrum()]:
        path = tmp_path / f"{type(frame).__name__}.wdf"
        frame.save(path)
        with xr.open_dataset(path, engine="h5netcdf") as file:
            assert file.attrs["version"] == "0.5"
        loaded = wd.load(path)
        assert isinstance(loaded, (wd.SpectrogramFrame, wd.CepstrogramFrame))
        np.testing.assert_array_equal(loaded.frame_center_times, frame.frame_center_times)
        np.testing.assert_allclose(loaded.data, frame.data)


@pytest.mark.parametrize("kind", ["spectrogram", "cepstrogram"])
def test_missing_origin_is_explicit_and_old_wdf_remains_readable(tmp_path: Path, kind: str) -> None:
    spectrum = wd.from_numpy(np.zeros(128), sampling_rate=8000).stft(n_fft=32, hop_length=8)
    frame = spectrum if kind == "spectrogram" else spectrum.cepstrum()
    path = tmp_path / "old.wdf"
    frame.save(path)
    with xr.open_dataset(path, engine="h5netcdf") as file:
        old = file.load()
    state = json.loads(old.attrs["constructor_json"])
    del state["frame_time_origin"]
    old.attrs["constructor_json"] = json.dumps(state)
    old.attrs["version"] = "0.4"
    old.to_netcdf(path, engine="h5netcdf", mode="w")
    loaded = wd.load(path)
    assert isinstance(loaded, (wd.SpectrogramFrame, wd.CepstrogramFrame))
    assert loaded.frame_time_origin is None
    with pytest.raises(ValueError, match="unknown"):
        _ = loaded.frame_center_times
    np.testing.assert_array_equal(loaded.times, frame.times)
    np.testing.assert_allclose(loaded.data, frame.data)
    manual = wd.SpectrogramFrame(da.zeros((17, 4)), 8000, 32, 8)
    assert manual.frame_time_origin is None
    with pytest.raises(ValueError, match="unknown"):
        _ = manual.frame_center_times
    known = wd.SpectrogramFrame(da.zeros((17, 4)), 8000, 32, 8, frame_time_origin=-0.001)
    np.testing.assert_array_equal(known.frame_center_times, np.array([[-0.001, 0.0, 0.001, 0.002]]))


@pytest.mark.parametrize("origin", [True, "0", float("nan"), float("inf")])
def test_invalid_origin_is_rejected(origin: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="frame_time_origin"):
        wd.SpectrogramFrame(da.zeros((17, 4)), 8000, 32, 8, frame_time_origin=origin)


@pytest.mark.parametrize("kind", ["spectrogram", "cepstrogram"])
@pytest.mark.parametrize("change", ["null", "nan", "bool", "missing", "legacy-version", "unexpected"])
def test_wdf_frame_time_extension_rejects_corrupt_state(tmp_path: Path, kind: str, change: str) -> None:
    spectrum = wd.from_numpy(np.zeros(128), sampling_rate=8000).stft(n_fft=32, hop_length=8)
    frame = spectrum if kind == "spectrogram" else spectrum.cepstrum()
    path = tmp_path / "corrupt.wdf"
    frame.save(path)
    with xr.open_dataset(path, engine="h5netcdf") as file:
        altered = file.load()
    state = json.loads(altered.attrs["constructor_json"])
    if change == "missing":
        del state["frame_time_origin"]
    elif change == "legacy-version":
        altered.attrs["version"] = "0.4"
    elif change == "unexpected":
        state["unknown_field"] = 1
    else:
        state["frame_time_origin"] = {"null": None, "nan": float("nan"), "bool": True}[change]
    altered.attrs["constructor_json"] = json.dumps(state)
    altered.to_netcdf(path, engine="h5netcdf", mode="w")
    with pytest.raises(ValueError):
        wd.load(path)


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_frame_centers_preserve_scipy_rounding_for_pooling_boundaries(rate: int) -> None:
    source = wd.from_numpy(np.zeros(rate * 18), sampling_rate=rate)
    spectrum = source.stft(n_fft=2048, hop_length=512)
    expected = ShortTimeFFT(get_window("hann", 2048), hop=512, fs=rate).t(rate * 18)
    np.testing.assert_array_equal(spectrum.frame_center_times[0], expected)
