"""Public shape and broadcast contracts for spectral NumPy properties."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import dask.array as da
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import wandas as wd
import wandas.frames.mixins.spectral_properties_mixin as spectral_properties_mixin
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.frames.spectral import SpectralFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.pipeline import RecipePlan
from wandas.utils.util import DB_FLOOR

_SAMPLING_RATE = 8.0
_N_FFT = 8
_N_FREQ = _N_FFT // 2 + 1
_N_TIME = 3


def _metadata(refs: list[float], factors: list[float] | None = None) -> list[ChannelMetadata]:
    """Build deterministic channel metadata with independent references."""
    factors = factors or [1.0] * len(refs)
    return [
        ChannelMetadata(
            label=f"channel-{index}",
            calibration=ChannelCalibration(factor=factor, ref=ref),
        )
        for index, (ref, factor) in enumerate(zip(refs, factors, strict=True))
    ]


def _spectral_values(n_channels: int = 2) -> np.ndarray:
    """Return a deterministic spectrum with nontrivial phases and a zero bin."""
    values = np.array(
        [
            [0.0 + 0.0j, 1.0 + 1.0j, -2.0 + 0.5j, 3.0 - 1.0j, -1.0 - 2.0j],
            [0.0 + 0.0j, 1.0 + 1.0j, -2.0 + 0.5j, 3.0 - 1.0j, -1.0 - 2.0j],
        ],
        dtype=np.complex128,
    )
    return values[:n_channels]


def _spectrogram_values(n_channels: int = 2) -> np.ndarray:
    """Return a deterministic complex spectrogram with a frequency gradient."""
    base = _spectral_values(2)
    values = np.stack([base + (time * 0.25) for time in range(_N_TIME)], axis=-1)
    return values[:n_channels]


def _manual_spectral(
    n_channels: int = 2,
    *,
    refs: list[float] | None = None,
    factors: list[float] | None = None,
) -> SpectralFrame:
    """Create a manually constructed spectral frame."""
    values = _spectral_values(n_channels)
    refs = refs or [1.0] * n_channels
    metadata = _metadata(refs, factors)
    data = values if n_channels > 1 else values[0]
    return SpectralFrame(
        data=da.from_array(data, chunks=(1, -1) if n_channels > 1 else (-1,)),
        sampling_rate=_SAMPLING_RATE,
        n_fft=_N_FFT,
        window="boxcar",
        label="manual-spectrum",
        metadata={"recording": {"take": "shape-contract"}},
        channel_metadata=metadata,
    )


def _manual_spectrogram(
    n_channels: int = 2,
    *,
    refs: list[float] | None = None,
    factors: list[float] | None = None,
) -> SpectrogramFrame:
    """Create a manually constructed spectrogram frame."""
    values = _spectrogram_values(n_channels)
    refs = refs or [1.0] * n_channels
    metadata = _metadata(refs, factors)
    data = values if n_channels > 1 else values[0]
    return SpectrogramFrame(
        data=da.from_array(data, chunks=(1, -1, -1) if n_channels > 1 else (-1, -1)),
        sampling_rate=_SAMPLING_RATE,
        n_fft=_N_FFT,
        hop_length=2,
        win_length=_N_FFT,
        window="boxcar",
        label="manual-spectrogram",
        metadata={"recording": {"take": "shape-contract"}},
        channel_metadata=metadata,
    )


def _property_values(frame: SpectralFrame | SpectrogramFrame) -> dict[str, np.ndarray]:
    """Materialize every public spectral property covered by the contract."""
    values = {
        "data": frame.data,
        "magnitude": frame.magnitude,
        "phase": frame.phase,
        "power": frame.power,
        "dB": frame.dB,
        "dBA": frame.dBA,
    }
    if isinstance(frame, SpectralFrame):
        values["unwrapped_phase"] = frame.unwrapped_phase
    return values


def _expected_levels(
    raw_values: np.ndarray,
    refs: list[float],
    factors: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate calibrated magnitude, dB, and power independently."""
    calibrated = raw_values * np.asarray(factors, dtype=float).reshape((len(factors),) + (1,) * (raw_values.ndim - 1))
    magnitude = np.abs(calibrated)
    reference = np.asarray(refs, dtype=float).reshape((len(refs),) + (1,) * (raw_values.ndim - 1))
    level = 20 * np.log10(np.maximum(magnitude / reference, DB_FLOOR))
    return magnitude, level, magnitude**2


def _independent_a_weighting_db(frequencies: np.ndarray) -> np.ndarray:
    """Evaluate the standard A-weighting expression independently of Wandas."""
    f = np.asarray(frequencies, dtype=float)
    f2 = f**2
    ra = (12194.0**2 * f2**2) / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20.0 * np.log10(ra) + 2.0


def _public_shape(values: np.ndarray, n_channels: int) -> np.ndarray:
    """Remove the analytical singleton channel axis like ``Frame.data``."""
    return values[0] if n_channels == 1 else values


@pytest.mark.parametrize("n_channels", [1, 2])
def test_manual_spectral_properties_match_public_data_shape_and_values(n_channels: int) -> None:
    """All SpectralFrame properties follow data and retain their definitions."""
    refs = [1.0, 2.0][:n_channels]
    factors = [2.0, 0.5][:n_channels]
    frame = _manual_spectral(n_channels, refs=refs, factors=factors)
    assert isinstance(frame._data, da.Array)
    initial_chunks = frame._data.chunks
    before_data = frame._compute().copy()
    before_metadata = frame.metadata
    before_history = frame.operation_history

    values = _property_values(frame)
    public_values = _spectral_values(n_channels)
    magnitude, level, power = _expected_levels(public_values, refs, factors)

    expected_shapes = [_N_FREQ] if n_channels == 1 else [n_channels, _N_FREQ]
    for name, value in values.items():
        assert value.shape == tuple(expected_shapes), name
    np.testing.assert_allclose(
        values["data"], _public_shape(magnitude * np.exp(1j * np.angle(public_values)), n_channels)
    )
    np.testing.assert_allclose(values["magnitude"], _public_shape(magnitude, n_channels))
    np.testing.assert_allclose(values["phase"], _public_shape(np.angle(public_values), n_channels))
    np.testing.assert_allclose(
        values["unwrapped_phase"], _public_shape(np.unwrap(np.angle(public_values), axis=-1), n_channels)
    )
    np.testing.assert_allclose(values["power"], _public_shape(power, n_channels))
    np.testing.assert_allclose(values["dB"], _public_shape(level, n_channels))

    weights = _independent_a_weighting_db(frame.freqs)
    np.testing.assert_allclose(values["dBA"], _public_shape(level + weights, n_channels))

    assert frame.metadata == before_metadata
    assert frame.operation_history == before_history
    assert isinstance(frame._data, da.Array)
    assert frame._data.chunks == initial_chunks
    np.testing.assert_array_equal(frame._compute(), before_data)


@pytest.mark.parametrize("n_channels", [1, 2])
def test_manual_spectrogram_properties_match_public_data_shape_and_values(n_channels: int) -> None:
    """All SpectrogramFrame properties follow data and retain their definitions."""
    refs = [1.0, 2.0][:n_channels]
    factors = [2.0, 0.5][:n_channels]
    frame = _manual_spectrogram(n_channels, refs=refs, factors=factors)
    assert isinstance(frame._data, da.Array)
    initial_chunks = frame._data.chunks
    before_data = frame._compute().copy()
    before_metadata = frame.metadata
    before_history = frame.operation_history

    values = _property_values(frame)
    public_values = _spectrogram_values(n_channels)
    magnitude, level, power = _expected_levels(public_values, refs, factors)

    expected_shape = [_N_FREQ, _N_TIME] if n_channels == 1 else [n_channels, _N_FREQ, _N_TIME]
    for name, value in values.items():
        assert value.shape == tuple(expected_shape), name
    np.testing.assert_allclose(
        values["data"], _public_shape(magnitude * np.exp(1j * np.angle(public_values)), n_channels)
    )
    np.testing.assert_allclose(values["magnitude"], _public_shape(magnitude, n_channels))
    np.testing.assert_allclose(values["phase"], _public_shape(np.angle(public_values), n_channels))
    np.testing.assert_allclose(values["power"], _public_shape(power, n_channels))
    np.testing.assert_allclose(values["dB"], _public_shape(level, n_channels))

    weights = _independent_a_weighting_db(frame.freqs)
    expected_weights = weights.reshape((_N_FREQ, 1)) if n_channels == 1 else weights.reshape((1, _N_FREQ, 1))
    np.testing.assert_allclose(values["dBA"], _public_shape(level + expected_weights, n_channels))

    assert frame.metadata == before_metadata
    assert frame.operation_history == before_history
    assert isinstance(frame._data, da.Array)
    assert frame._data.chunks == initial_chunks
    np.testing.assert_array_equal(frame._compute(), before_data)


@pytest.mark.parametrize("frame_factory", [_manual_spectral, _manual_spectrogram])
def test_db_reference_broadcast_is_channel_specific(
    frame_factory: Callable[..., SpectralFrame | SpectrogramFrame],
) -> None:
    """Different references affect only their corresponding public channel."""
    frame = frame_factory(2, refs=[1.0, 2.0])
    values = frame.data
    if isinstance(frame, SpectralFrame):
        expected_difference = np.full((_N_FREQ,), -20 * np.log10(2.0))
    else:
        expected_difference = np.full((_N_FREQ, _N_TIME), -20 * np.log10(2.0))

    assert frame.dB.shape == values.shape
    np.testing.assert_allclose(frame.dB[1][1:] - frame.dB[0][1:], expected_difference[1:])


def test_db_floor_and_calibration_are_preserved() -> None:
    """Public shape normalization does not change calibration or the dB floor."""
    frame = _manual_spectral(1, refs=[4.0], factors=[2.0])
    expected = 20 * np.log10(np.maximum(np.abs(_spectral_values(1)[0]) * 2.0 / 4.0, DB_FLOOR))

    np.testing.assert_allclose(frame.dB, expected)
    assert frame.dB[0] == pytest.approx(20 * np.log10(DB_FLOOR))


def test_a_weighting_uses_frequency_axis_for_every_public_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinct synthetic weights prove that dBA broadcasts over frequency."""
    weights = np.linspace(-12.0, 12.0, _N_FREQ)

    def fake_a_weighting(frequencies: np.ndarray, min_db: float | None = None) -> np.ndarray:
        assert frequencies.shape == (_N_FREQ,)
        return weights.copy()

    monkeypatch.setattr(spectral_properties_mixin, "a_weighting_db", fake_a_weighting)
    for frame in (_manual_spectral(1), _manual_spectral(2), _manual_spectrogram(1), _manual_spectrogram(2)):
        delta = frame.dBA - frame.dB
        if isinstance(frame, SpectralFrame):
            expected = weights if frame.n_channels == 1 else weights.reshape((1, _N_FREQ))
        else:
            expected = weights.reshape((_N_FREQ, 1)) if frame.n_channels == 1 else weights.reshape((1, _N_FREQ, 1))
        np.testing.assert_allclose(delta, np.broadcast_to(expected, delta.shape))


@pytest.mark.parametrize("n_channels", [1, 2])
def test_fft_and_welch_public_paths_follow_property_shape_contract(n_channels: int) -> None:
    """FFT and Welch domain transitions expose the same property ranks as data."""
    samples = np.arange(32.0)
    source_values = np.vstack([samples, samples + 1.0])[:n_channels]
    source = ChannelFrame.from_numpy(source_values if n_channels > 1 else source_values[0], _SAMPLING_RATE)

    for frame in (
        source.fft(n_fft=_N_FFT, window="boxcar"),
        source.welch(n_fft=_N_FFT, hop_length=4, win_length=_N_FFT, window="boxcar"),
    ):
        values = _property_values(frame)
        expected_shape = [_N_FREQ] if n_channels == 1 else [n_channels, _N_FREQ]
        assert frame.data.shape == tuple(expected_shape)
        assert all(value.shape == frame.data.shape for value in values.values())

    fft_frame = source.fft(n_fft=_N_FFT, window="boxcar")
    expected_fft = np.fft.rfft(source_values, n=_N_FFT, axis=-1)
    expected_fft[..., 1:-1] *= 2.0
    expected_fft /= _N_FFT
    expected_fft = _public_shape(expected_fft, n_channels)
    np.testing.assert_allclose(fft_frame.data, expected_fft)

    from scipy import signal

    _, expected_welch = signal.welch(
        source_values,
        nperseg=_N_FFT,
        noverlap=4,
        nfft=_N_FFT,
        window="boxcar",
        average="mean",
        detrend="constant",
        scaling="spectrum",
        axis=-1,
    )
    expected_welch = np.sqrt(expected_welch)
    expected_welch[..., 1:-1] *= np.sqrt(2.0)
    expected_welch = _public_shape(expected_welch, n_channels)
    welch_frame = source.welch(n_fft=_N_FFT, hop_length=4, win_length=_N_FFT, window="boxcar")
    np.testing.assert_allclose(welch_frame.data, expected_welch)


@pytest.mark.parametrize("n_channels", [1, 2])
def test_stft_public_path_follow_property_shape_contract(n_channels: int) -> None:
    """STFT domain transitions expose the same property ranks as data."""
    samples = np.arange(32.0)
    source_values = np.vstack([samples, samples + 1.0])[:n_channels]
    source = ChannelFrame.from_numpy(source_values if n_channels > 1 else source_values[0], _SAMPLING_RATE)
    frame = source.stft(n_fft=_N_FFT, hop_length=4, win_length=_N_FFT, window="boxcar")

    values = _property_values(frame)
    expected_shape = [_N_FREQ, frame.n_frames] if n_channels == 1 else [n_channels, _N_FREQ, frame.n_frames]
    assert frame.data.shape == tuple(expected_shape)
    assert all(value.shape == frame.data.shape for value in values.values())

    from scipy.signal import ShortTimeFFT
    from scipy.signal.windows import get_window

    short_time_fft = ShortTimeFFT(
        win=get_window("boxcar", _N_FFT),
        hop=4,
        fs=_SAMPLING_RATE,
        mfft=_N_FFT,
        scale_to="magnitude",
    )
    expected_stft = np.asarray(short_time_fft.stft(source_values))
    expected_stft[..., 1:-1, :] *= 2.0
    expected_stft = _public_shape(expected_stft, n_channels)
    np.testing.assert_allclose(frame.data, expected_stft)


@pytest.mark.parametrize("n_channels", [1, 2])
def test_recipe_roundtrip_preserves_spectral_property_shapes(n_channels: int) -> None:
    """Recipe extraction and replay preserve the public property contract."""
    samples = np.arange(32.0)
    source_values = np.vstack([samples, samples + 1.0])[:n_channels]
    source = ChannelFrame.from_numpy(source_values if n_channels > 1 else source_values[0], _SAMPLING_RATE)
    processed_frames = (
        source.fft(n_fft=_N_FFT, window="boxcar"),
        source.welch(n_fft=_N_FFT, hop_length=4, win_length=_N_FFT, window="boxcar"),
        source.stft(n_fft=_N_FFT, hop_length=4, win_length=_N_FFT, window="boxcar"),
    )

    for processed in processed_frames:
        plan = RecipePlan.from_frame(processed)
        loaded = RecipePlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        replayed = cast(SpectralFrame | SpectrogramFrame, loaded.apply({"input_0": source}))

        assert isinstance(replayed._data, da.Array)
        assert replayed.data.shape == processed.data.shape
        assert all(value.shape == replayed.data.shape for value in _property_values(replayed).values())
        np.testing.assert_allclose(replayed.data, processed.data)


@pytest.mark.parametrize("n_channels", [1, 2])
@pytest.mark.parametrize("frame_factory", [_manual_spectral, _manual_spectrogram])
def test_wdf_roundtrip_preserves_spectral_property_shapes(
    frame_factory: Callable[..., SpectralFrame | SpectrogramFrame],
    n_channels: int,
    tmp_path: Path,
) -> None:
    """WDF persists internal storage and restores the same public properties."""
    frame = frame_factory(n_channels)
    path = tmp_path / f"{type(frame).__name__}-{n_channels}.wdf"
    frame.save(path)
    loaded = cast(SpectralFrame | SpectrogramFrame, wd.load(path))

    assert type(loaded) is type(frame)
    assert loaded._xr.dims == frame._xr.dims
    for name, value in _property_values(frame).items():
        loaded_value = _property_values(loaded)[name]
        assert loaded_value.shape == value.shape, name
        np.testing.assert_allclose(loaded_value, value)


def _as_axes(result: Axes | Iterator[Axes]) -> list[Axes]:
    """Normalize a plot result for assertions without changing plotting code."""
    if isinstance(result, Axes):
        return [result]
    return list(result)


def _assert_line_values(axis: Axes, frequencies: np.ndarray, expected: np.ndarray) -> None:
    """Verify both axes and values for a plotted spectral line."""
    assert len(axis.lines) == 1
    line = axis.lines[0]
    np.testing.assert_allclose(np.asarray(line.get_xdata()), frequencies)
    np.testing.assert_allclose(np.asarray(line.get_ydata()), expected)


def _assert_mesh_values(axis: Axes, expected: np.ndarray) -> None:
    """Verify the materialized values in a plotted spectrogram mesh."""
    assert len(axis.collections) == 1
    np.testing.assert_allclose(np.asarray(axis.collections[0].get_array()), expected)


def test_plotting_restores_channel_rank_at_the_boundary() -> None:
    """Frequency, matrix, and spectrogram plots accept squeezed public properties."""
    spectral_single = _manual_spectral(1)
    spectral_multi = _manual_spectral(2)
    spectrogram_single = _manual_spectrogram(1)
    spectrogram_multi = _manual_spectrogram(2)

    try:
        single_axes = _as_axes(spectral_single.plot(overlay=False, Aw=False))
        assert len(single_axes) == 1
        _assert_line_values(single_axes[0], spectral_single.freqs, spectral_single.dB)

        multi_axes = _as_axes(spectral_multi.plot(overlay=False, Aw=True))
        assert len(multi_axes) == 2
        assert [len(axis.lines) for axis in multi_axes] == [1, 1]
        for axis, expected in zip(multi_axes, spectral_multi.dBA, strict=True):
            _assert_line_values(axis, spectral_multi.freqs, expected)

        overlay_axis = spectral_multi.plot(overlay=True, Aw=False)
        assert isinstance(overlay_axis, Axes)
        assert len(overlay_axis.lines) == 2
        for line, expected in zip(overlay_axis.lines, spectral_multi.dB, strict=True):
            np.testing.assert_allclose(np.asarray(line.get_xdata()), spectral_multi.freqs)
            np.testing.assert_allclose(np.asarray(line.get_ydata()), expected)

        caller_axis = plt.subplots()[1]
        assert spectral_single.plot(ax=caller_axis, Aw=True) is caller_axis
        _assert_line_values(caller_axis, spectral_single.freqs, spectral_single.dBA)

        matrix_axes = _as_axes(spectral_single.plot(plot_type="matrix", overlay=False, Aw=True))
        assert len(matrix_axes) == 1
        _assert_line_values(matrix_axes[0], spectral_single.freqs, spectral_single.dBA)

        single_spectrogram_axes = _as_axes(spectrogram_single.plot(Aw=False))
        single_data_axes = [axis for axis in single_spectrogram_axes if axis.get_xlabel() == "Time [s]"]
        assert len(single_data_axes) == 1
        _assert_mesh_values(single_data_axes[0], spectrogram_single.dB)

        caller_spectrogram_axis = plt.subplots()[1]
        assert spectrogram_single.plot(ax=caller_spectrogram_axis, Aw=True) is caller_spectrogram_axis
        _assert_mesh_values(caller_spectrogram_axis, spectrogram_single.dBA)

        multi_spectrogram_axes = _as_axes(spectrogram_multi.plot(Aw=True))
        multi_data_axes = [axis for axis in multi_spectrogram_axes if axis.get_xlabel() == "Time [s]"]
        assert len(multi_data_axes) == 2
        assert all(len(axis.collections) == 1 for axis in multi_data_axes)
        for axis, expected in zip(multi_data_axes, spectrogram_multi.dBA, strict=True):
            _assert_mesh_values(axis, expected)
    finally:
        plt.close("all")
