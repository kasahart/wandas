"""Normalized cross-correlation (``ChannelFrame.cross_correlation``) and ``CrossCorrelationFrame``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from scipy import signal as ss

import wandas as wd
from tests.frame_helpers import channel_first_values
from wandas.frames.correlation import CrossCorrelationFrame
from wandas.pipeline import RecipePlan
from wandas.processing import get_operation

_FS = 1000.0


def _delayed_pair(delay: int, n: int = 2000) -> np.ndarray:
    x = np.random.default_rng(593).standard_normal(n)
    return np.stack([x, np.roll(x, delay)])


@pytest.mark.parametrize("delay", [7, -12])
def test_delay_of_a_shifted_copy_is_found_with_its_sign(delay: int) -> None:
    frame = wd.from_numpy(_delayed_pair(delay), sampling_rate=_FS, ch_labels=["in", "out"])
    xc = frame.cross_correlation()

    assert type(xc) is CrossCorrelationFrame
    np.testing.assert_allclose(xc.select_pair(1, 0).delays, [delay / _FS])
    np.testing.assert_allclose(xc.select_pair(0, 1).delays, [-delay / _FS])
    np.testing.assert_allclose(xc.select_pair(1, 0).peak_coefficients, [1.0], atol=0.02)


def test_coefficients_equal_normalized_scipy_correlate_and_self_is_one_at_zero_lag() -> None:
    data = np.random.default_rng(1).standard_normal((2, 301))
    xc = wd.from_numpy(data, sampling_rate=_FS).cross_correlation()
    values = channel_first_values(xc)
    energy = np.sum(data**2, axis=-1)

    assert xc.shape == (4, 601)
    np.testing.assert_allclose(xc.lags, ss.correlation_lags(301, 301) / _FS)
    for output in range(2):
        for input_ in range(2):
            expected = ss.correlate(data[output], data[input_]) / np.sqrt(energy[output] * energy[input_])
            np.testing.assert_allclose(values[output * 2 + input_], expected, rtol=1e-10, atol=1e-12)
    zero_lag = int(np.argmin(np.abs(xc.lags)))
    np.testing.assert_allclose(values[[0, 3], zero_lag], 1.0)
    assert np.all(np.abs(values) <= 1.0 + 1e-12)


def test_max_lag_crops_the_lag_axis_without_changing_values() -> None:
    frame = wd.from_numpy(_delayed_pair(5), sampling_rate=_FS)
    full = frame.cross_correlation()
    cropped = frame.cross_correlation(max_lag=0.010)

    assert cropped.lag_start == -10
    assert cropped.n_lags == 21
    np.testing.assert_allclose(cropped.lags[[0, -1]], [-0.010, 0.010])
    centre = int(np.argmin(np.abs(full.lags)))
    np.testing.assert_allclose(channel_first_values(cropped), channel_first_values(full)[:, centre - 10 : centre + 11])
    assert frame.cross_correlation(max_lag=0).n_lags == 1


def test_result_is_lazy_typed_and_keeps_source_state() -> None:
    frame = wd.from_numpy(_delayed_pair(3), sampling_rate=_FS, ch_labels=["a", "b"]).with_calibration(
        {"b": wd.ChannelCalibration(factor=2.0, unit="Pa")}
    )
    with patch.object(DaArray, "compute", autospec=True) as compute:
        xc = frame.cross_correlation(max_lag=0.01)
        compute.assert_not_called()

    assert isinstance(xc._data, DaArray)
    assert xc.labels == ["corr(a, a)", "corr(a, b)", "corr(b, a)", "corr(b, b)"]
    assert {channel.unit for channel in xc.channels} == {"1"}
    assert xc.source_channel_ids == tuple(frame._channel_ids)
    assert xc.operation_history[-1]["operation"] == "wandas.audio.cross_correlation"
    assert xc.previous is frame
    # The coefficient does not depend on the calibration scale.
    np.testing.assert_allclose(xc.select_pair(1, 0).peak_coefficients, [1.0], atol=0.02)


def test_lag_slicing_selection_and_arithmetic() -> None:
    xc = wd.from_numpy(_delayed_pair(2), sampling_rate=_FS).cross_correlation(max_lag=0.01)
    sliced = xc[:, 5:12]
    assert type(sliced) is CrossCorrelationFrame
    assert sliced.lag_start == xc.lag_start + 5
    np.testing.assert_allclose(sliced.lags, xc.lags[5:12])
    np.testing.assert_allclose(channel_first_values(sliced), channel_first_values(xc)[:, 5:12])
    with pytest.raises(ValueError, match="contiguous"):
        xc[:, ::2]
    pair = xc.select_pair(1, 0)
    assert pair.n_pairs == 1
    assert pair.shape == (21,)
    with pytest.raises(TypeError, match="Arithmetic"):
        xc + xc


def test_zero_energy_channel_gives_nan_rows() -> None:
    data = np.stack([np.zeros(50), np.random.default_rng(2).standard_normal(50)])
    with np.errstate(all="ignore"):
        values = channel_first_values(wd.from_numpy(data, sampling_rate=_FS).cross_correlation())
    assert np.isnan(values[0]).all()
    assert np.isnan(values[1]).all()
    assert not np.isnan(values[3]).any()


@pytest.mark.parametrize(("max_lag", "error"), [(-0.1, ValueError), (float("inf"), TypeError), ("1", TypeError)])
def test_invalid_max_lag_is_rejected(max_lag: object, error: type[Exception]) -> None:
    frame = wd.from_numpy(_delayed_pair(1, n=20), sampling_rate=_FS)
    with pytest.raises(error, match="max_lag"):
        frame.cross_correlation(max_lag=cast(Any, max_lag))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_precision_follows_the_input(dtype: type[np.floating]) -> None:
    xc = wd.from_numpy(_delayed_pair(1, n=64).astype(dtype), sampling_rate=_FS).cross_correlation()
    assert xc._data.dtype == dtype
    assert channel_first_values(xc).dtype == dtype
    assert get_operation("cross_correlation")(_FS).calculate_output_dtype(np.dtype(dtype)) == dtype


def test_wdf_roundtrip_keeps_lags_and_pairs(tmp_path: Path) -> None:
    xc = wd.from_numpy(_delayed_pair(4), sampling_rate=_FS).cross_correlation(max_lag=0.02)[[1, 2], 3:30]
    path = tmp_path / "xcorr.wdf"
    xc.save(path)
    loaded = cast(CrossCorrelationFrame, wd.load(path))

    assert type(loaded) is CrossCorrelationFrame
    assert loaded.lag_start == xc.lag_start
    assert loaded.pair_state == xc.pair_state
    assert loaded.source_channel_ids == xc.source_channel_ids
    np.testing.assert_allclose(channel_first_values(loaded), channel_first_values(xc))


def test_recipe_roundtrip_replays_correlation_and_pair_selection() -> None:
    source = wd.from_numpy(_delayed_pair(6), sampling_rate=_FS)
    authored = source.remove_dc().cross_correlation(max_lag=0.02).select_pair(1, 0)
    plan = RecipePlan.from_dict(RecipePlan.from_frame(authored, input_names=("signal",)).to_dict())
    replayed = cast(CrossCorrelationFrame, plan.apply({"signal": source}))

    assert [node.operation for node in plan.nodes][-2:] == [
        "wandas.audio.cross_correlation",
        "wandas.frame.select_pair",
    ]
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(authored), rtol=1e-12)
    np.testing.assert_allclose(replayed.delays, [6 / _FS])


def test_plot_and_dataframe_use_the_lag_axis() -> None:
    xc = wd.from_numpy(_delayed_pair(2), sampling_rate=_FS).cross_correlation(max_lag=0.005)
    ax = cast(Any, xc.select_pair(1, 0).plot(overlay=True))
    line = ax.get_lines()[0]
    np.testing.assert_allclose(line.get_xdata(), xc.lags)
    assert ax.get_xlabel().startswith("Lag [s]")
    assert ax.get_ylabel() == "Correlation coefficient [1]"
    plt.close("all")

    frame = xc.to_dataframe()
    assert frame.index.name == "lag"
    np.testing.assert_allclose(frame.index.to_numpy(), xc.lags)


def test_inverted_channel_keeps_its_delay_and_a_negative_peak() -> None:
    data = _delayed_pair(4)
    data[1] *= -1.0
    pair = wd.from_numpy(data, sampling_rate=_FS).cross_correlation().select_pair(1, 0)
    np.testing.assert_allclose(pair.delays, [4 / _FS])
    np.testing.assert_allclose(pair.peak_coefficients, [-1.0], atol=0.02)


def _constructor_kwargs(xc: CrossCorrelationFrame) -> dict[str, Any]:
    return {
        "data": xc._data,
        "sampling_rate": xc.sampling_rate,
        "pair_state": xc.pair_state,
        "lag_start": xc.lag_start,
        "source_channel_ids": xc.source_channel_ids,
    }


def test_constructor_validates_lag_state_rank_and_dtype() -> None:
    xc = wd.from_numpy(_delayed_pair(1, n=20), sampling_rate=_FS).cross_correlation(max_lag=0.003)
    kwargs = _constructor_kwargs(xc)
    np.testing.assert_array_equal(CrossCorrelationFrame(**kwargs).coefficients, xc.coefficients)
    with pytest.raises(TypeError, match="lag_start"):
        CrossCorrelationFrame(**{**kwargs, "lag_start": 1.5})
    with pytest.raises(ValueError, match="rank"):
        CrossCorrelationFrame(**{**kwargs, "data": np.zeros((4, 7, 1))})
    with pytest.raises(ValueError, match="at least one lag"):
        CrossCorrelationFrame(**{**kwargs, "data": np.zeros((4, 0))})
    with pytest.raises(TypeError, match="real numeric"):
        CrossCorrelationFrame(**{**kwargs, "data": np.zeros((4, 7), dtype=complex)})
    one = CrossCorrelationFrame(**{**kwargs, "data": np.zeros(7), "pair_state": xc.pair_state[:1]})
    assert one.n_pairs == 1
    with pytest.raises(ValueError, match="key length"):
        xc[0, 1:2, 3]
    with pytest.raises(ValueError, match="slice selectors"):
        xc[:, 3]


def test_wdf_rejects_a_non_integer_lag_start() -> None:
    from wandas.io import wdf_frames

    xc = wd.from_numpy(_delayed_pair(1, n=20), sampling_rate=_FS).cross_correlation(max_lag=0.003)
    state = wdf_frames._correlation_state(xc)
    state["lag_start"] = 1.0
    with pytest.raises(ValueError, match="lag_start"):
        wdf_frames._validate_correlation_constructor_state(state, xc._data)


def test_plot_passes_line_options() -> None:
    xc = wd.from_numpy(_delayed_pair(1, n=50), sampling_rate=_FS).cross_correlation(max_lag=0.003)
    ax = cast(Any, xc.plot(overlay=True, alpha=0.5, label=["a", "b", "c", "d"]))
    assert [line.get_alpha() for line in ax.get_lines()] == [0.5] * 4
    plt.close("all")
