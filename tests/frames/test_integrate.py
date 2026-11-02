"""Band-limited frequency-domain integration (``ChannelFrame.integrate``)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from dask.array.core import Array as DaArray

import wandas as wd
from tests.frame_helpers import channel_first_values
from wandas.pipeline import RecipePlan
from wandas.processing import get_operation

_FS = 1000.0
_T = np.arange(4000) / _FS  # 4 s: every tone below sits on an exact FFT bin


def _tone(freq: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * _T)


def _accel(*rows: np.ndarray) -> wd.ChannelFrame:
    return wd.from_numpy(np.stack(rows), sampling_rate=_FS, ch_units="m/s^2")


def test_integral_of_a_periodic_tone_matches_the_closed_form() -> None:
    omega = 2 * np.pi * 50.0
    frame = _accel(_tone(50.0), _tone(50.0, amplitude=3.0))

    velocity = channel_first_values(frame.integrate())
    displacement = channel_first_values(frame.integrate(order=2))

    # Integral of A sin(wt) without its constant: -A cos(wt) / w, then -A sin(wt) / w**2.
    np.testing.assert_allclose(velocity, -np.stack([1.0, 3.0])[:, None] * np.cos(omega * _T) / omega, atol=1e-12)
    np.testing.assert_allclose(displacement, -np.stack([1.0, 3.0])[:, None] * np.sin(omega * _T) / omega**2, atol=1e-12)


def test_band_and_zero_frequency_are_removed() -> None:
    frame = _accel(2.0 + _tone(5.0) + _tone(50.0) + _tone(400.0))

    velocity = channel_first_values(frame.integrate(fmin=10.0, fmax=200.0))

    omega = 2 * np.pi * 50.0
    np.testing.assert_allclose(velocity[0], -np.cos(omega * _T) / omega, atol=1e-12)


def test_units_follow_motion_chain_and_calibration_is_applied() -> None:
    frame = _accel(_tone(50.0)).with_calibration([wd.ChannelCalibration(factor=2.0, unit="m/s^2")])

    velocity = frame.integrate()
    displacement = frame.integrate(order=2)

    assert velocity.channels[0].unit == "m/s"
    assert displacement.channels[0].unit == "m"
    assert velocity.channels[0].calibration.factor == 1.0
    assert velocity.labels == ["int(ch0)"]
    omega = 2 * np.pi * 50.0
    np.testing.assert_allclose(velocity.rms, [2.0 / omega / np.sqrt(2)], rtol=1e-9)
    assert frame.integrate(unit="mm/s").channels[0].unit == "mm/s"


def test_numpy_integer_order_infers_the_unit() -> None:
    assert _accel(_tone(50.0)).integrate(order=np.int64(1)).channels[0].unit == "m/s"  # ty: ignore[invalid-argument-type]
    assert _accel(_tone(50.0)).integrate(order=np.int32(2)).channels[0].unit == "m"  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(("unit", "order"), [("", 1), ("V", 1), ("m", 1), ("m/s", 2)])
def test_unit_that_cannot_be_inferred_asks_for_unit(unit: str, order: int) -> None:
    frame = wd.from_numpy(_tone(50.0)[None, :], sampling_rate=_FS, ch_units=unit)
    with pytest.raises(ValueError, match="Pass unit="):
        frame.integrate(order=order)
    assert frame.integrate(order=order, unit="V s").channels[0].unit == "V s"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"order": 0}, "Invalid integration order"),
        ({"order": 1.5}, "Invalid integration order"),
        ({"fmin": 0.0}, "Lower band edge"),
        ({"fmax": 600.0}, "Upper band edge"),
        ({"fmin": 100.0, "fmax": 50.0}, "Invalid integration band"),
        ({"fmin": float("nan")}, "Lower band edge frequency must be finite"),
        ({"fmax": float("nan")}, "Upper band edge frequency must be finite"),
    ],
)
def test_invalid_parameters_raise(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _accel(_tone(50.0)).integrate(**kwargs)  # ty: ignore[invalid-argument-type]


def test_integrate_is_lazy_and_records_one_step() -> None:
    frame = _accel(_tone(50.0), _tone(80.0))
    with patch.object(DaArray, "compute", autospec=True) as compute:
        velocity = frame.integrate(fmin=10.0)
        compute.assert_not_called()

    assert type(velocity) is wd.ChannelFrame
    assert velocity.n_samples == frame.n_samples
    assert velocity.operation_history[-1] == {
        "operation": "wandas.audio.integrate",
        "version": 1,
        "params": {"fmin": 10.0},
    }
    np.testing.assert_array_equal(np.asarray(frame._data), np.stack([_tone(50.0), _tone(80.0)]))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_integrate_keeps_input_precision(dtype: type[np.floating]) -> None:
    frame = wd.from_numpy(_tone(50.0)[None, :].astype(dtype), sampling_rate=_FS, ch_units="m/s^2")
    assert channel_first_values(frame.integrate()).dtype == dtype


def test_integrate_operation_is_registered() -> None:
    operation = get_operation("integrate")(_FS, order=2)
    assert operation.name == "integrate"
    assert operation.calculate_output_dtype(np.dtype(np.float32)) == np.float32


def test_integrate_recipe_roundtrip_keeps_unit() -> None:
    source = _accel(_tone(50.0) + _tone(5.0))
    authored = source.integrate(fmin=10.0, fmax=200.0)
    plan = RecipePlan.from_dict(RecipePlan.from_frame(authored, input_names=("signal",)).to_dict())

    assert [node.operation for node in plan.nodes][-1] == "wandas.audio.integrate"
    replayed = plan.apply({"signal": source})
    assert replayed.channels[0].unit == "m/s"
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(authored), rtol=1e-12)
