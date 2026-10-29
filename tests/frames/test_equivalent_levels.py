"""Equivalent level (``ChannelFrame.leq``) and percentile levels (``percentile_levels``)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import wandas as wd

_FS = 48000.0
_T = np.arange(int(_FS * 4)) / _FS


def _tone(amplitude: float, frequency: float = 1000.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * frequency * _T)


def test_leq_of_a_tone_is_its_rms_level() -> None:
    frame = wd.from_numpy(np.stack([_tone(1.0), _tone(0.1)]), sampling_rate=_FS)
    np.testing.assert_allclose(frame.leq(), 20 * np.log10([1 / np.sqrt(2), 0.1 / np.sqrt(2)]), atol=1e-6)


def test_leq_of_calibrated_pressure_is_db_spl_and_matches_the_level_reference() -> None:
    frame = wd.from_numpy(_tone(0.5)[np.newaxis], sampling_rate=_FS, ch_labels=["mic"]).with_calibration(
        {"mic": wd.ChannelCalibration(factor=2.0, unit="Pa")}
    )
    reference = frame.channels[0].level_reference
    assert reference.unit == "dB SPL"
    np.testing.assert_allclose(frame.leq(), 20 * np.log10(1.0 / np.sqrt(2) / 2e-5), atol=1e-6)
    np.testing.assert_allclose(frame.leq("A")[0], reference.to_level(frame.a_weighting().rms[0]), atol=1e-9)


def test_leq_weighting_follows_the_curve() -> None:
    frame = wd.from_numpy(_tone(1.0, frequency=100.0)[np.newaxis], sampling_rate=_FS)
    # A-weighting is about -19.1 dB and C-weighting about -0.3 dB at 100 Hz.
    np.testing.assert_allclose(frame.leq("A") - frame.leq("Z"), -19.1, atol=0.2)
    np.testing.assert_allclose(frame.leq("C") - frame.leq(None), -0.3, atol=0.2)
    with pytest.raises(ValueError, match="frequency weighting"):
        frame.leq(cast(Any, "B"))


def test_percentile_levels_split_loud_and_quiet_parts() -> None:
    t = np.arange(int(_FS * 10)) / _FS
    loud = (t >= 4.0) & (t < 7.0)  # loud 30 % of the time
    signal = np.sin(2 * np.pi * 1000.0 * t) * np.where(loud, 1.0, 0.01)
    frame = wd.from_numpy(signal[np.newaxis], sampling_rate=_FS)
    l5, l95 = frame.percentile_levels([5, 95])[0]

    # The Fast level settles within 0.5 dB after about 0.3 s, so the 5 % points
    # (0.5 s) at both ends are on the steady loud and quiet levels.
    np.testing.assert_allclose(l5, 20 * np.log10(1 / np.sqrt(2)), atol=0.5)
    np.testing.assert_allclose(l95, 20 * np.log10(0.01 / np.sqrt(2)), atol=0.5)


def test_percentile_levels_shape_and_equivalence_to_sound_level() -> None:
    rng = np.random.default_rng(592)
    frame = wd.from_numpy(rng.standard_normal((2, 48000)), sampling_rate=_FS)
    level = np.asarray(frame.sound_level(freq_weighting="A", time_weighting="Slow", dB=True).data)

    one = frame.percentile_levels(10, freq_weighting="A", time_weighting="Slow")
    many = frame.percentile_levels([10, 90], freq_weighting="A", time_weighting="Slow")
    assert one.shape == (2,)
    assert many.shape == (2, 2)
    np.testing.assert_allclose(one, np.percentile(level, 90, axis=-1))
    np.testing.assert_allclose(many, np.percentile(level, [90, 10], axis=-1).T)


def test_level_reductions_leave_the_frame_unchanged() -> None:
    frame = wd.from_numpy(np.stack([_tone(1.0)]), sampling_rate=_FS)
    history = list(frame.operation_history)
    frame.leq("A")
    frame.percentile_levels([5, 95])
    assert list(frame.operation_history) == history


@pytest.mark.parametrize(
    ("amplitude", "dtype", "weighting"),
    [(1e200, np.float64, "Z"), (1e20, np.float32, "Z"), (1e20, np.float32, "A")],
)
def test_leq_of_large_values_does_not_overflow(amplitude: float, dtype: Any, weighting: str) -> None:
    unit = wd.from_numpy(_tone(1.0)[np.newaxis], sampling_rate=_FS)
    large = wd.from_numpy((_tone(1.0) * amplitude).astype(dtype)[np.newaxis], sampling_rate=_FS)
    expected = unit.leq(cast(Any, weighting)) + 20 * np.log10(amplitude)
    np.testing.assert_allclose(large.leq(cast(Any, weighting)), expected, atol=1e-3)


def test_leq_applies_calibration_in_the_log_domain() -> None:
    frame = wd.from_numpy(np.full((1, 4800), 1e308), sampling_rate=_FS, ch_labels=["x"]).with_calibration(
        {"x": wd.ChannelCalibration(factor=2.0, ref=1e308)}
    )
    np.testing.assert_allclose(frame.leq(), [20 * np.log10(2.0)], atol=1e-9)


def test_leq_of_silence_is_the_reference_floor() -> None:
    frame = wd.from_numpy(np.zeros((1, 4800)), sampling_rate=_FS)
    np.testing.assert_array_equal(frame.leq("A"), [-240.0])
