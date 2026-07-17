"""Reference-signal calibration derivation and application contracts."""

import dask.array as da
import numpy as np
import pytest
from scipy.io import wavfile

from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame, _validate_with_calibration_recipe
from wandas.pipeline.errors import RecipeExecutionError
from wandas.pipeline.model import RecipePlan


def _frame(data: np.ndarray, labels: list[str], *, units: list[str] | None = None) -> ChannelFrame:
    channel_units = units or ["V"] * len(labels)
    return ChannelFrame(
        data=da.from_array(data, chunks=(1, -1)),
        sampling_rate=8_000,
        channel_metadata=[
            ChannelMetadata(label=label, unit=unit, extra={"serial": f"SN-{index:03d}"})
            for index, (label, unit) in enumerate(zip(labels, channel_units, strict=True))
        ],
        metadata={"asset": "rig-7"},
    )


def test_derive_calibration_returns_label_mapping_without_mutating_signal() -> None:
    calibration_signal = _frame(
        np.array(
            [
                [0.5, -0.5, 0.5, -0.5],
                [0.25, -0.25, 0.25, -0.25],
            ]
        ),
        ["microphone-left", "microphone-right"],
    )
    original_data = calibration_signal.data.copy()
    original_channels = calibration_signal.channels.to_list()

    calibrations = calibration_signal.derive_calibration(target_level=94.0, unit="Pa")

    target_rms = 2e-5 * 10 ** (94.0 / 20.0)
    assert list(calibrations) == ["microphone-left", "microphone-right"]
    np.testing.assert_allclose(
        [calibration.factor for calibration in calibrations.values()],
        [target_rms / 0.5, target_rms / 0.25],
        rtol=1e-12,
        atol=0.0,
    )
    assert [calibration.unit for calibration in calibrations.values()] == ["Pa", "Pa"]
    assert [calibration.ref for calibration in calibrations.values()] == [2e-5, 2e-5]
    np.testing.assert_array_equal(calibration_signal.data, original_data)
    assert calibration_signal.channels.to_list() == original_channels
    assert calibration_signal.operation_history == []


def test_derive_calibration_broadcasts_one_target_to_many_channels() -> None:
    labels = [f"sensor-{index:03d}" for index in range(100)]
    recorded_rms = np.arange(1.0, 101.0)
    calibration_signal = _frame(
        np.stack([recorded_rms, -recorded_rms], axis=1),
        labels,
    )
    measurement = _frame(np.ones((100, 2)), labels)

    calibrations = calibration_signal.derive_calibration(target_rms=1.0, unit="m/s^2")
    calibrated = measurement.with_calibration(calibrations)

    assert len(calibrations) == 100
    np.testing.assert_allclose(calibrated.data[:, 0], 1.0 / recorded_rms)


def test_derived_calibration_requires_matching_reader_sample_scale(tmp_path) -> None:
    path = tmp_path / "reference.wav"
    wavfile.write(path, 8_000, np.array([16_384, -16_384], dtype=np.int16))
    normalized_reference = ChannelFrame.read_wav(path, labels=["microphone"], normalize=True)
    normalized_measurement = ChannelFrame.read_wav(path, labels=["microphone"], normalize=True)
    raw_measurement = ChannelFrame.read_wav(path, labels=["microphone"], normalize=False)

    normalized_calibrations = normalized_reference.derive_calibration(target_rms=1.0, unit="Pa")
    calibrated = normalized_measurement.with_calibration(normalized_calibrations)

    np.testing.assert_allclose(calibrated.data, np.array([1.0, -1.0]), rtol=0.0, atol=1e-12)
    assert calibrated.channels[0].calibration.sample_scale == "audio-normalized-float"
    with pytest.raises(ValueError, match="sample scale mismatch"):
        raw_measurement.with_calibration(normalized_calibrations)


def test_raw_pcm_derived_calibration_applies_to_same_wav_subtype(tmp_path) -> None:
    path = tmp_path / "reference.wav"
    wavfile.write(path, 8_000, np.array([16_384, -16_384], dtype=np.int16))
    raw_reference = ChannelFrame.read_wav(path, labels=["microphone"], normalize=False)
    raw_measurement = ChannelFrame.read_wav(path, labels=["microphone"], normalize=False)

    raw_calibrations = raw_reference.derive_calibration(target_rms=1.0, unit="Pa")
    calibrated = raw_measurement.with_calibration(raw_calibrations)

    np.testing.assert_allclose(calibrated.data, np.array([1.0, -1.0]), rtol=0.0, atol=1e-12)


def test_derive_calibration_rejects_raw_unsigned_pcm(tmp_path) -> None:
    path = tmp_path / "unsigned-reference.wav"
    wavfile.write(path, 8_000, np.array([192, 64, 192, 64], dtype=np.uint8))
    raw_reference = ChannelFrame.read_wav(path, labels=["microphone"], normalize=False)
    normalized_reference = ChannelFrame.read_wav(path, labels=["microphone"], normalize=True)
    normalized_measurement = ChannelFrame.read_wav(path, labels=["microphone"], normalize=True)

    assert raw_reference.channels[0].calibration.sample_scale == "wav-native-pcm_u8"
    with pytest.raises(ValueError, match="raw unsigned PCM"):
        raw_reference.derive_calibration(target_rms=1.0, unit="Pa")

    calibrations = normalized_reference.derive_calibration(target_rms=1.0, unit="Pa")
    np.testing.assert_allclose(
        normalized_measurement.with_calibration(calibrations).data,
        np.array([1.0, -1.0, 1.0, -1.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_reader_derived_calibration_rejects_processed_measurement(tmp_path) -> None:
    path = tmp_path / "reference.wav"
    wavfile.write(path, 8_000, np.array([16_384, -16_384], dtype=np.int16))
    raw_reference = ChannelFrame.read_wav(path, labels=["microphone"], normalize=False)
    normalized_measurement = ChannelFrame.read_wav(path, labels=["microphone"], normalize=False).normalize()
    calibrations = raw_reference.derive_calibration(target_rms=1.0, unit="Pa")

    with pytest.raises(ValueError, match="unprocessed measurement source Frame"):
        normalized_measurement.with_calibration({0: calibrations["microphone"]})


@pytest.mark.parametrize("normalize", [False, True])
def test_float_wav_sample_scale_is_distinct_from_normalized_pcm(tmp_path, normalize: bool) -> None:
    pcm_path = tmp_path / "pcm.wav"
    float_path = tmp_path / "float.wav"
    wavfile.write(pcm_path, 8_000, np.array([16_384, -16_384], dtype=np.int16))
    wavfile.write(float_path, 8_000, np.array([2.0, -2.0], dtype=np.float32))
    pcm_reference = ChannelFrame.read_wav(pcm_path, labels=["microphone"], normalize=True)
    float_reference = ChannelFrame.read_wav(float_path, labels=["microphone"], normalize=normalize)
    float_measurement = ChannelFrame.read_wav(float_path, labels=["microphone"], normalize=normalize)

    pcm_calibrations = pcm_reference.derive_calibration(target_rms=1.0, unit="Pa")
    float_calibrations = float_reference.derive_calibration(target_rms=1.0, unit="Pa")

    with pytest.raises(ValueError, match="sample scale mismatch"):
        float_measurement.with_calibration(pcm_calibrations)
    np.testing.assert_allclose(
        float_measurement.with_calibration(float_calibrations).data,
        np.array([1.0, -1.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_local_reader_rejects_conflicting_file_type_before_recording_sample_scale(tmp_path) -> None:
    path = tmp_path / "reference.wav"
    wavfile.write(path, 8_000, np.array([16_384, -16_384], dtype=np.int16))

    with pytest.raises(ValueError, match="File type does not match local path extension"):
        ChannelFrame.from_file(path, file_type=".csv")


def test_derived_calibration_recipe_preserves_reader_sample_scale(tmp_path) -> None:
    path = tmp_path / "reference.wav"
    wavfile.write(path, 8_000, np.array([16_384, -16_384], dtype=np.int16))
    reference = ChannelFrame.read_wav(path, labels=["microphone"], normalize=True)
    normalized_measurement = ChannelFrame.read_wav(path, labels=["microphone"], normalize=True)
    raw_measurement = ChannelFrame.read_wav(path, labels=["microphone"], normalize=False)
    calibrations = reference.derive_calibration(target_rms=1.0, unit="Pa")
    calibrated = normalized_measurement.with_calibration(calibrations)

    plan = RecipePlan.from_frame(calibrated, input_names=("signal",))
    payload = plan.to_dict()
    replayed = RecipePlan.from_dict(payload).apply({"signal": normalized_measurement})

    assert calibrated.operation_history[-1]["params"]["calibrations"]["c0"]["sample_scale"] == (
        "audio-normalized-float"
    )
    np.testing.assert_allclose(replayed.data, calibrated.data, rtol=0.0, atol=1e-12)
    with pytest.raises(RecipeExecutionError, match="Recipe operation failed"):
        plan.apply({"signal": raw_measurement})


@pytest.mark.parametrize("sample_scale", [True, ""])
def test_derived_calibration_recipe_rejects_invalid_sample_scale(sample_scale: object) -> None:
    params = {
        "calibrations": {
            "c0": {
                "factor": 2.0,
                "unit": "Pa",
                "ref": 2e-5,
                "sample_scale": sample_scale,
            }
        }
    }

    with pytest.raises((TypeError, ValueError), match="sample scale"):
        _validate_with_calibration_recipe(params)


def test_derive_calibration_normalizes_unit_and_requires_physical_domain() -> None:
    calibration_signal = _frame(np.array([[0.5, -0.5]]), ["microphone"])

    calibrations = calibration_signal.derive_calibration(target_level=94.0, unit=" Pa ")

    assert calibrations["microphone"].unit == "Pa"
    assert calibrations["microphone"].ref == 2e-5
    with pytest.raises(ValueError, match="Invalid calibration unit"):
        calibration_signal.derive_calibration(target_rms=1.0, unit="")


def test_separately_recorded_sound_and_acceleration_calibrations_apply_together() -> None:
    microphone_calibration_signal = _frame(
        np.array([[0.5, -0.5, 0.5, -0.5]]),
        ["microphone"],
    )
    accelerometer_calibration_signal = _frame(
        np.array([[0.25, -0.25, 0.25, -0.25]]),
        ["accelerometer"],
    )
    measurement = _frame(
        np.array(
            [
                [1.0, -1.0],
                [2.0, -2.0],
            ]
        ),
        ["microphone", "accelerometer"],
    )

    calibrations = {
        **microphone_calibration_signal.derive_calibration(target_level=94.0, unit="Pa"),
        **accelerometer_calibration_signal.derive_calibration(target_rms=1.0, unit="m/s^2", ref=1.0),
    }
    calibrated = measurement.with_calibration(calibrations)

    microphone_factor = 2e-5 * 10 ** (94.0 / 20.0) / 0.5
    np.testing.assert_allclose(
        calibrated.data,
        np.array(
            [
                [microphone_factor, -microphone_factor],
                [8.0, -8.0],
            ]
        ),
        rtol=1e-12,
        atol=0.0,
    )
    assert [channel.unit for channel in calibrated.channels] == ["Pa", "m/s^2"]
    np.testing.assert_array_equal(measurement.data, [[1.0, -1.0], [2.0, -2.0]])
    assert calibrated.operation_history[-1]["operation"] == "wandas.channel.with_calibration"


def test_derive_calibration_rejects_ambiguous_or_already_calibrated_signal() -> None:
    duplicate_labels = _frame(np.ones((2, 4)), ["sensor", "sensor"])
    already_calibrated = _frame(np.ones((1, 4)), ["sensor"]).with_calibration([ChannelCalibration(2.0, "Pa")])

    with pytest.raises(ValueError, match="unique non-empty channel labels"):
        duplicate_labels.derive_calibration(target_rms=1.0, unit="Pa")
    with pytest.raises(ValueError, match="already has calibration factors"):
        already_calibrated.derive_calibration(target_rms=1.0, unit="Pa")


def test_derive_calibration_rejects_reference_that_consumed_prior_calibration() -> None:
    raw_reference = _frame(np.array([[0.5, -0.5, 0.5, -0.5]]), ["microphone"])
    consumed_reference = raw_reference.with_calibration([ChannelCalibration(2.0, "Pa")]).remove_dc()

    assert consumed_reference.channels[0].calibration.factor == 1.0
    assert [record["operation"] for record in consumed_reference.operation_history] == [
        "wandas.channel.with_calibration",
        "wandas.audio.remove_dc",
    ]
    with pytest.raises(ValueError, match="unprocessed source Frame"):
        consumed_reference.derive_calibration(target_rms=1.0, unit="Pa")


def test_derive_calibration_rejects_processed_source_with_calibration_metadata() -> None:
    source_reference = ChannelFrame(
        data=da.from_array(np.array([[0.5, -0.5, 0.5, -0.5]]), chunks=(1, -1)),
        sampling_rate=8_000,
        channel_metadata=[
            ChannelMetadata(
                label="microphone",
                calibration=ChannelCalibration(2.0, "Pa"),
            )
        ],
    )
    consumed_reference = source_reference.remove_dc()

    assert consumed_reference.channels[0].calibration.factor == 1.0
    assert [record["operation"] for record in consumed_reference.operation_history] == ["wandas.audio.remove_dc"]
    with pytest.raises(ValueError, match="unprocessed source Frame"):
        consumed_reference.derive_calibration(target_rms=1.0, unit="Pa")
