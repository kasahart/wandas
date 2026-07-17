"""Tests for WDF (Wandas Data File) I/O functionality."""

import json
from pathlib import Path
from unittest.mock import PropertyMock, patch

import dask.array
import h5py
import numpy as np
import pytest
from scipy.io import wavfile

import wandas as wd
from tests.io_helpers import mock_urlopen_stream
from wandas.frames.channel import ChannelFrame
from wandas.io import readers as io_readers
from wandas.io import wdf_io
from wandas.pipeline import RecipePlan


def _write_minimal_wdf(path: Path, **attrs: object) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        for key, value in attrs.items():
            f.attrs[key] = value
        channels = f.create_group("channels")
        channel = channels.create_group("0")
        channel.create_dataset("data", data=np.zeros(4, dtype=np.float32))
        channel.attrs["label"] = "mic"
        channel.attrs["unit"] = ""


def test_wdf_roundtrip_known_signal(known_signal_frame, tmp_path: Path) -> None:
    """WDF round-trip with known_signal_frame: data, SR, labels preserved (I/O Policy).

    WDF is Wandas' native format — full metadata round-trip is mandatory.
    Uses assert_allclose with rtol=1e-6 for format conversion tolerance.
    """
    path = tmp_path / "known_signal.wdf"
    known_signal_frame.save(path)

    loaded = ChannelFrame.load(path)

    # Verify numerical data (rtol=1e-6: WDF HDF5 round-trip tolerance)
    np.testing.assert_allclose(loaded.compute(), known_signal_frame.compute(), rtol=1e-6)
    # Verify sampling rate
    assert loaded.sampling_rate == known_signal_frame.sampling_rate
    # Verify channel labels
    assert loaded._channel_metadata[0].label == "left"
    assert loaded._channel_metadata[1].label == "right"
    # Verify Dask lazy loading (Pillar 1)
    assert isinstance(loaded._data, dask.array.core.Array)


def test_wdf_roundtrip_preserves_data_and_channel_calibration(tmp_path: Path) -> None:
    raw = np.array([[1.0, 2.0], [3.0, 4.0]])
    frame = ChannelFrame.from_numpy(raw, 8_000, ch_labels=["microphone", "accelerometer"])
    frame._set_channel_calibration(
        0,
        wd.ChannelCalibration(sample_scale="audio-normalized-float"),
    )
    configured = frame.with_calibration(
        [
            wd.ChannelCalibration(0.02, "Pa", sample_scale="audio-normalized-float"),
            wd.ChannelCalibration(9.81, "m/s^2", 1.0),
        ]
    )
    path = tmp_path / "calibrated.wdf"

    configured.save(path)
    with h5py.File(path, "r") as f:
        assert f.attrs["version"] == "0.3"
        assert f["channels"]["0"].attrs["calibration_factor"] == 0.02
        assert f["channels"]["0"].attrs["calibration_sample_scale"] == "audio-normalized-float"
    loaded = ChannelFrame.load(path)

    np.testing.assert_array_equal(loaded._data.compute(), raw)
    np.testing.assert_array_equal(loaded.data, configured.data)
    assert [channel.calibration for channel in loaded.channels] == [
        wd.ChannelCalibration(0.02, "Pa", sample_scale="audio-normalized-float"),
        wd.ChannelCalibration(9.81, "m/s^2", 1.0),
    ]
    assert loaded.operation_history == configured.operation_history


def test_wdf_roundtrip_preserves_source_time_offset(known_signal_frame, tmp_path: Path) -> None:
    """WDF stores and restores source_time_offset as frame state."""
    known_signal_frame.source_time_offset = [3.25, 7.5]
    path = tmp_path / "source_time_offset.wdf"

    known_signal_frame.save(path)
    with h5py.File(path, "r") as f:
        assert "source_time_offset" not in f.attrs
        assert f["channels"]["0"].attrs["source_time_offset"] == 3.25
        assert f["channels"]["1"].attrs["source_time_offset"] == 7.5

    loaded = ChannelFrame.load(path)

    np.testing.assert_array_equal(loaded.source_time_offset, np.array([3.25, 7.5]))
    np.testing.assert_array_equal(loaded.source_time[:, 0], np.array([3.25, 7.5]))


def test_wdf_load_channel_source_time_offset_takes_priority_over_root(tmp_path: Path) -> None:
    """New channel attr offsets take priority over legacy root offsets."""
    path = tmp_path / "channel_offset_priority.wdf"
    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        f.attrs["label"] = ""
        f.attrs["source_time_offset"] = np.array([99.0, 100.0])
        channels = f.create_group("channels")
        for i, offset in enumerate([1.25, 2.5]):
            channel = channels.create_group(str(i))
            channel.create_dataset("data", data=np.zeros(4, dtype=np.float32))
            channel.attrs["label"] = f"mic{i}"
            channel.attrs["unit"] = ""
            channel.attrs["source_time_offset"] = offset

    loaded = ChannelFrame.load(path)

    np.testing.assert_array_equal(loaded.source_time_offset, np.array([1.25, 2.5]))


def test_wdf_load_legacy_root_source_time_offset(tmp_path: Path) -> None:
    """Legacy root source_time_offset remains readable."""
    path = tmp_path / "legacy_root_offset.wdf"
    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        f.attrs["label"] = ""
        f.attrs["source_time_offset"] = np.array([4.0, 8.0])
        channels = f.create_group("channels")
        for i in range(2):
            channel = channels.create_group(str(i))
            channel.create_dataset("data", data=np.zeros(4, dtype=np.float32))
            channel.attrs["label"] = f"mic{i}"
            channel.attrs["unit"] = ""

    loaded = ChannelFrame.load(path)

    np.testing.assert_array_equal(loaded.source_time_offset, np.array([4.0, 8.0]))


def test_wdf_load_defaults_missing_source_time_offset_to_zero(tmp_path: Path) -> None:
    """Legacy WDF files without source_time_offset load with zero offset."""
    path = tmp_path / "legacy_no_offset.wdf"
    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        f.attrs["label"] = ""
        channels = f.create_group("channels")
        channel = channels.create_group("0")
        channel.create_dataset("data", data=np.zeros(4, dtype=np.float32))
        channel.attrs["label"] = "mic0"
        channel.attrs["unit"] = ""

    loaded = ChannelFrame.load(path)

    np.testing.assert_array_equal(loaded.source_time_offset, np.array([0.0]))


def test_wdf_load_rejects_non_finite_source_time_offset(tmp_path: Path) -> None:
    """Invalid persisted source_time_offset values are rejected on load."""
    path = tmp_path / "invalid_offset.wdf"
    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        f.attrs["label"] = ""
        f.attrs["source_time_offset"] = np.nan
        channels = f.create_group("channels")
        channel = channels.create_group("0")
        channel.create_dataset("data", data=np.zeros(4, dtype=np.float32))
        channel.attrs["label"] = "mic0"
        channel.attrs["unit"] = ""

    with pytest.raises(ValueError, match="source_time_offset must be finite"):
        ChannelFrame.load(path)


def test_wdf_load_rejects_source_time_offset_length_mismatch(tmp_path: Path) -> None:
    """Persisted source_time_offset arrays must match WDF channel count."""
    path = tmp_path / "invalid_offset_length.wdf"
    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        f.attrs["label"] = ""
        f.attrs["source_time_offset"] = np.array([1.0, 2.0])
        channels = f.create_group("channels")
        channel = channels.create_group("0")
        channel.create_dataset("data", data=np.zeros(4, dtype=np.float32))
        channel.attrs["label"] = "mic0"
        channel.attrs["unit"] = ""

    with pytest.raises(ValueError, match="source_time_offset length must match number of channels"):
        ChannelFrame.load(path)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Test saving and loading a ChannelFrame with full metadata preservation."""
    # Seeded RNG for reproducibility (Grand Policy: no random data)
    rng = np.random.default_rng(42)
    sr = 48000
    data = rng.standard_normal((2, sr))

    # Create ChannelFrame with metadata
    cf = ChannelFrame.from_numpy(
        data,
        sr,
        label="Test Frame",
        metadata={"test_key": "test_value"},
        ch_labels=["Left", "Right"],
        ch_units=["Pa", "Pa"],
    )

    # Set additional metadata on channels
    cf._channel_metadata[0].extra["sensitivity"] = 50.0
    cf._channel_metadata[1].extra["sensitivity"] = 48.5

    # Save to file
    path = tmp_path / "test_roundtrip.wdf"
    cf.save(path)

    # Reload from file
    cf2 = ChannelFrame.load(path)

    # Verify basic properties
    assert cf2.sampling_rate == sr, f"Sampling rate mismatch: {cf2.sampling_rate} != {sr}"
    assert cf2.n_channels == 2, f"Channel count mismatch: {cf2.n_channels} != 2"
    assert cf2.label == "Test Frame", f"Label mismatch: {cf2.label}"
    assert cf2.metadata.get("test_key") == "test_value", "Custom metadata key not preserved"

    # A source frame has no operation history to persist.
    assert cf2.operation_history == []

    # Verify channel data — HDF5 round-trip with same dtype: exact match expected
    np.testing.assert_array_equal(cf2.data, cf.data)

    # Verify Dask lazy loading preserved after WDF load (Pillar 1)
    assert isinstance(cf2._data, dask.array.core.Array), "WDF load must produce Dask array"

    # Verify channel metadata
    assert cf2._channel_metadata[0].label == "Left"
    assert cf2._channel_metadata[0].unit == "Pa"
    assert cf2._channel_metadata[0].extra.get("sensitivity") == 50.0

    assert cf2._channel_metadata[1].label == "Right"
    assert cf2._channel_metadata[1].unit == "Pa"
    assert cf2._channel_metadata[1].extra.get("sensitivity") == 48.5


def test_wdf_roundtrip_preserves_stable_channel_ids(tmp_path: Path) -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    frame = ChannelFrame(
        data=dask.array.from_array(data, chunks=(1, -1)),
        sampling_rate=2.0,
        channel_metadata=[
            {"label": "left", "unit": "Pa", "extra": {"sensor": "a"}},
            {"label": "right", "unit": "Pa", "extra": {"sensor": "b"}},
        ],
        channel_ids=["mic-a", "mic-b"],
    )
    path = tmp_path / "stable_ids.wdf"

    frame.save(path)
    loaded = ChannelFrame.load(path)

    assert loaded._xr.coords["channel"].values.tolist() == ["mic-a", "mic-b"]
    assert loaded._xr.attrs["channel_extra"] == {
        "mic-a": {"sensor": "a"},
        "mic-b": {"sensor": "b"},
    }


def test_wdf_roundtrips_operation_history_as_source_display_prefix(known_signal_frame, tmp_path: Path) -> None:
    """WDF restores canonical history for display without making it executable."""
    path = tmp_path / "operation_history.wdf"
    processed = known_signal_frame.normalize().low_pass_filter(cutoff=1000, order=4)
    expected_history = processed.operation_history

    processed.save(path)

    loaded = ChannelFrame.load(path)

    assert loaded.operation_history == expected_history
    assert [record["operation"] for record in loaded.operation_history] == [
        "wandas.audio.normalize",
        "wandas.audio.lowpass_filter",
    ]
    assert loaded.lineage.operation is None


def test_wdf_loaded_history_prefix_is_not_recipe_lineage(known_signal_frame, tmp_path: Path) -> None:
    """WDF load keeps the history prefix inspection-only and outside Recipe lineage."""
    path = tmp_path / "snapshot_not_lineage.wdf"
    processed = known_signal_frame.normalize().low_pass_filter(cutoff=1000, order=4)
    expected_history = processed.operation_history

    processed.save(path)
    loaded = ChannelFrame.load(path)

    assert loaded.operation_history == expected_history
    assert loaded.lineage.operation is None
    assert RecipePlan.from_frame(loaded).nodes == ()

    followup = loaded.high_pass_filter(cutoff=200, order=4)

    assert [record["operation"] for record in followup.operation_history] == [
        "wandas.audio.normalize",
        "wandas.audio.lowpass_filter",
        "wandas.audio.highpass_filter",
    ]
    assert [node.operation for node in RecipePlan.from_frame(followup).nodes] == ["wandas.audio.highpass_filter"]


def test_wdf_repeated_save_load_preserves_composed_history_order(tmp_path: Path) -> None:
    """Repeated WDF boundaries keep composed history records once and in order."""
    first_path = tmp_path / "snapshot_cycle_1.wdf"
    second_path = tmp_path / "snapshot_cycle_2.wdf"
    first = ChannelFrame.from_numpy(np.linspace(-1.0, 1.0, 64).reshape(1, -1), 16000).normalize()
    expected_after_first_load = first.operation_history

    first.save(first_path)
    loaded = ChannelFrame.load(first_path)
    second = loaded.low_pass_filter(cutoff=1000, order=4)
    expected_after_followup = second.operation_history
    second.save(second_path)
    reloaded = ChannelFrame.load(second_path)

    assert loaded.operation_history == expected_after_first_load
    assert [record["operation"] for record in second.operation_history] == [
        "wandas.audio.normalize",
        "wandas.audio.lowpass_filter",
    ]
    assert reloaded.operation_history == expected_after_followup


def test_wdf_roundtrips_custom_operation_history(tmp_path: Path) -> None:
    def scale(x: np.ndarray, gain: float) -> np.ndarray:
        return x * gain

    path = tmp_path / "custom_history.wdf"
    processed = ChannelFrame.from_numpy(np.array([[1.0, -1.0, 0.5]]), 16000).apply(scale, gain=2.0)
    expected_history = processed.operation_history

    processed.save(path)
    loaded = ChannelFrame.load(path)

    assert loaded.operation_history == expected_history
    assert loaded.operation_history == [{"operation": "wandas.custom.apply", "version": 1, "params": {"gain": 2.0}}]


def test_wdf_roundtrips_multi_input_operation_history(tmp_path: Path) -> None:
    path = tmp_path / "multi_input_history.wdf"
    left = ChannelFrame.from_numpy(np.array([[1.0, -1.0, 0.5]]), 16000).normalize()
    right = ChannelFrame.from_numpy(np.array([[0.25, 0.5, -0.25]]), 16000).remove_dc()
    processed = left + right
    expected_history = processed.operation_history

    processed.save(path)
    loaded = ChannelFrame.load(path)

    assert loaded.operation_history == expected_history
    assert [record["operation"] for record in loaded.operation_history] == [
        "wandas.audio.normalize",
        "wandas.audio.remove_dc",
        "wandas.operator.add",
    ]


def test_wdf_loaded_history_prefix_extends_across_followup_operations(tmp_path: Path) -> None:
    path = tmp_path / "loaded_followup.wdf"
    loaded = ChannelFrame.from_numpy(np.array([[1.0, -1.0, 0.5]]), 16000).normalize()
    loaded.save(path)

    result = ChannelFrame.load(path).low_pass_filter(cutoff=1000, order=4)

    assert [record["operation"] for record in result.operation_history] == [
        "wandas.audio.normalize",
        "wandas.audio.lowpass_filter",
    ]


def test_wdf_loaded_history_prefix_extends_across_binary_followup(tmp_path: Path) -> None:
    left_path = tmp_path / "loaded_left.wdf"
    right_path = tmp_path / "loaded_right.wdf"
    left = ChannelFrame.from_numpy(np.array([[1.0, -1.0, 0.5]]), 16000).normalize()
    right = ChannelFrame.from_numpy(np.array([[0.25, 0.5, -0.25]]), 16000).remove_dc()
    left.save(left_path)
    right.save(right_path)

    result = ChannelFrame.load(left_path) + ChannelFrame.load(right_path)

    assert [record["operation"] for record in result.operation_history] == [
        "wandas.audio.normalize",
        "wandas.audio.remove_dc",
        "wandas.operator.add",
    ]


def test_wdf_loaded_history_prefix_extends_after_channel_update(tmp_path: Path) -> None:
    source_path = tmp_path / "loaded_source.wdf"
    renamed_path = tmp_path / "loaded_renamed.wdf"
    processed = ChannelFrame.from_numpy(np.array([[1.0, -1.0, 0.5]]), 16000).normalize()
    processed.save(source_path)

    loaded = ChannelFrame.load(source_path)
    renamed = loaded.rename_channels({0: "renamed"})
    renamed.save(renamed_path)
    reloaded = ChannelFrame.load(renamed_path)

    assert [record["operation"] for record in loaded.operation_history] == ["wandas.audio.normalize"]
    assert [record["operation"] for record in renamed.operation_history] == [
        "wandas.audio.normalize",
        "wandas.channel.rename_channels",
    ]
    assert reloaded.operation_history == renamed.operation_history


def test_wdf_loaded_history_prefix_extends_across_domain_transition(tmp_path: Path) -> None:
    source_path = tmp_path / "loaded_stft_source.wdf"
    processed = ChannelFrame.from_numpy(np.linspace(-1.0, 1.0, 16).reshape(1, -1), 16000).normalize()
    processed.save(source_path)

    result = ChannelFrame.load(source_path).stft(n_fft=8, hop_length=4)

    assert [record["operation"] for record in result.operation_history] == [
        "wandas.audio.normalize",
        "wandas.audio.stft",
    ]


def test_add_channel_merges_loaded_frame_history_prefix(tmp_path: Path) -> None:
    added_path = tmp_path / "loaded_added_channel.wdf"
    added = ChannelFrame.from_numpy(np.array([[1.0, -1.0, 0.5]]), 16000).normalize()
    added.save(added_path)
    base = ChannelFrame.from_numpy(np.array([[0.25, 0.5, -0.25]]), 16000).remove_dc()

    result = base.add_channel(ChannelFrame.load(added_path))

    assert [record["operation"] for record in result.operation_history] == [
        "wandas.audio.remove_dc",
        "wandas.audio.normalize",
        "wandas.channel.add_channel",
    ]


def test_wdf_loaded_history_prefix_extends_through_inverse_stft(tmp_path: Path) -> None:
    source_path = tmp_path / "loaded_istft_source.wdf"
    processed = ChannelFrame.from_numpy(np.linspace(-1.0, 1.0, 16).reshape(1, -1), 16000).normalize()
    processed.save(source_path)

    result = ChannelFrame.load(source_path).stft(n_fft=8, hop_length=4).istft()

    assert [record["operation"] for record in result.operation_history] == [
        "wandas.audio.normalize",
        "wandas.audio.stft",
        "wandas.spectrogram.to_channel_frame",
    ]


def test_wdf_loaded_history_prefix_extends_through_stft_frame_extraction(tmp_path: Path) -> None:
    source_path = tmp_path / "loaded_stft_frame_source.wdf"
    processed = ChannelFrame.from_numpy(np.linspace(-1.0, 1.0, 16).reshape(1, -1), 16000).normalize()
    processed.save(source_path)

    result = ChannelFrame.load(source_path).stft(n_fft=8, hop_length=4).get_frame_at(0)

    assert [record["operation"] for record in result.operation_history] == [
        "wandas.audio.normalize",
        "wandas.audio.stft",
        "wandas.spectrogram.get_frame_at",
    ]


def test_wdf_loaded_history_prefix_extends_through_inverse_fft(tmp_path: Path) -> None:
    source_path = tmp_path / "loaded_ifft_source.wdf"
    processed = ChannelFrame.from_numpy(np.linspace(-1.0, 1.0, 16).reshape(1, -1), 16000).normalize()
    processed.save(source_path)

    result = ChannelFrame.load(source_path).fft(n_fft=16).ifft()

    assert [record["operation"] for record in result.operation_history] == [
        "wandas.audio.normalize",
        "wandas.audio.fft",
        "wandas.spectral.ifft",
    ]


def test_mix_with_snr_merges_loaded_frame_history_prefix(tmp_path: Path) -> None:
    noise_path = tmp_path / "loaded_snr_noise.wdf"
    noise = ChannelFrame.from_numpy(np.array([[1.0, -1.0, 0.5]]), 16000).normalize()
    noise.save(noise_path)
    signal = ChannelFrame.from_numpy(np.array([[0.25, 0.5, -0.25]]), 16000).remove_dc()

    result = signal.mix(ChannelFrame.load(noise_path), snr_db=12.0)

    assert [record["operation"] for record in result.operation_history] == [
        "wandas.audio.remove_dc",
        "wandas.audio.normalize",
        "wandas.audio.mix",
    ]
    assert result.operation_history[-1]["params"] == {"snr_db": 12.0}


def test_mix_with_snr_excludes_implicit_padding_from_loaded_history_prefix(tmp_path: Path) -> None:
    noise_path = tmp_path / "loaded_short_snr_noise.wdf"
    noise = ChannelFrame.from_numpy(np.array([[1.0, -1.0]]), 16000).normalize()
    noise.save(noise_path)
    signal = ChannelFrame.from_numpy(np.array([[0.25, 0.5, -0.25, 0.75]]), 16000).remove_dc()

    result = signal.mix(ChannelFrame.load(noise_path), align="pad", snr_db=12.0)

    assert [record["operation"] for record in result.operation_history] == [
        "wandas.audio.remove_dc",
        "wandas.audio.normalize",
        "wandas.audio.mix",
    ]
    assert result.operation_history[-1]["params"] == {"align": "pad", "snr_db": 12.0}


def test_save_wdf_dtype_float32_converts_stored_data(tmp_path: Path) -> None:
    """Test saving with dtype conversion."""
    rng = np.random.default_rng(1)
    sr = 44100
    data = rng.standard_normal((1, sr)).astype(np.float64)
    cf = ChannelFrame.from_numpy(data, sr)

    # Save with float32 dtype
    path = tmp_path / "test_dtype.wdf"
    cf.save(path, dtype="float32")

    # Verify dtype in saved file
    with h5py.File(path, "r") as f:
        assert f["channels/0/data"].dtype == np.dtype("float32")


@pytest.mark.parametrize("dtype", ["int8", "complex64", "object"])
def test_save_wdf_rejects_invalid_dtype_for_sample_scale_provenance(tmp_path: Path, dtype: str) -> None:
    wav_path = tmp_path / "raw.wav"
    wavfile.write(wav_path, 8_000, np.array([16_384, -16_384], dtype=np.int16))
    frame = ChannelFrame.read_wav(wav_path, labels=["microphone"], normalize=False)
    path = tmp_path / f"invalid-{dtype}.wdf"

    with pytest.raises(ValueError, match="invalidate calibration sample scale"):
        frame.save(path, dtype=dtype)

    assert not path.exists()


def test_save_wdf_allows_safe_widening_with_sample_scale_provenance(tmp_path: Path) -> None:
    wav_path = tmp_path / "raw.wav"
    wavfile.write(wav_path, 8_000, np.array([16_384, -16_384], dtype=np.int16))
    frame = ChannelFrame.read_wav(wav_path, labels=["microphone"], normalize=False)

    safe_path = tmp_path / "widened.wdf"
    frame.save(safe_path, dtype="float64")
    loaded = ChannelFrame.load(safe_path)
    calibrated = loaded.with_calibration(
        {
            "microphone": wd.ChannelCalibration(
                factor=1.0 / 16_384.0,
                unit="Pa",
                sample_scale="wav-native-pcm_16",
            )
        }
    )

    assert loaded._data.dtype == np.dtype("float64")
    assert loaded.channels[0].calibration.sample_scale == "wav-native-pcm_16"
    np.testing.assert_array_equal(loaded._data.compute(), frame._data.compute())
    np.testing.assert_allclose(calibrated.data, np.array([1.0, -1.0]), atol=1e-12)


def test_save_wdf_no_compression_stores_uncompressed(tmp_path: Path) -> None:
    """Test saving without compression."""
    rng = np.random.default_rng(2)
    sr = 22050
    data = rng.standard_normal((1, sr))
    cf = ChannelFrame.from_numpy(data, sr)

    path = tmp_path / "test_no_compress.wdf"
    cf.save(path, compress=None)

    # Verify that no compression was used
    with h5py.File(path, "r") as f:
        assert f["channels/0/data"].compression is None


def test_save_wdf_existing_file_overwrite_false_raises_file_exists(tmp_path: Path) -> None:
    """Test that attempting to overwrite without overwrite=True raises an error."""
    rng = np.random.default_rng(3)
    sr = 8000
    data = rng.standard_normal((1, sr))
    cf = ChannelFrame.from_numpy(data, sr)

    path = tmp_path / "test_exists.wdf"
    cf.save(path)
    assert path.exists(), "First save must create the file"

    # Second save without overwrite should fail (I/O Policy: overwrite protection)
    with pytest.raises(FileExistsError):
        cf.save(path, overwrite=False)

    # Second save with overwrite=True should succeed
    cf.save(path, overwrite=True)
    assert path.exists(), "Overwrite must keep the file"


def test_save_wdf_no_extension_adds_wdf_suffix(tmp_path: Path) -> None:
    """Test that .wdf extension is automatically added."""
    rng = np.random.default_rng(4)
    sr = 16000
    data = rng.standard_normal((1, sr))
    cf = ChannelFrame.from_numpy(data, sr)

    path = tmp_path / "test_file"  # No extension
    cf.save(path)

    # Should have added .wdf extension
    assert (tmp_path / "test_file.wdf").exists(), ".wdf extension must be auto-appended"


def test_save_load_unsupported_format_raises_not_implemented() -> None:
    """Test that unsupported formats raise NotImplementedError."""
    rng = np.random.default_rng(5)
    sr = 16000
    data = rng.standard_normal((1, sr))
    cf = ChannelFrame.from_numpy(data, sr)

    with pytest.raises(NotImplementedError):
        cf.save("test.wdf", format="unsupported")

    with pytest.raises(NotImplementedError):
        ChannelFrame.load("test.wdf", format="unsupported")


def test_save_wdf_missing_h5py_does_not_compute(tmp_path: Path) -> None:
    """Missing h5py fails before materializing the frame data."""

    class UncomputableFrame:
        def compute(self) -> np.ndarray:
            raise AssertionError("save() computed data before checking h5py")

    with patch.object(wdf_io, "require_h5py", side_effect=ImportError('Install it with: pip install "wandas[io]"')):
        with pytest.raises(ImportError, match=r"wandas\[io\]"):
            wdf_io.save(UncomputableFrame(), tmp_path / "missing_h5py.wdf")  # ty: ignore[invalid-argument-type]


def test_load_wdf_modified_version_still_loads(tmp_path: Path) -> None:
    """Test version handling in WDF files."""
    rng = np.random.default_rng(6)
    sr = 8000
    data = rng.standard_normal((1, sr))
    cf = ChannelFrame.from_numpy(data, sr)

    path = tmp_path / "test_version.wdf"
    cf.save(path)

    # Modify the version in the file
    with h5py.File(path, "r+") as f:
        f.attrs["version"] = "0.2"

    # Should still load but log a warning
    cf2 = ChannelFrame.load(path)
    assert cf2.n_samples == sr, f"Expected {sr} samples after version-modified load, got {cf2.n_samples}"


def test_save_wdf_does_not_create_operation_history_group(tmp_path: Path) -> None:
    """Current WDF stores operation history in root attrs, not an HDF5 group."""
    rng = np.random.default_rng(7)
    sr = 16000
    data = rng.standard_normal((1, sr))
    cf = ChannelFrame.from_numpy(data, sr).normalize()

    path = tmp_path / "test_non_serializable.wdf"
    wdf_io.save(cf, path)

    with h5py.File(path, "r") as f:
        assert "operation_history" not in f


def test_save_wdf_writes_operation_history_json(tmp_path: Path) -> None:
    frame = ChannelFrame.from_numpy(np.array([[1.0, -1.0, 0.5]]), 16000).normalize()
    path = tmp_path / "history.wdf"

    wdf_io.save(frame, path)

    with h5py.File(path, "r") as f:
        assert f.attrs["version"] == "0.3"
        assert f.attrs[wdf_io.OPERATION_HISTORY_SCHEMA_ATTR] == wdf_io.OPERATION_HISTORY_SCHEMA_VERSION
        history = json.loads(f.attrs[wdf_io.OPERATION_HISTORY_JSON_ATTR])
    assert history == frame.operation_history
    assert history == [{"operation": "wandas.audio.normalize", "version": 1, "params": {}}]


def test_save_wdf_operation_history_json_is_strict_json(tmp_path: Path) -> None:
    frame = ChannelFrame.from_numpy(np.array([[1.0, -1.0, 0.5]]), 16000).normalize()
    path = tmp_path / "strict_history.wdf"

    wdf_io.save(frame, path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-strict JSON constant: {value}")

    with h5py.File(path, "r") as f:
        json.loads(
            f.attrs[wdf_io.OPERATION_HISTORY_JSON_ATTR],
            parse_constant=reject_constant,
        )


def test_save_wdf_validates_operation_history_before_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "existing.wdf"
    original = ChannelFrame.from_numpy(np.array([[1.0, 2.0, 3.0]]), 16000)
    original.save(path)
    candidate = ChannelFrame.from_numpy(np.array([[4.0, 5.0, 6.0]]), 16000)

    with patch.object(type(candidate), "operation_history", new_callable=PropertyMock) as history:
        history.side_effect = ValueError("bad history")
        with pytest.raises(ValueError, match="bad history"):
            wdf_io.save(candidate, path, overwrite=True)

    loaded = ChannelFrame.load(path)

    np.testing.assert_array_equal(loaded.compute(), original.compute())


def test_load_no_channels(tmp_path: Path) -> None:
    """Test loading a file with no channels raises ValueError."""
    path = tmp_path / "empty_channels.wdf"

    # Create a dummy HDF5 file with no channels
    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        f.create_group("channels")  # Empty group

    with pytest.raises(ValueError, match="No channel data found"):
        wdf_io.load(path)


def test_load_wdf_restores_operation_history_as_source_prefix(tmp_path: Path) -> None:
    path = tmp_path / "manual_history.wdf"
    history = [{"operation": "wandas.test.loaded", "version": 1, "params": {"gain": 2.0}}]

    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        f.attrs[wdf_io.OPERATION_HISTORY_SCHEMA_ATTR] = wdf_io.OPERATION_HISTORY_SCHEMA_VERSION
        f.attrs[wdf_io.OPERATION_HISTORY_JSON_ATTR] = json.dumps(history)
        channels = f.create_group("channels")
        channel = channels.create_group("0")
        channel.create_dataset("data", data=np.zeros(4, dtype=np.float32))
        channel.attrs["label"] = "mic"
        channel.attrs["unit"] = ""

    loaded = wdf_io.load(path)

    assert loaded.operation_history == history
    assert loaded.lineage.operation is None
    assert RecipePlan.from_frame(loaded).nodes == ()


def test_load_wdf_migrates_operation_summaries_json_to_history_prefix(tmp_path: Path) -> None:
    path = tmp_path / "legacy_summaries.wdf"
    summaries = [
        {"operation": "normalize", "params": {"axis": -1}},
        {
            "operation": "custom",
            "params": {"gain": 2.0},
            "implementation": {"module": "example", "qualname": "gain"},
        },
    ]
    _write_minimal_wdf(
        path,
        version="0.1",
        operation_summaries_schema=1,
        operation_summaries_json=json.dumps(summaries),
    )

    loaded = wdf_io.load(path)

    assert loaded.operation_history == [
        {"operation": "normalize", "version": 1, "params": {"axis": -1}},
        {
            "operation": "custom",
            "version": 1,
            "params": {
                "gain": 2.0,
                "implementation": {"module": "example", "qualname": "gain"},
            },
        },
    ]
    assert loaded.lineage.operation is None
    assert RecipePlan.from_frame(loaded).nodes == ()


def test_load_wdf_migrates_legacy_operation_history_group(tmp_path: Path) -> None:
    path = tmp_path / "legacy_history.wdf"
    with h5py.File(path, "w") as f:
        f.attrs["version"] = "0.1"
        f.attrs["sampling_rate"] = 16000.0
        channels = f.create_group("channels")
        channel = channels.create_group("0")
        channel.create_dataset("data", data=np.zeros(4, dtype=np.float32))
        history = f.create_group("operation_history")
        first = history.create_group("operation_0")
        first.attrs["operation"] = "normalize"
        first.attrs["params"] = json.dumps({"axis": -1})
        second = history.create_group("operation_1")
        second.attrs["operation"] = "**"
        second.attrs["with"] = "2"
        third = history.create_group("operation_2")
        third.attrs["operation"] = "legacy_custom"
        third.attrs["params"] = "{bad_json"

    loaded = wdf_io.load(path)

    assert loaded.operation_history == [
        {"operation": "normalize", "version": 1, "params": {"axis": -1}},
        {"operation": "**", "version": 1, "params": {"with": 2}},
        {"operation": "legacy_custom", "version": 1, "params": {"legacy_params": "{bad_json"}},
    ]
    assert loaded.lineage.operation is None
    assert RecipePlan.from_frame(loaded).nodes == ()


def test_load_wdf_rejects_unsupported_operation_summaries_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad_summaries_schema.wdf"
    _write_minimal_wdf(
        path,
        version="0.1",
        operation_summaries_schema=999,
        operation_summaries_json=json.dumps([]),
    )

    with pytest.raises(ValueError, match="Unsupported WDF operation summaries schema"):
        wdf_io.load(path)


def test_load_wdf_rejects_invalid_operation_summaries_json_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad_summaries_json.wdf"
    _write_minimal_wdf(
        path,
        version="0.1",
        operation_summaries_schema=1,
        operation_summaries_json=json.dumps({"operation": "loaded"}),
    )

    with pytest.raises(ValueError, match="Invalid WDF operation summaries JSON"):
        wdf_io.load(path)


def test_load_wdf_rejects_non_strict_operation_summaries_json(tmp_path: Path) -> None:
    path = tmp_path / "nan_summaries_json.wdf"
    _write_minimal_wdf(
        path,
        version="0.1",
        operation_summaries_schema=1,
        operation_summaries_json='[{"operation": "loaded", "params": {"gain": NaN}}]',
    )

    with pytest.raises(ValueError, match="strict JSON"):
        wdf_io.load(path)


def test_load_wdf_rejects_malformed_legacy_operation_history_group(tmp_path: Path) -> None:
    path = tmp_path / "malformed_legacy_history.wdf"
    _write_minimal_wdf(path, version="0.1")
    with h5py.File(path, "a") as f:
        history = f.create_group("operation_history")
        record = history.create_group("unexpected")
        record.attrs["operation"] = "normalize"

    with pytest.raises(ValueError, match="Invalid legacy WDF operation history group"):
        wdf_io.load(path)


def test_load_wdf_rejects_conflicting_legacy_history_fields(tmp_path: Path) -> None:
    path = tmp_path / "conflicting_legacy_history.wdf"
    summaries = [{"operation": "custom", "params": {"implementation": "params"}, "implementation": "top"}]
    _write_minimal_wdf(
        path,
        version="0.1",
        operation_summaries_schema=1,
        operation_summaries_json=json.dumps(summaries),
    )

    with pytest.raises(ValueError, match="duplicate field 'implementation'"):
        wdf_io.load(path)


def test_load_wdf_rejects_legacy_history_without_operation_name(tmp_path: Path) -> None:
    path = tmp_path / "unnamed_legacy_history.wdf"
    _write_minimal_wdf(
        path,
        version="0.1",
        operation_summaries_schema=1,
        operation_summaries_json=json.dumps([{"params": {"gain": 2.0}}]),
    )

    with pytest.raises(ValueError, match="non-blank 'operation' string"):
        wdf_io.load(path)


def test_load_wdf_rejects_unsupported_operation_history_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad_history_schema.wdf"

    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        f.attrs[wdf_io.OPERATION_HISTORY_SCHEMA_ATTR] = 999
        f.attrs[wdf_io.OPERATION_HISTORY_JSON_ATTR] = json.dumps([])
        channels = f.create_group("channels")
        channel = channels.create_group("0")
        channel.create_dataset("data", data=np.zeros(4, dtype=np.float32))
        channel.attrs["label"] = "mic"
        channel.attrs["unit"] = ""

    with pytest.raises(ValueError, match="Unsupported WDF operation history schema"):
        wdf_io.load(path)


def test_load_wdf_rejects_invalid_operation_history_json_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad_history_json.wdf"

    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        f.attrs[wdf_io.OPERATION_HISTORY_SCHEMA_ATTR] = wdf_io.OPERATION_HISTORY_SCHEMA_VERSION
        f.attrs[wdf_io.OPERATION_HISTORY_JSON_ATTR] = json.dumps({"operation": "wandas.test.loaded"})
        channels = f.create_group("channels")
        channel = channels.create_group("0")
        channel.create_dataset("data", data=np.zeros(4, dtype=np.float32))
        channel.attrs["label"] = "mic"
        channel.attrs["unit"] = ""

    with pytest.raises(ValueError, match="Invalid WDF operation history JSON"):
        wdf_io.load(path)


def test_load_wdf_rejects_non_strict_operation_history_json(tmp_path: Path) -> None:
    path = tmp_path / "nan_history_json.wdf"
    _write_minimal_wdf(
        path,
        operation_history_schema=1,
        operation_history_json=('[{"operation": "wandas.test.loaded", "version": 1, "params": {"gain": NaN}}]'),
    )

    with pytest.raises(ValueError, match="strict JSON"):
        wdf_io.load(path)


def test_load_file_not_found(tmp_path: Path) -> None:
    """Test loading a non-existent file raises FileNotFoundError."""
    path = tmp_path / "non_existent.wdf"
    with pytest.raises(FileNotFoundError, match="File not found"):
        wdf_io.load(path)


def test_source_file_roundtrip(tmp_path: Path) -> None:
    """Source file metadata survives a save/load round-trip as a plain dict key."""
    source = "audio/input.wav"
    data = np.array([[1, 2, 3]], dtype=np.float32)
    sr = 16000
    cf = ChannelFrame.from_numpy(
        data,
        sr,
        metadata={"key": "val", "_source_file": source},
    )
    assert type(cf.metadata) is dict
    assert cf.metadata["_source_file"] == source

    path = tmp_path / "test_source_file.wdf"
    cf.save(path)
    with h5py.File(path, "r") as f:
        meta = f["meta"]
        saved_metadata = json.loads(meta.attrs["json"])
        assert saved_metadata["_source_file"] == source
        assert meta.attrs["_source_file"] == source
        assert "source_file" not in meta.attrs

    cf2 = ChannelFrame.load(path)

    assert type(cf2.metadata) is dict
    assert cf2.metadata["_source_file"] == source
    assert cf2.metadata["key"] == "val"


def test_source_file_absent_roundtrip(tmp_path: Path) -> None:
    """Round-trip without source file keeps metadata as a plain dict."""
    data = np.array([[1, 2, 3]], dtype=np.float32)
    sr = 16000
    cf = ChannelFrame.from_numpy(data, sr, metadata={"only": "dict"})
    assert type(cf.metadata) is dict
    assert "_source_file" not in cf.metadata

    path = tmp_path / "test_no_source_file.wdf"
    cf.save(path)
    cf2 = ChannelFrame.load(path)

    assert type(cf2.metadata) is dict
    assert cf2.metadata == {"only": "dict"}


def test_from_wdf_legacy_source_file_attr_maps_to_metadata_key(tmp_path: Path) -> None:
    """Legacy meta/source_file attrs populate _source_file when JSON lacks it."""
    path = tmp_path / "legacy_source_file.wdf"
    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        f.attrs["label"] = ""

        channels = f.create_group("channels")
        channel = channels.create_group("0")
        channel.create_dataset("data", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
        channel.attrs["label"] = ""
        channel.attrs["unit"] = ""

        meta = f.create_group("meta")
        meta.attrs["json"] = json.dumps({"key": "val"})
        meta.attrs["source_file"] = "legacy/input.wav"

    loaded = ChannelFrame.load(path)
    assert type(loaded.metadata) is dict
    assert loaded.metadata["key"] == "val"
    assert loaded.metadata["_source_file"] == "legacy/input.wav"


def test_load_wdf_rejects_non_object_metadata_json(tmp_path: Path) -> None:
    """Malformed WDF metadata JSON fails with an actionable error."""
    path = tmp_path / "array_metadata.wdf"
    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        f.attrs["label"] = ""

        channels = f.create_group("channels")
        channel = channels.create_group("0")
        channel.create_dataset("data", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
        channel.attrs["label"] = ""
        channel.attrs["unit"] = ""

        meta = f.create_group("meta")
        meta.attrs["json"] = json.dumps(["not", "a", "dict"])

    with pytest.raises(ValueError, match="WDF meta/json must decode to a JSON object"):
        ChannelFrame.load(path)


@pytest.mark.parametrize("scheme", ["https", "HTTPS"])
def test_load_wdf_from_url(tmp_path: Path, scheme: str) -> None:
    """Test WDF load from mocked URL preserves data and metadata (Pillar 2, 4).

    Verifies urlopen is called, numerical data matches (rtol=1e-5 for float32),
    and label/channel metadata survives the URL download path.
    """
    rng = np.random.default_rng(42)
    sr = 8000
    data = rng.standard_normal((2, sr)).astype(np.float32)
    cf = ChannelFrame.from_numpy(data, sr, label="URL Test", ch_labels=["A", "B"])
    wdf_path = tmp_path / "test_url.wdf"
    cf.save(wdf_path)
    wdf_bytes = wdf_path.read_bytes()

    url = f"{scheme}://example.com/data/test_url.wdf"
    with mock_urlopen_stream(wdf_bytes) as mock_fn:
        cf2 = wdf_io.load(url)

    mock_fn.assert_called_once_with(url, timeout=10.0)
    assert cf2.sampling_rate == sr, f"SR mismatch: {cf2.sampling_rate}"
    assert cf2.n_channels == 2, f"Channel count mismatch: {cf2.n_channels}"
    assert cf2.label == "URL Test", f"Label mismatch: {cf2.label}"
    assert cf2._channel_metadata[0].label == "A", "Channel 0 label not preserved"
    assert cf2._channel_metadata[1].label == "B", "Channel 1 label not preserved"
    # Verify Dask lazy loading from URL path (Pillar 1)
    assert isinstance(cf2._data, dask.array.core.Array), "URL WDF load must produce Dask array"
    # Float32 HDF5 round-trip: rtol=1e-5 for float32 precision
    np.testing.assert_allclose(cf2.compute(), data, rtol=1e-5)


def test_load_wdf_from_url_streams_in_chunks(tmp_path: Path) -> None:
    """URL WDF loads must stream in bounded chunks instead of full-response reads."""
    rng = np.random.default_rng(42)
    sr = 8000
    data = rng.standard_normal((1, sr)).astype(np.float32)
    cf = ChannelFrame.from_numpy(data, sr, label="URL Stream Test", ch_labels=["A"])
    wdf_path = tmp_path / "test_stream_url.wdf"
    cf.save(wdf_path)
    wdf_bytes = wdf_path.read_bytes()

    with mock_urlopen_stream(
        wdf_bytes,
        forbid_unbounded_read=True,
        expected_chunk_size=io_readers.URL_DOWNLOAD_CHUNK_SIZE,
    ):
        loaded = wdf_io.load("https://example.com/data/test_stream_url.wdf")

    assert loaded.sampling_rate == sr
    np.testing.assert_allclose(loaded.compute(), data, rtol=1e-5)


def test_load_wdf_from_url_over_size_limit_without_content_length_raises(tmp_path: Path) -> None:
    """URL WDF loads must stop oversized streamed downloads without Content-Length."""
    rng = np.random.default_rng(42)
    sr = 8000
    data = rng.standard_normal((1, sr)).astype(np.float32)
    cf = ChannelFrame.from_numpy(data, sr, label="URL Stream Test", ch_labels=["A"])
    wdf_path = tmp_path / "test_stream_limit_url.wdf"
    cf.save(wdf_path)
    wdf_bytes = wdf_path.read_bytes()

    with patch.object(io_readers, "MAX_URL_DOWNLOAD_BYTES", 128):
        with mock_urlopen_stream(wdf_bytes, include_content_length=False):
            with pytest.raises(OSError, match=r"Streaming WDF file would exceed size limit"):
                wdf_io.load("https://example.com/data/test_stream_limit_url.wdf")


def test_load_wdf_from_url_declared_size_limit_raises_before_streaming(tmp_path: Path) -> None:
    """URL WDF loads must reject oversized Content-Length before streaming data."""
    rng = np.random.default_rng(42)
    sr = 8000
    data = rng.standard_normal((1, sr)).astype(np.float32)
    cf = ChannelFrame.from_numpy(data, sr, label="Declared Size Test", ch_labels=["A"])
    wdf_path = tmp_path / "test_declared_limit_url.wdf"
    cf.save(wdf_path)
    wdf_bytes = wdf_path.read_bytes()

    with patch.object(io_readers, "MAX_URL_DOWNLOAD_BYTES", 128):
        with mock_urlopen_stream(wdf_bytes, include_content_length=True) as mock_fn:
            with pytest.raises(OSError, match=r"Declared size of WDF file exceeds download limit"):
                wdf_io.load("https://example.com/data/test_declared_limit_url.wdf")

    mock_resp = mock_fn.return_value
    mock_resp.read.assert_not_called()


def test_load_wdf_from_url_invalid_chunk_size_raises(tmp_path: Path) -> None:
    """WDF URL loads must surface helper validation for non-positive chunk size."""
    rng = np.random.default_rng(42)
    sr = 8000
    data = rng.standard_normal((1, sr)).astype(np.float32)
    cf = ChannelFrame.from_numpy(data, sr, label="Chunk Size Test", ch_labels=["A"])
    wdf_path = tmp_path / "test_invalid_chunk_url.wdf"
    cf.save(wdf_path)
    wdf_bytes = wdf_path.read_bytes()

    with mock_urlopen_stream(wdf_bytes):
        with pytest.raises(ValueError, match=r"Download chunk size must be greater than zero"):
            io_readers.download_url_to_temporary_file(
                "https://example.com/data/test_invalid_chunk_url.wdf",
                timeout=10.0,
                suffix=".wdf",
                resource_name="WDF file",
                chunk_size=0,
            )


def test_load_wdf_from_url_handles_partial_reads(tmp_path: Path) -> None:
    """URL WDF loads must continue until EOF even when reads return small chunks."""
    rng = np.random.default_rng(42)
    sr = 8000
    data = rng.standard_normal((1, sr)).astype(np.float32)
    cf = ChannelFrame.from_numpy(data, sr, label="Partial Read Test", ch_labels=["A"])
    wdf_path = tmp_path / "test_partial_reads_url.wdf"
    cf.save(wdf_path)
    wdf_bytes = wdf_path.read_bytes()

    with mock_urlopen_stream(
        wdf_bytes,
        forbid_unbounded_read=True,
        include_content_length=False,
        expected_chunk_size=io_readers.URL_DOWNLOAD_CHUNK_SIZE,
        max_chunk_bytes=257,
    ):
        loaded = wdf_io.load("https://example.com/data/test_partial_reads_url.wdf")

    np.testing.assert_allclose(loaded.compute(), data, rtol=1e-5)


def test_load_wdf_from_url_download_failure() -> None:
    """Test that a URL download failure raises OSError with a clear message."""
    import urllib.error

    url = "https://example.com/data/missing.wdf"

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        with pytest.raises(OSError, match=r"Failed to download WDF file from URL"):
            wdf_io.load(url)


def test_load_wdf_missing_h5py_does_not_download_url() -> None:
    """Missing h5py fails before downloading a remote WDF file."""
    url = "https://example.com/data/large.wdf"

    with (
        patch.object(wdf_io, "require_h5py", side_effect=ImportError('Install it with: pip install "wandas[io]"')),
        patch("urllib.request.urlopen") as mock_urlopen,
    ):
        with pytest.raises(ImportError, match=r"wandas\[io\]"):
            wdf_io.load(url)

    mock_urlopen.assert_not_called()


def test_decode_hdf5_str_invalid_utf8() -> None:
    """_decode_hdf5_str falls back to str() for non-UTF-8 bytes (lines 39-42)."""
    from wandas.io.wdf_io import _decode_hdf5_str

    invalid_bytes = b"\xff\xfe"
    result = _decode_hdf5_str(invalid_bytes)
    # Should return str() fallback rather than raising
    assert isinstance(result, str)
    assert result == str(invalid_bytes)


def test_load_wdf_decodes_channel_label_and_unit_bytes(tmp_path: Path) -> None:
    """Legacy HDF5 byte string channel attrs load as text metadata."""
    path = tmp_path / "byte_channel_attrs.wdf"
    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        f.attrs["label"] = ""

        channels = f.create_group("channels")
        channel = channels.create_group("0")
        channel.create_dataset("data", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
        channel.attrs["label"] = np.bytes_(b"mic0")
        channel.attrs["unit"] = np.bytes_(b"Pa")

    loaded = ChannelFrame.load(path)

    assert loaded.channels[0].label == "mic0"
    assert loaded.channels[0].unit == "Pa"
    assert loaded.channels[0].ref == 2e-5


def test_load_wdf_decodes_frame_label_bytes(tmp_path: Path) -> None:
    """Legacy HDF5 byte string frame labels load as text labels."""
    path = tmp_path / "byte_frame_label.wdf"
    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        f.attrs["label"] = np.bytes_(b"legacy-label")

        channels = f.create_group("channels")
        channel = channels.create_group("0")
        channel.create_dataset("data", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
        channel.attrs["label"] = "mic0"
        channel.attrs["unit"] = "Pa"

    loaded = ChannelFrame.load(path)

    assert loaded.label == "legacy-label"
    assert loaded.channels[0].label == "mic0"


def test_load_wdf_meta_json_as_bytes(tmp_path: Path) -> None:
    """Loading a WDF file where meta JSON is stored as numpy.bytes_ triggers line 229."""
    sr = 16000
    data = np.random.default_rng(0).standard_normal((1, sr)).astype(np.float32)
    # metadata= ensures the save function creates a "meta" HDF5 group
    cf = ChannelFrame.from_numpy(data, sr, label="BytesTest", metadata={"key": "val"})
    wdf_path = tmp_path / "bytes_meta.wdf"
    cf.save(wdf_path)

    # Rewrite the meta JSON attribute as numpy.bytes_ — h5py stores and returns this
    # as numpy.bytes_, which triggers the isinstance(meta_json, (bytes, np.bytes_)) branch
    with h5py.File(wdf_path, "r+") as f:
        if "meta" in f:
            meta_json_str = f["meta"].attrs.get("json", "{}")
            if isinstance(meta_json_str, str):
                del f["meta"].attrs["json"]
                f["meta"].attrs["json"] = np.bytes_(meta_json_str.encode("utf-8"))

    # Load should succeed even when meta_json is numpy.bytes_
    loaded = wdf_io.load(wdf_path)
    assert loaded.sampling_rate == sr
    assert loaded.n_channels == 1
