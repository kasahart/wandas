import dask.array as da
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.frames.roughness import RoughnessFrame


def test_trim_stft_chain_keeps_source_context() -> None:
    data = da.from_array(np.ones((1, 160), dtype=float))
    frame = ChannelFrame(data, sampling_rate=40.0)

    trimmed = frame.trim(1.0, 3.0)
    spec = trimmed.stft(n_fft=16, hop_length=8, win_length=16)

    assert trimmed.source_time_range == pytest.approx((1.0, 3.0))
    assert spec.source_time_offset == pytest.approx(1.0)
    assert spec.duration > 0.0
    assert spec.source_time_range == pytest.approx(trimmed.source_time_range)
    np.testing.assert_allclose(spec.source_times, spec.times + 1.0)


def test_trim_resampling_chain_keeps_consistent_duration_and_context() -> None:
    data = da.from_array(np.arange(100, dtype=float).reshape(1, 100))
    frame = ChannelFrame(data, sampling_rate=10.0)

    result = frame.trim(2.0, 7.0).resampling(20.0)

    assert result.source_time_offset == pytest.approx(2.0)
    assert result.n_samples == 100
    assert result.duration == pytest.approx(5.0)
    assert result.source_time_range == pytest.approx((2.0, 7.0))
    assert result.time[0] == pytest.approx(0.0)
    assert result.source_time[0] == pytest.approx(2.0)


def test_spectrogram_source_time_range_is_bounded_by_previous_frame() -> None:
    data = da.from_array(np.ones((1, 80), dtype=float))
    frame = ChannelFrame(data, sampling_rate=40.0, source_time_offset=1.0)

    spec = frame.stft(n_fft=16, hop_length=8, win_length=16)

    assert spec.duration == pytest.approx(2.2)
    assert frame.source_time_range == pytest.approx((1.0, 3.0))
    assert spec.source_time_range == pytest.approx((1.0, 3.0))


def test_resampling_keeps_source_range_for_non_integer_output_grid() -> None:
    data = da.from_array(np.arange(3, dtype=float).reshape(1, 3), chunks=(1, -1))
    frame = ChannelFrame(data, sampling_rate=10.0, source_time_offset=5.0)

    result = frame.resampling(2.0)

    assert result.n_samples == 1
    assert result.duration == pytest.approx(0.5)
    assert result.source_time_range == pytest.approx((5.0, 5.3))


def test_fix_length_padding_preserves_real_source_range() -> None:
    data = da.from_array(np.ones((1, 4), dtype=float), chunks=(1, -1))
    frame = ChannelFrame(data, sampling_rate=4.0, source_time_offset=5.0)

    result = frame.fix_length(length=8)

    assert result.duration == pytest.approx(2.0)
    assert result.source_time_range == pytest.approx(frame.source_time_range)


def test_fix_length_truncation_shrinks_source_range() -> None:
    data = da.from_array(np.ones((1, 8), dtype=float), chunks=(1, -1))
    frame = ChannelFrame(data, sampling_rate=4.0, source_time_offset=5.0)

    result = frame.fix_length(length=4)

    assert result.source_time_range == pytest.approx((5.0, 6.0))


def test_non_contiguous_channel_time_indexing_is_rejected() -> None:
    data = da.from_array(np.arange(10, dtype=float).reshape(1, 10), chunks=(1, -1))
    frame = ChannelFrame(data, sampling_rate=10.0)

    with pytest.raises(ValueError, match="Non-contiguous time indexing"):
        _ = frame[:, ::2]

    with pytest.raises(ValueError, match="Non-contiguous time indexing"):
        _ = frame[:, [0, 2, 4]]


def test_non_contiguous_spectrogram_time_indexing_is_rejected() -> None:
    data = da.from_array(np.ones((1, 64), dtype=float), chunks=(1, -1))
    spec = ChannelFrame(data, sampling_rate=32.0).stft(n_fft=16, hop_length=8, win_length=16)

    with pytest.raises(ValueError, match="Non-contiguous time indexing"):
        _ = spec[:, :, ::2]

    with pytest.raises(ValueError, match="Non-contiguous time indexing"):
        _ = spec[:, :, [0, 2]]


def test_channel_update_preserves_explicit_source_range() -> None:
    data = da.from_array(np.ones((1, 10), dtype=float), chunks=(1, -1))
    frame = ChannelFrame(data, sampling_rate=10.0).fix_length(length=20)

    result = frame.rename_channels({0: "renamed"})

    assert frame.source_time_range == pytest.approx((0.0, 1.0))
    assert result.source_time_range == pytest.approx(frame.source_time_range)


def test_trim_clips_explicit_source_range_to_selected_samples() -> None:
    data = da.from_array(np.ones((1, 10), dtype=float), chunks=(1, -1))
    frame = ChannelFrame(data, sampling_rate=10.0).fix_length(length=20)

    real_samples = frame.trim(0.0, 0.5)
    padded_samples = frame.trim(1.5, 2.0)

    assert real_samples.source_time_range == pytest.approx((0.0, 0.5))
    assert padded_samples.source_time_range == pytest.approx((1.5, 1.5))


def test_channel_time_indexing_clips_explicit_source_range_to_selected_samples() -> None:
    data = da.from_array(np.ones((1, 10), dtype=float), chunks=(1, -1))
    frame = ChannelFrame(data, sampling_rate=10.0).fix_length(length=20)

    real_samples = frame[:, 0:5]
    padded_samples = frame[:, 15:20]

    assert real_samples.source_time_range == pytest.approx((0.0, 0.5))
    assert padded_samples.source_time_range == pytest.approx((1.5, 1.5))


def test_roughness_dw_preserves_input_source_range() -> None:
    sampling_rate = 44100.0
    time = np.linspace(0.0, 1.0, int(sampling_rate), endpoint=False)
    signal = np.sin(2 * np.pi * 1000.0 * time).reshape(1, -1)
    frame = ChannelFrame(da.from_array(signal, chunks=(1, 4410)), sampling_rate=sampling_rate, source_time_offset=2.0)

    result = frame.roughness_dw(overlap=0.5)

    assert result.source_time_range == pytest.approx(frame.source_time_range)


def test_roughness_time_indexing_advances_source_range() -> None:
    previous = ChannelFrame(
        da.from_array(np.ones((1, 10), dtype=float), chunks=(1, -1)),
        sampling_rate=10.0,
        source_time_offset=3.0,
    )
    roughness = RoughnessFrame(
        data=da.from_array(np.ones((1, 47, 10), dtype=float), chunks=(1, 47, -1)),
        sampling_rate=10.0,
        bark_axis=np.linspace(0.5, 23.5, 47),
        overlap=0.5,
        previous=previous,
        source_time_offset=previous.source_time_offset,
    )

    result = roughness[:, :, 2:5]

    assert result.source_time_offset == pytest.approx(3.2)
    assert result.source_time_range == pytest.approx((3.2, 3.5))
