import dask.array as da
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame


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
