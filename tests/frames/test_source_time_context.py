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
