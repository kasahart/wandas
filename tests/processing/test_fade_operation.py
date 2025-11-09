import dask.array as da
import numpy as np
import pytest
from scipy.signal import windows as sp_windows

from wandas.processing.base import create_operation


def _to_dask(arr: np.ndarray):
    return da.from_array(arr, chunks=-1)


def test_fade_noop_when_zero():
    sr = 1000
    n = 100
    sig = np.ones((1, n), dtype=float)
    dsig = _to_dask(sig)

    op = create_operation("fade", sr, fade_ms=0.0)
    out = op.process(dsig).compute()

    assert out.shape == sig.shape
    assert np.allclose(out, sig)


def test_fade_tukey_matches_expected_single_channel():
    sr = 1000
    n = 200
    # choose fade_ms such that fade_len is 20 samples
    fade_len = 20
    fade_ms = fade_len * 1000.0 / sr

    sig = np.ones((1, n), dtype=float)
    dsig = _to_dask(sig)

    op = create_operation("fade", sr, fade_ms=fade_ms)
    out = op.process(dsig).compute()

    # expected tukey window using Fade's static method
    from wandas.processing.effects import Fade

    alpha = Fade.calculate_tukey_alpha(fade_len, n)
    expected = sp_windows.tukey(n, alpha=alpha)

    assert out.shape == sig.shape
    np.testing.assert_allclose(out[0], expected, rtol=1e-10, atol=1e-12)


def test_fade_preserves_multi_channel_shape():
    sr = 8000
    n = 512
    fade_len = 32
    fade_ms = fade_len * 1000.0 / sr

    sig = np.vstack(
        [
            np.ones(n, dtype=float),
            np.linspace(0.0, 1.0, n, dtype=float),
        ]
    )
    dsig = _to_dask(sig)

    op = create_operation("fade", sr, fade_ms=fade_ms)
    out = op.process(dsig).compute()

    assert out.shape == sig.shape


def test_fade_too_long_raises():
    sr = 1000
    n = 100
    # fade_len such that 2*fade_len >= n
    fade_len = 50
    fade_ms = fade_len * 1000.0 / sr

    sig = np.ones((1, n), dtype=float)
    dsig = _to_dask(sig)

    op = create_operation("fade", sr, fade_ms=fade_ms)

    with pytest.raises(ValueError):
        op.process(dsig).compute()


def test_fade_negative_fade_ms_raises():
    """Test that negative fade_ms raises ValueError."""
    sr = 1000

    with pytest.raises(ValueError, match="fade_ms must be non-negative"):
        create_operation("fade", sr, fade_ms=-10.0)


def test_fade_small_fade_duration():
    """Test fade with very small duration."""
    sr = 44100
    n = 1000
    fade_ms = 0.5  # 0.5 milliseconds

    sig = np.ones((1, n), dtype=float)
    dsig = _to_dask(sig)

    op = create_operation("fade", sr, fade_ms=fade_ms)
    out = op.process(dsig).compute()

    assert out.shape == sig.shape
    # Most of the signal should be unaffected
    assert np.sum(out[0] > 0.9) > n * 0.9


def test_fade_large_fade_duration():
    """Test fade with large duration (but valid)."""
    sr = 8000
    n = 1000
    fade_len = 400  # 40% of signal length (within valid range)
    fade_ms = fade_len * 1000.0 / sr

    sig = np.ones((1, n), dtype=float)
    dsig = _to_dask(sig)

    op = create_operation("fade", sr, fade_ms=fade_ms)
    out = op.process(dsig).compute()

    assert out.shape == sig.shape
    # Beginning and end should be faded
    assert out[0, 0] < 0.1
    assert out[0, -1] < 0.1


def test_fade_symmetric():
    """Test that fade is symmetric at both ends."""
    sr = 8000
    n = 500
    fade_len = 50
    fade_ms = fade_len * 1000.0 / sr

    sig = np.ones((1, n), dtype=float)
    dsig = _to_dask(sig)

    op = create_operation("fade", sr, fade_ms=fade_ms)
    out = op.process(dsig).compute()

    # First fade_len samples should mirror last fade_len samples
    np.testing.assert_allclose(out[0, :fade_len], out[0, -fade_len:][::-1], rtol=1e-10)


def test_fade_preserves_middle():
    """Test that fade preserves the middle portion of the signal."""
    sr = 8000
    n = 1000
    fade_len = 100
    fade_ms = fade_len * 1000.0 / sr

    sig = np.ones((1, n), dtype=float) * 2.0  # Signal with amplitude 2.0
    dsig = _to_dask(sig)

    op = create_operation("fade", sr, fade_ms=fade_ms)
    out = op.process(dsig).compute()

    # Middle portion should be unchanged (multiply by 1.0)
    middle_start = fade_len + 50
    middle_end = n - fade_len - 50
    np.testing.assert_allclose(
        out[0, middle_start:middle_end], 2.0, rtol=1e-10, atol=1e-12
    )


def test_fade_with_real_signal():
    """Test fade with a real sine wave signal."""
    sr = 16000
    n = 8000
    fade_ms = 50.0

    # Create sine wave
    t = np.arange(n) / sr
    sig = np.sin(2 * np.pi * 440 * t).reshape(1, -1)
    dsig = _to_dask(sig)

    op = create_operation("fade", sr, fade_ms=fade_ms)
    out = op.process(dsig).compute()

    assert out.shape == sig.shape
    # First sample should be faded
    assert abs(out[0, 0]) < abs(sig[0, 0])
    # Last sample should be faded
    assert abs(out[0, -1]) < abs(sig[0, -1])


def test_fade_multichannel_independence():
    """Test that fade is applied independently to each channel."""
    sr = 8000
    n = 500
    fade_ms = 25.0

    # Create different signals for each channel
    sig = np.vstack(
        [
            np.ones(n) * 1.0,
            np.ones(n) * 2.0,
            np.ones(n) * 3.0,
        ]
    )
    dsig = _to_dask(sig)

    op = create_operation("fade", sr, fade_ms=fade_ms)
    out = op.process(dsig).compute()

    # Each channel should be faded independently
    # Check that relative amplitudes are preserved in the middle
    middle_idx = n // 2
    np.testing.assert_allclose(
        out[1, middle_idx] / out[0, middle_idx], 2.0, rtol=1e-10
    )
    np.testing.assert_allclose(
        out[2, middle_idx] / out[0, middle_idx], 3.0, rtol=1e-10
    )


def test_fade_default_parameter():
    """Test fade with default fade_ms parameter."""
    sr = 16000
    n = 8000

    sig = np.ones((1, n), dtype=float)
    dsig = _to_dask(sig)

    # Should use default fade_ms=50
    op = create_operation("fade", sr)
    out = op.process(dsig).compute()

    assert out.shape == sig.shape
    # Should have some fading effect
    assert out[0, 0] < 0.1


def test_fade_with_short_signal():
    """Test fade with signal barely longer than required minimum."""
    sr = 1000
    n = 100
    fade_ms = 25.0  # 25 samples, 2*25=50 < 100, should work

    sig = np.ones((1, n), dtype=float)
    dsig = _to_dask(sig)

    op = create_operation("fade", sr, fade_ms=fade_ms)
    out = op.process(dsig).compute()

    assert out.shape == sig.shape


def test_fade_exact_boundary():
    """Test fade at exact boundary where 2*fade_len equals signal length."""
    sr = 1000
    n = 100
    fade_len = 49  # 2*49 = 98 < 100, should work
    fade_ms = fade_len * 1000.0 / sr

    sig = np.ones((1, n), dtype=float)
    dsig = _to_dask(sig)

    op = create_operation("fade", sr, fade_ms=fade_ms)
    out = op.process(dsig).compute()

    assert out.shape == sig.shape
