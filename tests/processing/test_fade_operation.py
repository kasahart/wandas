import dask.array as da
import numpy as np
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

    # expected tukey window
    alpha = float(2 * fade_len) / float(n)
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
    import pytest

    with pytest.raises(ValueError):
        op.process(dsig).compute()
