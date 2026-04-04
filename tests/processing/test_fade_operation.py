import numpy as np
import pytest
from dask.array.core import Array as DaArray
from scipy.signal import windows as sp_windows

from wandas.processing.base import create_operation
from wandas.processing.effects import Fade
from wandas.utils.dask_helpers import da_from_array

_SR: int = 1000


class TestFade:
    """Fade operation: Layer 1 + Layer 2 + Layer 3 (scipy Tukey reference)."""

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_fade_noop_when_fade_ms_zero(self) -> None:
        """Zero fade_ms produces identity output."""
        sig = np.ones((1, 100), dtype=float)
        dsig = da_from_array(sig, chunks=(1, -1))

        op = create_operation("fade", _SR, fade_ms=0.0)
        out_da = op.process(dsig)
        assert isinstance(out_da, DaArray)  # Pillar 1: Dask graph preserved
        out = out_da.compute()

        assert out.shape == sig.shape
        np.testing.assert_array_equal(out, sig)

    def test_fade_too_long_raises_error(self) -> None:
        """Fade length >= half signal length raises ValueError."""
        sig = np.ones((1, 100), dtype=float)
        dsig = da_from_array(sig, chunks=(1, -1))

        # fade_len = 50, 2*50 >= 100 → error
        op = create_operation("fade", _SR, fade_ms=50.0)
        with pytest.raises(ValueError, match=r"Fade length too long"):
            op.process(dsig).compute()

    # -- Layer 2: Domain (shape + immutability) ----------------------------

    def test_fade_preserves_immutability_and_dask_type(self) -> None:
        """Input unchanged after fade; result is DaArray."""
        sig = np.ones((1, 200), dtype=float)
        dsig = da_from_array(sig, chunks=(1, -1))
        input_copy = sig.copy()

        op = create_operation("fade", _SR, fade_ms=20.0)
        result_da = op.process(dsig)

        # Pillar 1: immutability
        assert result_da is not dsig
        np.testing.assert_array_equal(dsig.compute(), input_copy)
        assert isinstance(result_da, DaArray)

    def test_fade_preserves_multichannel_shape(self) -> None:
        """Multi-channel signal preserves (channels, samples) shape."""
        n = 512
        fade_ms = 32 * 1000.0 / 8000
        sig = np.vstack(
            [
                np.ones(n, dtype=float),
                np.linspace(0.0, 1.0, n, dtype=float),
            ]
        )
        dsig = da_from_array(sig, chunks=(1, -1))

        op = create_operation("fade", 8000, fade_ms=fade_ms)
        out_da = op.process(dsig)
        assert isinstance(out_da, DaArray)  # Pillar 1: Dask graph preserved
        out = out_da.compute()
        assert out.shape == sig.shape

    # -- Layer 3: scipy Tukey reference ------------------------------------

    def test_fade_tukey_matches_scipy_reference(self) -> None:
        """Single-channel fade matches scipy.signal.windows.tukey.

        Tolerance: rtol=1e-10, atol=1e-12 — float64 window multiplication.
        """
        n = 200
        fade_len = 20
        fade_ms = fade_len * 1000.0 / _SR

        sig = np.ones((1, n), dtype=float)
        dsig = da_from_array(sig, chunks=(1, -1))

        op = create_operation("fade", _SR, fade_ms=fade_ms)
        out_da = op.process(dsig)
        assert isinstance(out_da, DaArray)  # Pillar 1: Dask graph preserved
        out = out_da.compute()

        alpha = Fade.calculate_tukey_alpha(fade_len, n)
        expected = sp_windows.tukey(n, alpha=alpha)

        assert out.shape == sig.shape
        np.testing.assert_allclose(
            out[0],
            expected,
            rtol=1e-10,
            atol=1e-12,  # float64 window multiplication precision
        )
