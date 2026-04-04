"""Tests for DC removal operation."""

import numpy as np

import wandas as wd
from wandas.processing import create_operation

_SR: int = 1000


class TestRemoveDC:
    """Tests for RemoveDC operation: Layer 1 + Layer 2 + Layer 3."""

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_remove_dc_pure_dc_offset_returns_zero_mean(self) -> None:
        """Pure DC offset signal returns zero mean after removal."""
        t = np.linspace(0, 1, _SR, endpoint=False)
        dc_offset = 2.0
        signal_data = np.ones_like(t) + dc_offset

        signal = wd.from_numpy(data=signal_data.reshape(1, -1), sampling_rate=_SR, ch_labels=["DC Signal"])
        clean_signal = signal.remove_dc()

        # atol=1e-10: DC removal yields near-zero mean (float64 precision)
        np.testing.assert_allclose(clean_signal.data.mean(), 0.0, atol=1e-10)

    def test_remove_dc_zero_mean_signal_unchanged(self) -> None:
        """Signal already at zero mean remains unchanged.

        Tolerance: atol=1e-10 — float64 precision.
        """
        t = np.linspace(0, 1, _SR, endpoint=False)
        signal_data = np.sin(2 * np.pi * 50 * t)

        signal = wd.from_numpy(data=signal_data.reshape(1, -1), sampling_rate=_SR, ch_labels=["Zero Mean"])
        clean_signal = signal.remove_dc()

        # atol=1e-10: zero-mean signal should be unchanged after DC removal (float64 precision)
        np.testing.assert_allclose(clean_signal.data, signal.data, atol=1e-10)

    def test_remove_dc_direct_operation_1d(self) -> None:
        """RemoveDC operation on 1D array subtracts mean."""

        op = create_operation("remove_dc", sampling_rate=_SR)
        data_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = op._process_array(data_1d)
        np.testing.assert_allclose(result, data_1d - data_1d.mean())

    def test_remove_dc_direct_operation_2d(self) -> None:
        """RemoveDC operation on 2D array subtracts per-channel mean."""

        op = create_operation("remove_dc", sampling_rate=_SR)
        data_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = op._process_array(data_2d)
        expected = data_2d - data_2d.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(result, expected)

    # -- Layer 2: Domain (immutability + metadata + shapes) ----------------

    def test_remove_dc_returns_new_instance(self) -> None:
        """Pillar 1: remove_dc returns a new frame; original unchanged."""
        rng = np.random.default_rng(42)
        signal_data = rng.standard_normal((2, _SR)) + 5.0

        signal = wd.from_numpy(data=signal_data, sampling_rate=_SR, ch_labels=["Ch1", "Ch2"])
        original_data = signal.data.copy()
        clean_signal = signal.remove_dc()

        assert clean_signal is not signal
        np.testing.assert_array_equal(signal.data, original_data)

    def test_remove_dc_preserves_shape(self) -> None:
        n_channels = 4
        rng = np.random.default_rng(42)
        signal_data = rng.standard_normal((n_channels, 2000)) + 5.0

        signal = wd.from_numpy(
            data=signal_data,
            sampling_rate=_SR,
            ch_labels=[f"Ch{i + 1}" for i in range(n_channels)],
        )
        clean_signal = signal.remove_dc()

        assert clean_signal.shape == signal.shape
        assert clean_signal.n_channels == signal.n_channels
        assert clean_signal.n_samples == signal.n_samples

    def test_remove_dc_operation_history_grows_by_one(self) -> None:
        """Pillar 2: operation_history increases by 1 with correct registry key."""
        rng = np.random.default_rng(42)
        signal_data = rng.standard_normal((1, _SR)) + 2.0
        signal = wd.from_numpy(data=signal_data, sampling_rate=_SR, ch_labels=["Test"])
        clean_signal = signal.remove_dc()

        assert len(clean_signal.operation_history) == len(signal.operation_history) + 1
        assert clean_signal.operation_history[-1]["operation"] == "remove_dc"

    def test_remove_dc_multi_channel_each_zero_mean(self) -> None:
        t = np.linspace(0, 1, _SR, endpoint=False)
        signal_data = np.array(
            [
                1.5 + np.sin(2 * np.pi * 50 * t),
                -2.0 + np.sin(2 * np.pi * 100 * t),
                0.5 + np.sin(2 * np.pi * 150 * t),
            ]
        )
        signal = wd.from_numpy(data=signal_data, sampling_rate=_SR, ch_labels=["Ch1", "Ch2", "Ch3"])
        clean_signal = signal.remove_dc()

        for i in range(3):
            # atol=1e-10: per-channel DC removal yields near-zero mean (float64 precision)
            np.testing.assert_allclose(clean_signal.data[i].mean(), 0.0, atol=1e-10)

    # -- Layer 3: Numerical verification -----------------------------------

    def test_remove_dc_preserves_ac_rms(self) -> None:
        """AC component RMS preserved after DC removal.

        Tolerance: rtol=0.01 — windowing effect on finite-length sine.
        """
        t = np.linspace(0, 1, _SR, endpoint=False)
        dc_offset = 3.5
        signal_data = dc_offset + np.sin(2 * np.pi * 50 * t)

        signal = wd.from_numpy(data=signal_data.reshape(1, -1), sampling_rate=_SR, ch_labels=["Signal with DC"])
        clean_signal = signal.remove_dc()

        # atol=1e-10: DC removal yields near-zero mean (float64 precision)
        np.testing.assert_allclose(clean_signal.data.mean(), 0.0, atol=1e-10)
        expected_rms = 1.0 / np.sqrt(2)
        np.testing.assert_allclose(clean_signal.rms[0], expected_rms, rtol=0.01)

    def test_remove_dc_method_chaining_with_filter(self) -> None:
        """Method chaining: remove_dc -> low_pass_filter records both operations."""
        t = np.linspace(0, 1, _SR, endpoint=False)
        rng = np.random.default_rng(42)
        signal_data = 2.0 + np.sin(2 * np.pi * 50 * t) + 0.1 * rng.standard_normal(len(t))

        signal = wd.from_numpy(data=signal_data.reshape(1, -1), sampling_rate=_SR, ch_labels=["Noisy Signal"])
        processed = signal.remove_dc().low_pass_filter(cutoff=100)

        assert len(processed.operation_history) >= 2
        assert any(op["operation"] == "remove_dc" for op in processed.operation_history)
        assert any(op["operation"] == "lowpass_filter" for op in processed.operation_history)
