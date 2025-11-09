"""Tests for DC removal operation."""

import numpy as np

import wandas as wd


class TestRemoveDC:
    """Tests for RemoveDC operation."""

    def test_remove_dc_basic(self) -> None:
        """Test basic DC removal."""
        # Create signal with DC offset
        sampling_rate = 1000
        duration = 1.0
        time = np.linspace(0, duration, int(duration * sampling_rate))

        # Pure DC offset
        dc_offset = 2.0
        signal_data = np.ones_like(time) + dc_offset

        # Create ChannelFrame
        signal = wd.from_numpy(
            data=signal_data.reshape(1, -1),
            sampling_rate=sampling_rate,
            ch_labels=["DC Signal"],
        )

        # Remove DC
        clean_signal = signal.remove_dc()

        # Check that mean is close to zero
        assert np.allclose(clean_signal.data.mean(), 0.0, atol=1e-10)

    def test_remove_dc_with_ac_component(self) -> None:
        """Test DC removal with AC component."""
        sampling_rate = 1000
        duration = 1.0
        time = np.linspace(0, duration, int(duration * sampling_rate))

        # Signal with AC and DC components
        dc_offset = 3.5
        frequency = 50  # 50Hz
        signal_data = dc_offset + np.sin(2 * np.pi * frequency * time)

        # Create ChannelFrame
        signal = wd.from_numpy(
            data=signal_data.reshape(1, -1),
            sampling_rate=sampling_rate,
            ch_labels=["Signal with DC"],
        )

        # Remove DC
        clean_signal = signal.remove_dc()

        # Check that mean is close to zero
        assert np.allclose(clean_signal.data.mean(), 0.0, atol=1e-10)

        # Check that AC component is preserved (RMS should be similar)
        # For sine wave, RMS = amplitude / sqrt(2) ≈ 0.707
        expected_rms = 1.0 / np.sqrt(2)
        assert np.allclose(clean_signal.rms[0], expected_rms, rtol=0.01)

    def test_remove_dc_multi_channel(self) -> None:
        """Test DC removal with multiple channels."""
        sampling_rate = 1000
        duration = 1.0
        time = np.linspace(0, duration, int(duration * sampling_rate))

        # Different DC offsets for each channel
        dc_offset_1 = 1.5
        dc_offset_2 = -2.0
        dc_offset_3 = 0.5

        signal_ch1 = dc_offset_1 + np.sin(2 * np.pi * 50 * time)
        signal_ch2 = dc_offset_2 + np.sin(2 * np.pi * 100 * time)
        signal_ch3 = dc_offset_3 + np.sin(2 * np.pi * 150 * time)

        signal_data = np.stack([signal_ch1, signal_ch2, signal_ch3])

        # Create ChannelFrame
        signal = wd.from_numpy(
            data=signal_data,
            sampling_rate=sampling_rate,
            ch_labels=["Ch1", "Ch2", "Ch3"],
        )

        # Remove DC
        clean_signal = signal.remove_dc()

        # Check that mean is close to zero for each channel
        for i in range(3):
            channel_mean = clean_signal.data[i].mean()
            assert np.allclose(channel_mean, 0.0, atol=1e-10)

    def test_remove_dc_preserves_shape(self) -> None:
        """Test that DC removal preserves signal shape."""
        sampling_rate = 1000
        n_samples = 2000
        n_channels = 4

        # Random signal with DC offset
        signal_data = np.random.randn(n_channels, n_samples) + 5.0

        signal = wd.from_numpy(
            data=signal_data,
            sampling_rate=sampling_rate,
            ch_labels=[f"Ch{i + 1}" for i in range(n_channels)],
        )

        clean_signal = signal.remove_dc()

        # Check shape is preserved
        assert clean_signal.shape == signal.shape
        assert clean_signal.n_channels == signal.n_channels
        assert clean_signal.n_samples == signal.n_samples

    def test_remove_dc_operation_history(self) -> None:
        """Test that DC removal is recorded in operation history."""
        sampling_rate = 1000
        signal_data = np.random.randn(1, 1000) + 2.0

        signal = wd.from_numpy(
            data=signal_data, sampling_rate=sampling_rate, ch_labels=["Test"]
        )

        clean_signal = signal.remove_dc()

        # Check operation history
        assert len(clean_signal.operation_history) == len(signal.operation_history) + 1
        assert clean_signal.operation_history[-1]["operation"] == "remove_dc"

    def test_remove_dc_zero_mean_signal(self) -> None:
        """Test DC removal on signal already centered at zero."""
        sampling_rate = 1000
        duration = 1.0
        time = np.linspace(0, duration, int(duration * sampling_rate))

        # Signal already centered at zero
        signal_data = np.sin(2 * np.pi * 50 * time)

        signal = wd.from_numpy(
            data=signal_data.reshape(1, -1),
            sampling_rate=sampling_rate,
            ch_labels=["Zero Mean"],
        )

        clean_signal = signal.remove_dc()

        # Result should be very close to original
        # (small numerical errors may exist due to floating point arithmetic)
        assert np.allclose(clean_signal.data, signal.data, atol=1e-10)

    def test_remove_dc_direct_operation(self) -> None:
        """Test RemoveDC operation class directly."""
        sampling_rate = 1000

        # Create operation using the factory
        from wandas.processing import create_operation

        remove_dc_op = create_operation("remove_dc", sampling_rate=sampling_rate)

        # Test with 1D array
        data_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result_1d = remove_dc_op._process_array(data_1d)
        expected_1d = data_1d - data_1d.mean()
        assert np.allclose(result_1d, expected_1d)

        # Test with 2D array
        data_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result_2d = remove_dc_op._process_array(data_2d)
        expected_2d = data_2d - data_2d.mean(axis=1, keepdims=True)
        assert np.allclose(result_2d, expected_2d)

    def test_remove_dc_with_method_chaining(self) -> None:
        """Test DC removal with method chaining."""
        sampling_rate = 1000
        duration = 1.0
        time = np.linspace(0, duration, int(duration * sampling_rate))

        # Signal with DC offset and high-frequency noise
        signal_data = (
            2.0 + np.sin(2 * np.pi * 50 * time) + 0.1 * np.random.randn(len(time))
        )

        signal = wd.from_numpy(
            data=signal_data.reshape(1, -1),
            sampling_rate=sampling_rate,
            ch_labels=["Noisy Signal"],
        )

        # Chain: remove DC -> low-pass filter
        processed = signal.remove_dc().low_pass_filter(cutoff=100)

        # Check that both operations are in history
        assert len(processed.operation_history) >= 2
        assert any(op["operation"] == "remove_dc" for op in processed.operation_history)
        assert any(
            op["operation"] == "lowpass_filter" for op in processed.operation_history
        )
