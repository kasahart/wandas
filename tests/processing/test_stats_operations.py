import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from wandas.processing.base import create_operation, get_operation
from wandas.processing.stats import (
    ABS,
    ChannelDifference,
    CrestFactor,
    Mean,
    Power,
    Sum,
)
from wandas.utils.types import NDArrayReal

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestABS:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.abs_op = ABS(self.sample_rate)

        # Create test signal with positive and negative values
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])
        self.signal_stereo: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 440 * t),
                np.sin(2 * np.pi * 440 * t + np.pi),  # phase-shifted by pi (negative)
            ]
        )

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(1, -1))

    def test_initialization(self) -> None:
        """Test initialization."""
        abs_op = ABS(self.sample_rate)
        assert abs_op.sampling_rate == self.sample_rate

    def test_shape_preservation(self) -> None:
        """Test that output shape matches input shape."""
        result = self.abs_op.process(self.dask_mono).compute()
        assert result.shape == self.signal_mono.shape

        result_stereo = self.abs_op.process(self.dask_stereo).compute()
        assert result_stereo.shape == self.signal_stereo.shape

    def test_abs_values(self) -> None:
        """Test that negative values become positive."""
        result = self.abs_op.process(self.dask_stereo).compute()

        # All values should be non-negative
        assert np.all(result >= 0)

        # Verify that the content is the absolute value
        np.testing.assert_allclose(result, np.abs(self.signal_stereo))

    def test_operation_registry(self) -> None:
        """Test that ABS is properly registered in the operation registry."""
        assert get_operation("abs") == ABS

        abs_op = create_operation("abs", self.sample_rate)
        assert isinstance(abs_op, ABS)
        assert abs_op.sampling_rate == self.sample_rate


class TestPowerOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.exponent: float = 2.0
        self.power_op = Power(self.sample_rate, exponent=self.exponent)

        # Create test signal
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])
        self.signal_stereo: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 440 * t),
                np.sin(2 * np.pi * 880 * t) * 0.5,  # Half amplitude
            ]
        )

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(1, -1))

    def test_initialization(self) -> None:
        """Test initialization with different exponents."""
        power_op = Power(self.sample_rate, exponent=3.0)
        assert power_op.sampling_rate == self.sample_rate
        assert power_op.exp == 3.0

    def test_shape_preservation(self) -> None:
        """Test that output shape matches input shape."""
        result = self.power_op.process(self.dask_mono).compute()
        assert result.shape == self.signal_mono.shape

        result_stereo = self.power_op.process(self.dask_stereo).compute()
        assert result_stereo.shape == self.signal_stereo.shape

    def test_power_values(self) -> None:
        """Test that values are raised to the correct power."""
        result = self.power_op.process(self.dask_stereo).compute()

        # Verify the result is correct
        expected = np.power(self.signal_stereo, self.exponent)
        np.testing.assert_allclose(result, expected)

    def test_different_exponents(self) -> None:
        """Test with different exponent values."""
        # Test with fractional exponent
        power_sqrt = Power(self.sample_rate, exponent=0.5)
        result_sqrt = power_sqrt.process(self.dask_stereo).compute()
        expected_sqrt = np.sqrt(self.signal_stereo)
        np.testing.assert_allclose(result_sqrt, expected_sqrt)

        # Test with negative exponent
        power_recip = Power(self.sample_rate, exponent=-1.0)
        # To avoid division by zero, use a signal with no zeros
        nonzero_signal = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
        nonzero_dask = _da_from_array(nonzero_signal, chunks=(1, -1))
        result_recip = power_recip.process(nonzero_dask).compute()
        expected_recip = 1.0 / nonzero_signal
        np.testing.assert_allclose(result_recip, expected_recip)

    def test_operation_registry(self) -> None:
        """Test that Power is properly registered in the operation registry."""
        assert get_operation("power") == Power

        power_op = create_operation("power", self.sample_rate, exponent=3.0)
        assert isinstance(power_op, Power)
        assert power_op.sampling_rate == self.sample_rate
        assert power_op.exp == 3.0


class TestSum:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.sum_op = Sum(self.sample_rate)

        # Create test multi-channel signals
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])
        self.signal_stereo: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t) * 0.5])
        self.signal_quad: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 440 * t),
                np.sin(2 * np.pi * 880 * t) * 0.5,
                np.sin(2 * np.pi * 1320 * t) * 0.25,
                np.sin(2 * np.pi * 1760 * t) * 0.125,
            ]
        )

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(1, -1))
        self.dask_quad: DaArray = _da_from_array(self.signal_quad, chunks=(1, -1))

    def test_initialization(self) -> None:
        """Test initialization."""
        sum_op = Sum(self.sample_rate)
        assert sum_op.sampling_rate == self.sample_rate

    def test_shape_change(self) -> None:
        """Test that output has the expected shape (1, samples)."""
        # Mono input should remain same shape
        result_mono = self.sum_op.process(self.dask_mono).compute()
        assert result_mono.shape == self.signal_mono.shape

        # Multi-channel input should be summed to mono
        result_stereo = self.sum_op.process(self.dask_stereo).compute()
        assert result_stereo.shape == (1, self.signal_stereo.shape[1])

        result_quad = self.sum_op.process(self.dask_quad).compute()
        assert result_quad.shape == (1, self.signal_quad.shape[1])

    def test_sum_values(self) -> None:
        """Test that channels are properly summed."""
        # For mono, output should be identical to input
        result_mono = self.sum_op.process(self.dask_mono).compute()
        np.testing.assert_allclose(result_mono, self.signal_mono)

        # For multi-channel, output should be sum of channels
        result_stereo = self.sum_op.process(self.dask_stereo).compute()
        expected_stereo = np.sum(self.signal_stereo, axis=0, keepdims=True)
        np.testing.assert_allclose(result_stereo, expected_stereo)

        result_quad = self.sum_op.process(self.dask_quad).compute()
        expected_quad = np.sum(self.signal_quad, axis=0, keepdims=True)
        np.testing.assert_allclose(result_quad, expected_quad)

    def test_operation_registry(self) -> None:
        """Test that Sum is properly registered in the operation registry."""
        assert get_operation("sum") == Sum

        sum_op = create_operation("sum", self.sample_rate)
        assert isinstance(sum_op, Sum)
        assert sum_op.sampling_rate == self.sample_rate


class TestMean:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.mean_op = Mean(self.sample_rate)

        # Create test multi-channel signals
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])
        self.signal_stereo: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t) * 0.5])
        self.signal_quad: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 440 * t),
                np.sin(2 * np.pi * 880 * t) * 0.5,
                np.sin(2 * np.pi * 1320 * t) * 0.25,
                np.sin(2 * np.pi * 1760 * t) * 0.125,
            ]
        )

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(1, -1))
        self.dask_quad: DaArray = _da_from_array(self.signal_quad, chunks=(1, -1))

    def test_initialization(self) -> None:
        """Test initialization."""
        mean_op = Mean(self.sample_rate)
        assert mean_op.sampling_rate == self.sample_rate

    def test_shape_change(self) -> None:
        """Test that output has the expected shape (1, samples)."""
        # Mono input should remain same shape
        result_mono = self.mean_op.process(self.dask_mono).compute()
        assert result_mono.shape == self.signal_mono.shape

        # Multi-channel input should be averaged to mono
        result_stereo = self.mean_op.process(self.dask_stereo).compute()
        assert result_stereo.shape == (1, self.signal_stereo.shape[1])

        result_quad = self.mean_op.process(self.dask_quad).compute()
        assert result_quad.shape == (1, self.signal_quad.shape[1])

    def test_mean_values(self) -> None:
        """Test that channels are properly averaged."""
        # For mono, output should be identical to input
        result_mono = self.mean_op.process(self.dask_mono).compute()
        np.testing.assert_allclose(result_mono, self.signal_mono)

        # For multi-channel, output should be mean of channels
        result_stereo = self.mean_op.process(self.dask_stereo).compute()
        expected_stereo = self.signal_stereo.mean(axis=0, keepdims=True)

        np.testing.assert_allclose(result_stereo, expected_stereo)

        result_quad = self.mean_op.process(self.dask_quad).compute()
        expected_quad = self.dask_quad.mean(axis=0, keepdims=True)

        np.testing.assert_allclose(result_quad, expected_quad)

    def test_operation_registry(self) -> None:
        """Test that Mean is properly registered in the operation registry."""
        assert get_operation("mean") == Mean

        mean_op = create_operation("mean", self.sample_rate)
        assert isinstance(mean_op, Mean)
        assert mean_op.sampling_rate == self.sample_rate


class TestChannelDifference:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.other_channel: int = 0
        self.diff_op = ChannelDifference(self.sample_rate, other_channel=self.other_channel)

        # Create test multi-channel signals
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])
        self.signal_stereo: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 440 * t),
                np.sin(2 * np.pi * 440 * t) * 0.5 + 0.1,  # Different amplitude and DC offset
            ]
        )
        self.signal_quad: NDArrayReal = np.array(
            [
                np.ones(self.sample_rate),  # Reference channel (all ones)
                np.zeros(self.sample_rate),  # All zeros
                np.ones(self.sample_rate) * 2,  # All twos
                np.ones(self.sample_rate) * -1,  # All negative ones
            ]
        )

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(1, -1))
        self.dask_quad: DaArray = _da_from_array(self.signal_quad, chunks=(1, -1))

    def test_initialization(self) -> None:
        """Test initialization with different reference channels."""
        diff_op = ChannelDifference(self.sample_rate, other_channel=1)
        assert diff_op.sampling_rate == self.sample_rate
        assert diff_op.other_channel == 1

    def test_shape_preservation(self) -> None:
        """Test that output shape matches input shape."""
        result = self.diff_op.process(self.dask_stereo).compute()
        assert result.shape == self.signal_stereo.shape

        result_quad = self.diff_op.process(self.dask_quad).compute()
        assert result_quad.shape == self.signal_quad.shape

    def test_difference_values(self) -> None:
        """Test that channel differences are correctly calculated."""
        result = self.diff_op.process(self.dask_stereo).compute()

        # Difference with reference channel (channel 0)
        expected = self.signal_stereo - self.signal_stereo[self.other_channel]
        np.testing.assert_allclose(result, expected)

        # First channel (diff with itself) should be zeros
        np.testing.assert_allclose(result[0], np.zeros_like(result[0]))

        # Test with quad-channel signal
        result_quad = self.diff_op.process(self.dask_quad).compute()
        expected_quad = self.signal_quad - self.signal_quad[self.other_channel]
        np.testing.assert_allclose(result_quad, expected_quad)

        # With reference channel 0 (all ones), the differences should be:
        # Channel 0: All zeros (1-1)
        # Channel 1: All negative ones (0-1)
        # Channel 2: All ones (2-1)
        # Channel 3: All negative twos (-1-1)
        np.testing.assert_allclose(result_quad[0], np.zeros_like(result_quad[0]))
        np.testing.assert_allclose(result_quad[1], -np.ones_like(result_quad[1]))
        np.testing.assert_allclose(result_quad[2], np.ones_like(result_quad[2]))
        np.testing.assert_allclose(result_quad[3], -2 * np.ones_like(result_quad[3]))

    def test_different_reference_channel(self) -> None:
        """Test with a different reference channel."""
        diff_op2 = ChannelDifference(self.sample_rate, other_channel=2)
        result = diff_op2.process(self.dask_quad).compute()

        # With reference channel 2 (all twos), the differences should be:
        # Channel 0: All negative ones (1-2)
        # Channel 1: All negative twos (0-2)
        # Channel 2: All zeros (2-2)
        # Channel 3: All negative threes (-1-2)
        expected = self.signal_quad - self.signal_quad[2]
        np.testing.assert_allclose(result, expected)

        np.testing.assert_allclose(result[0], -np.ones_like(result[0]))
        np.testing.assert_allclose(result[1], -2 * np.ones_like(result[1]))
        np.testing.assert_allclose(result[2], np.zeros_like(result[2]))
        np.testing.assert_allclose(result[3], -3 * np.ones_like(result[3]))

    def test_operation_registry(self) -> None:
        """
        Test that ChannelDifference is properly registered in the operation registry.
        """
        assert get_operation("channel_difference") == ChannelDifference

        diff_op = create_operation("channel_difference", self.sample_rate, other_channel=1)
        assert isinstance(diff_op, ChannelDifference)
        assert diff_op.sampling_rate == self.sample_rate
        assert diff_op.other_channel == 1


class TestCrestFactor:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.cf_op = CrestFactor(self.sample_rate)

        # Pure sine wave: crest factor == sqrt(2) analytically
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_sine: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])
        self.signal_stereo: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 440 * t),
                np.sin(2 * np.pi * 880 * t) * 0.5,
            ]
        )
        # Silent signal (all zeros) -> crest factor is undefined (nan)
        self.signal_silent: NDArrayReal = np.zeros((1, self.sample_rate))

        self.dask_sine: DaArray = da.from_array(self.signal_sine, chunks=(1, -1))
        self.dask_stereo: DaArray = da.from_array(self.signal_stereo, chunks=(1, -1))
        self.dask_silent: DaArray = da.from_array(self.signal_silent, chunks=(1, -1))

    def test_initialization(self) -> None:
        """Test initialization."""
        cf_op = CrestFactor(self.sample_rate)
        assert cf_op.sampling_rate == self.sample_rate

    def test_output_shape(self) -> None:
        """Test that output shape is (n_channels, 1)."""
        result_mono = self.cf_op.process(self.dask_sine).compute()
        assert result_mono.shape == (1, 1)

        result_stereo = self.cf_op.process(self.dask_stereo).compute()
        assert result_stereo.shape == (2, 1)

    def test_sine_wave_crest_factor(self) -> None:
        """Crest factor of a pure sine is analytically sqrt(2)."""
        result = self.cf_op.process(self.dask_sine).compute()
        np.testing.assert_allclose(result[0, 0], np.sqrt(2), rtol=1e-5)

    def test_stereo_crest_factor(self) -> None:
        """Each channel of a stereo sine signal should have crest factor sqrt(2)."""
        result = self.cf_op.process(self.dask_stereo).compute()
        np.testing.assert_allclose(result[:, 0], np.full(2, np.sqrt(2)), rtol=1e-5)

    def test_dc_signal_crest_factor(self) -> None:
        """Crest factor of a constant DC signal is 1.0 (peak == RMS)."""
        dc_signal: NDArrayReal = np.ones((1, self.sample_rate)) * 3.0
        dask_dc: DaArray = da.from_array(dc_signal, chunks=(1, -1))
        result = self.cf_op.process(dask_dc).compute()
        np.testing.assert_allclose(result[0, 0], 1.0, rtol=1e-6)

    def test_silent_input_returns_nan(self) -> None:
        """Silent (all-zero) input must return nan rather than raise ZeroDivisionError."""
        result = self.cf_op.process(self.dask_silent).compute()
        assert result.shape == (1, 1)
        assert np.isnan(result[0, 0])

    def test_mixed_silent_and_active_channels(self) -> None:
        """Channels with signal return a value; silent channels return nan."""
        mixed: NDArrayReal = np.vstack([self.signal_sine, self.signal_silent])
        dask_mixed: DaArray = da.from_array(mixed, chunks=(1, -1))
        result = self.cf_op.process(dask_mixed).compute()
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result[0, 0], np.sqrt(2), rtol=1e-5)
        assert np.isnan(result[1, 0])

    def test_display_name(self) -> None:
        """Test display name."""
        assert self.cf_op.get_display_name() == "crest_factor"

    def test_operation_registry(self) -> None:
        """Test that CrestFactor is properly registered in the operation registry."""
        assert get_operation("crest_factor") == CrestFactor

        cf_op = create_operation("crest_factor", self.sample_rate)
        assert isinstance(cf_op, CrestFactor)
        assert cf_op.sampling_rate == self.sample_rate


class TestCrestFactorFrameMethod:
    """Integration tests for ChannelFrame.crest_factor()."""

    def setup_method(self) -> None:
        from wandas.frames.channel import ChannelFrame

        self.sample_rate = 16000
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        sine = np.sin(2 * np.pi * 440 * t)
        zeros = np.zeros(self.sample_rate)

        self.frame_sine = ChannelFrame.from_numpy(sine.reshape(1, -1), sampling_rate=self.sample_rate)
        self.frame_stereo = ChannelFrame.from_numpy(np.vstack([sine, sine * 0.5]), sampling_rate=self.sample_rate)
        self.frame_silent = ChannelFrame.from_numpy(zeros.reshape(1, -1), sampling_rate=self.sample_rate)

    def test_returns_channel_frame(self) -> None:
        from wandas.frames.channel import ChannelFrame

        result = self.frame_sine.crest_factor()
        assert isinstance(result, ChannelFrame)

    def test_shape_one_sample_per_channel(self) -> None:
        result = self.frame_stereo.crest_factor()
        assert result.n_channels == 2
        assert result.n_samples == 1

    def test_sine_value(self) -> None:
        result = self.frame_sine.crest_factor()
        data = result._data.compute()
        np.testing.assert_allclose(data[0, 0], np.sqrt(2), rtol=1e-5)

    def test_silent_frame_returns_nan(self) -> None:
        result = self.frame_silent.crest_factor()
        data = result._data.compute()
        assert np.isnan(data[0, 0])

    def test_immutability(self) -> None:
        """Original frame must not be modified."""
        original_data = self.frame_sine._data.compute().copy()
        _ = self.frame_sine.crest_factor()
        np.testing.assert_array_equal(self.frame_sine._data.compute(), original_data)

    def test_operation_history(self) -> None:
        """crest_factor must be recorded in operation_history."""
        result = self.frame_sine.crest_factor()
        assert len(result.operation_history) == len(self.frame_sine.operation_history) + 1
        assert result.operation_history[-1]["operation"] == "crest_factor"

    def test_lazy_evaluation(self) -> None:
        """Result data must remain a Dask array."""
        result = self.frame_sine.crest_factor()
        assert isinstance(result._data, da.Array)
