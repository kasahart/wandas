import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.processing.base import create_operation, get_operation
from wandas.processing.stats import (
    ABS,
    ChannelDifference,
    Mean,
    Power,
    Sum,
)
from wandas.utils.dask_helpers import da_from_array

_SR: int = 16000


class TestABS:
    """ABS operation: Layer 1 (unit) + Layer 2 (domain) + Layer 3 (numpy equivalence)."""

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_abs_init_stores_sample_rate(self) -> None:
        """Test ABS stores sampling rate."""
        abs_op = ABS(_SR)
        assert abs_op.sampling_rate == _SR

    def test_abs_registry_returns_correct_class(self) -> None:
        """Test ABS is registered as 'abs'."""
        assert get_operation("abs") == ABS
        abs_op = create_operation("abs", _SR)
        assert isinstance(abs_op, ABS)

    # -- Layer 2: Domain (shape + immutability) ----------------------------

    def test_abs_mono_preserves_shape_and_immutability(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Mono signal shape preserved; input unchanged."""
        dask_input, sr = pure_sine_440hz_dask
        abs_op = ABS(sr)
        input_copy = dask_input.compute().copy()

        # Act
        result_da = abs_op.process(dask_input)

        # Assert 1: Immutability
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)

        # Assert 2: Dask graph preserved
        assert isinstance(result_da, DaArray)

        # Assert 3: Shape
        result = result_da.compute()
        assert result.shape == input_copy.shape

    def test_abs_stereo_preserves_shape(self, stereo_sine_440_880hz_dask: tuple[DaArray, int]) -> None:
        """Stereo signal shape preserved."""
        dask_input, sr = stereo_sine_440_880hz_dask
        abs_op = ABS(sr)
        result = abs_op.process(dask_input).compute()
        assert result.shape == dask_input.compute().shape

    # -- Layer 3: Integration (numpy equivalence) --------------------------

    def test_abs_stereo_matches_numpy_abs(self, stereo_sine_440_880hz_dask: tuple[DaArray, int]) -> None:
        """ABS result must exactly match np.abs (wrapper equivalence)."""
        dask_input, sr = stereo_sine_440_880hz_dask
        abs_op = ABS(sr)

        result = abs_op.process(dask_input).compute()
        raw = dask_input.compute()

        # All values non-negative
        assert np.all(result >= 0)

        # Exact match with numpy — same algorithm
        np.testing.assert_allclose(result, np.abs(raw))


class TestPowerOperation:
    """Power operation: Layer 1 (unit) + Layer 2 (domain) + Layer 3 (numpy equivalence)."""

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_power_init_stores_exponent(self) -> None:
        """Test Power stores custom exponent."""
        power_op = Power(_SR, exponent=3.0)
        assert power_op.sampling_rate == _SR
        assert power_op.exp == 3.0

    def test_power_registry_returns_correct_class(self) -> None:
        """Test Power is registered as 'power'."""
        assert get_operation("power") == Power
        power_op = create_operation("power", _SR, exponent=3.0)
        assert isinstance(power_op, Power)
        assert power_op.exp == 3.0

    # -- Layer 2: Domain (shape + immutability) ----------------------------

    def test_power_mono_preserves_shape_and_immutability(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Mono signal shape preserved; input unchanged after power=2."""
        dask_input, sr = pure_sine_440hz_dask
        power_op = Power(sr, exponent=2.0)
        input_copy = dask_input.compute().copy()

        # Act
        result_da = power_op.process(dask_input)

        # Assert 1: Immutability
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)

        # Assert 2: Dask graph preserved
        assert isinstance(result_da, DaArray)

        # Assert 3: Shape
        result = result_da.compute()
        assert result.shape == input_copy.shape

    # -- Layer 3: Integration (numpy equivalence) --------------------------

    def test_power_stereo_squared_matches_numpy(self, stereo_sine_440_880hz_dask: tuple[DaArray, int]) -> None:
        """Power(2.0) must exactly match np.power(x, 2.0)."""
        dask_input, sr = stereo_sine_440_880hz_dask
        power_op = Power(sr, exponent=2.0)

        result = power_op.process(dask_input).compute()
        expected = np.power(dask_input.compute(), 2.0)
        # Same algorithm, exact match expected
        np.testing.assert_allclose(result, expected)

    @pytest.mark.filterwarnings("ignore:invalid value encountered in:RuntimeWarning")
    def test_power_sqrt_matches_numpy(self, stereo_sine_440_880hz_dask: tuple[DaArray, int]) -> None:
        """Power(0.5) must match np.sqrt for the same input (fractional exponent)."""
        dask_input, sr = stereo_sine_440_880hz_dask
        power_op = Power(sr, exponent=0.5)

        result = power_op.process(dask_input).compute()
        expected = np.sqrt(dask_input.compute())
        # sqrt of negative values yields NaN — same behavior expected
        np.testing.assert_allclose(result, expected)

    def test_power_reciprocal_matches_numpy(self) -> None:
        """Power(-1.0) must match 1/x for nonzero input."""
        nonzero = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
        dask_input = da_from_array(nonzero, chunks=(1, -1))
        power_op = Power(_SR, exponent=-1.0)

        result = power_op.process(dask_input).compute()
        expected = 1.0 / nonzero
        # Same algorithm, exact match expected
        np.testing.assert_allclose(result, expected)


class TestSum:
    """Sum operation: Layer 1 (unit) + Layer 2 (domain) + Layer 3 (numpy equivalence)."""

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_sum_init_stores_sample_rate(self) -> None:
        """Test Sum stores sampling rate."""
        sum_op = Sum(_SR)
        assert sum_op.sampling_rate == _SR

    def test_sum_registry_returns_correct_class(self) -> None:
        """Test Sum is registered as 'sum'."""
        assert get_operation("sum") == Sum
        sum_op = create_operation("sum", _SR)
        assert isinstance(sum_op, Sum)

    # -- Layer 2: Domain (shape reduction + immutability) -------------------

    def test_sum_mono_identity_and_immutability(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Mono summed with itself should be identity; input unchanged."""
        dask_input, sr = pure_sine_440hz_dask
        sum_op = Sum(sr)
        input_copy = dask_input.compute().copy()

        # Act
        result_da = sum_op.process(dask_input)

        # Assert 1: Immutability
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)

        # Assert 2: Dask graph preserved
        assert isinstance(result_da, DaArray)

        # Assert 3: Mono identity
        result = result_da.compute()
        assert result.shape == input_copy.shape
        np.testing.assert_allclose(result, input_copy)  # Same algorithm, exact match

    def test_sum_stereo_reduces_to_mono(self, stereo_sine_440_880hz_dask: tuple[DaArray, int]) -> None:
        """Stereo signal summed to (1, samples)."""
        dask_input, sr = stereo_sine_440_880hz_dask
        sum_op = Sum(sr)
        raw = dask_input.compute()

        result = sum_op.process(dask_input).compute()
        assert result.shape == (1, raw.shape[1])

    # -- Layer 3: Integration (numpy equivalence) --------------------------

    def test_sum_multichannel_matches_numpy_sum(self, stereo_sine_440_880hz_dask: tuple[DaArray, int]) -> None:
        """Sum must match np.sum(axis=0, keepdims=True)."""
        dask_input, sr = stereo_sine_440_880hz_dask
        sum_op = Sum(sr)

        result = sum_op.process(dask_input).compute()
        expected = np.sum(dask_input.compute(), axis=0, keepdims=True)
        np.testing.assert_allclose(result, expected)  # Same algorithm, exact match


class TestMean:
    """Mean operation: Layer 1 (unit) + Layer 2 (domain) + Layer 3 (numpy equivalence)."""

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_mean_init_stores_sample_rate(self) -> None:
        """Test Mean stores sampling rate."""
        mean_op = Mean(_SR)
        assert mean_op.sampling_rate == _SR

    def test_mean_registry_returns_correct_class(self) -> None:
        """Test Mean is registered as 'mean'."""
        assert get_operation("mean") == Mean
        mean_op = create_operation("mean", _SR)
        assert isinstance(mean_op, Mean)

    # -- Layer 2: Domain (shape + immutability) ----------------------------

    def test_mean_mono_identity_and_immutability(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Mono mean is identity; input unchanged."""
        dask_input, sr = pure_sine_440hz_dask
        mean_op = Mean(sr)
        input_copy = dask_input.compute().copy()

        # Act
        result_da = mean_op.process(dask_input)

        # Assert 1: Immutability
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)

        # Assert 2: Dask graph preserved
        assert isinstance(result_da, DaArray)

        # Assert 3: Mono identity
        result = result_da.compute()
        assert result.shape == input_copy.shape
        np.testing.assert_allclose(result, input_copy)

    def test_mean_stereo_reduces_to_mono(self, stereo_sine_440_880hz_dask: tuple[DaArray, int]) -> None:
        """Stereo mean reduces to (1, samples)."""
        dask_input, sr = stereo_sine_440_880hz_dask
        mean_op = Mean(sr)
        raw = dask_input.compute()

        result = mean_op.process(dask_input).compute()
        assert result.shape == (1, raw.shape[1])

    # -- Layer 3: Integration (numpy equivalence) --------------------------

    def test_mean_multichannel_matches_numpy_mean(self, stereo_sine_440_880hz_dask: tuple[DaArray, int]) -> None:
        """Mean must match np.mean(axis=0, keepdims=True)."""
        dask_input, sr = stereo_sine_440_880hz_dask
        mean_op = Mean(sr)

        result = mean_op.process(dask_input).compute()
        expected = dask_input.compute().mean(axis=0, keepdims=True)
        np.testing.assert_allclose(result, expected)  # Same algorithm, exact match


class TestChannelDifference:
    """ChannelDifference operation: Layer 1 + Layer 2 + Layer 3."""

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_channel_diff_init_stores_reference_channel(self) -> None:
        """Test ChannelDifference stores other_channel."""
        diff_op = ChannelDifference(_SR, other_channel=1)
        assert diff_op.sampling_rate == _SR
        assert diff_op.other_channel == 1

    def test_channel_diff_registry_returns_correct_class(self) -> None:
        """Test ChannelDifference is registered as 'channel_difference'."""
        assert get_operation("channel_difference") == ChannelDifference
        diff_op = create_operation("channel_difference", _SR, other_channel=1)
        assert isinstance(diff_op, ChannelDifference)
        assert diff_op.other_channel == 1

    # -- Layer 2: Domain (shape + immutability) ----------------------------

    def test_channel_diff_stereo_preserves_shape_and_immutability(
        self, stereo_sine_440_880hz_dask: tuple[DaArray, int]
    ) -> None:
        """Stereo difference preserves shape; input unchanged."""
        dask_input, sr = stereo_sine_440_880hz_dask
        diff_op = ChannelDifference(sr, other_channel=0)
        input_copy = dask_input.compute().copy()

        # Act
        result_da = diff_op.process(dask_input)

        # Assert 1: Immutability
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)

        # Assert 2: Dask graph preserved
        assert isinstance(result_da, DaArray)

        # Assert 3: Shape
        result = result_da.compute()
        assert result.shape == input_copy.shape

    # -- Layer 3: Integration (numpy equivalence) --------------------------

    def test_channel_diff_quad_ref0_matches_numpy_broadcast(self) -> None:
        """ChannelDifference(ref=0) must match manual subtraction.

        Quad-channel: [ones, zeros, twos, neg-ones].
        Expected diffs: [0, -1, 1, -2].
        """
        quad = np.array(
            [
                np.ones(_SR),  # ch0: reference
                np.zeros(_SR),  # ch1
                np.ones(_SR) * 2,  # ch2
                np.ones(_SR) * -1,  # ch3
            ]
        )
        dask_input = da_from_array(quad, chunks=(1, -1))
        diff_op = ChannelDifference(_SR, other_channel=0)

        result = diff_op.process(dask_input).compute()
        expected = quad - quad[0]
        np.testing.assert_allclose(result, expected)  # Same algorithm, exact match

        # Spot-check specific channels for clarity
        np.testing.assert_allclose(result[0], np.zeros(_SR))  # 1 - 1 = 0
        np.testing.assert_allclose(result[1], -np.ones(_SR))  # 0 - 1 = -1
        np.testing.assert_allclose(result[2], np.ones(_SR))  # 2 - 1 = 1
        np.testing.assert_allclose(result[3], -2 * np.ones(_SR))  # -1 - 1 = -2

    def test_channel_diff_quad_ref2_matches_numpy_broadcast(self) -> None:
        """ChannelDifference(ref=2) with same quad signal."""
        quad = np.array(
            [
                np.ones(_SR),
                np.zeros(_SR),
                np.ones(_SR) * 2,
                np.ones(_SR) * -1,
            ]
        )
        dask_input = da_from_array(quad, chunks=(1, -1))
        diff_op = ChannelDifference(_SR, other_channel=2)

        result = diff_op.process(dask_input).compute()
        expected = quad - quad[2]
        np.testing.assert_allclose(result, expected)

        np.testing.assert_allclose(result[0], -np.ones(_SR))  # 1 - 2 = -1
        np.testing.assert_allclose(result[1], -2 * np.ones(_SR))  # 0 - 2 = -2
        np.testing.assert_allclose(result[2], np.zeros(_SR))  # 2 - 2 = 0
        np.testing.assert_allclose(result[3], -3 * np.ones(_SR))  # -1 - 2 = -3
