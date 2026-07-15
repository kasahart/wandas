import io
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, cast
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from matplotlib.axes import Axes
from scipy.io import wavfile

import wandas as wd
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan
from wandas.utils.types import NDArrayReal

_da_from_array = da.from_array

# --- Module-level deterministic test constants ---
_SAMPLE_RATE: int = 16000
_rng = np.random.default_rng(42)
_DATA_2CH: NDArrayReal = _rng.random((2, 16000))  # 2 channels, 1 second
_DASK_2CH: DaArray = _da_from_array(_DATA_2CH, chunks=(1, 4000))

# Stereo sine fixture shared by RMS and CrestFactor test classes
_t = np.linspace(0, 1, int(_SAMPLE_RATE), endpoint=False)
_SINE_CH0: NDArrayReal = np.sin(2 * np.pi * 440 * _t).astype(np.float64)
_SINE_CH1: NDArrayReal = (2.0 * np.sin(2 * np.pi * 440 * _t)).astype(np.float64)
_STEREO_SINE: NDArrayReal = np.array([_SINE_CH0, _SINE_CH1])


class TestChannelFrame:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: float = _SAMPLE_RATE
        self.data: NDArrayReal = _DATA_2CH
        self.dask_data: DaArray = _DASK_2CH
        self.channel_frame: ChannelFrame = ChannelFrame(
            data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

    def test_initialization(self) -> None:
        """Test that initialization doesn't compute the data."""
        with mock.patch.object(DaArray, "compute", return_value=self.data) as mock_compute:
            # Just creating the object shouldn't call compute
            cf: ChannelFrame = ChannelFrame(self.dask_data, self.sample_rate)

            # Verify compute hasn't been called
            mock_compute.assert_not_called()

            # Check properties that don't require computation
            assert cf.sampling_rate == self.sample_rate
            assert cf.n_channels == 2
            assert cf.n_samples == 16000
            assert cf.duration == 1.0

            # Still no computation should have happened
            mock_compute.assert_not_called()

    def test_data_access_triggers_compute(self) -> None:
        """Test that accessing .data triggers computation."""
        with mock.patch.object(DaArray, "compute", return_value=self.data) as mock_compute:
            _: NDArrayReal = self.channel_frame.data
            mock_compute.assert_called_once()

    def test_mix_with_snr_scales_noise_and_stays_lazy(self) -> None:
        signal_data = np.ones((1, 8))
        signal = ChannelFrame(_da_from_array(signal_data, chunks=(1, 4)), self.sample_rate)
        noise = _da_from_array(np.ones(8), chunks=4)
        history_before = list(signal.operation_history)

        result = signal.mix(noise, snr_db=6.0)

        assert result is not signal
        assert isinstance(result._data, DaArray)
        assert signal.operation_history == history_before
        assert result.operation_history == [
            {
                "operation": "wandas.audio.mix",
                "version": 1,
                "params": {"snr_db": 6.0},
            }
        ]
        expected_noise_scale = 10 ** (-6.0 / 20.0)
        np.testing.assert_allclose(result.data, np.full(8, 1.0 + expected_noise_scale))

    def test_compute_method(self) -> None:
        """Test explicit compute method."""
        with mock.patch.object(DaArray, "compute", return_value=self.data) as mock_compute:
            result: NDArrayReal = self.channel_frame.compute()
            mock_compute.assert_called_once()
            np.testing.assert_array_equal(result, self.data)

    def test_time(self) -> None:
        """Test time property."""
        with mock.patch.object(DaArray, "compute", return_value=self.data) as mock_compute:
            time: NDArrayReal = self.channel_frame.time
            mock_compute.assert_not_called()
            expected_time = np.arange(16000) / 16000
            np.testing.assert_array_equal(time, expected_time)

    def test_source_time_adds_source_time_offset(self) -> None:
        """Test source-relative time property."""
        cf = ChannelFrame(self.dask_data, self.sample_rate, source_time_offset=2.5)

        np.testing.assert_array_equal(cf.source_time_offset, np.array([2.5, 2.5]))
        np.testing.assert_array_equal(cf.source_time, cf.time[None, :] + np.array([[2.5], [2.5]]))
        xr = cf.to_xarray()
        np.testing.assert_array_equal(xr.coords["source_time_offset"].values, np.array([2.5, 2.5]))
        assert "source_time_offset" not in xr.attrs

    def test_source_time_offset_accepts_per_channel_values(self) -> None:
        """source_time_offset is stored per channel."""
        cf = ChannelFrame(self.dask_data, self.sample_rate, source_time_offset=[2.5, 7.5])

        np.testing.assert_array_equal(cf.source_time_offset, np.array([2.5, 7.5]))
        np.testing.assert_array_equal(cf.source_time[:, 0], np.array([2.5, 7.5]))

    def test_source_time_offset_rejects_wrong_channel_count(self) -> None:
        """source_time_offset arrays must match the channel count."""
        with pytest.raises(ValueError, match="source_time_offset length must match number of channels"):
            ChannelFrame(self.dask_data, self.sample_rate, source_time_offset=[1.0])

    def test_source_time_offset_rejects_multidimensional_values(self) -> None:
        """source_time_offset must be scalar or channel-wise 1D."""
        with pytest.raises(ValueError, match="source_time_offset must be a scalar or a 1D array"):
            ChannelFrame(self.dask_data, self.sample_rate, source_time_offset=np.zeros((2, 1)))

    def test_source_time_offset_rejects_non_finite_values(self) -> None:
        """source_time_offset must remain finite."""
        cf = ChannelFrame(self.dask_data, self.sample_rate)

        with pytest.raises(ValueError, match="source_time_offset must be finite"):
            cf.source_time_offset = float("nan")
        with pytest.raises(ValueError, match="source_time_offset must be finite"):
            cf.source_time_offset = float("inf")
        with pytest.raises(TypeError, match="source_time_offset must be a finite numeric value"):
            cast(Any, cf).source_time_offset = "not-a-number"

    def test_time_slice_advances_source_time_offset(self) -> None:
        """Continuous sample slicing advances source-relative time."""
        result = self.channel_frame[:, 500:1500]

        np.testing.assert_array_equal(result.source_time_offset, np.array([500 / self.sample_rate] * 2))
        np.testing.assert_array_equal(result.source_time[:, 0], np.array([500 / self.sample_rate] * 2))

    def test_stepped_time_slice_raises_for_source_time_offset(self) -> None:
        """Stepped sample slicing would make source_time spacing ambiguous."""
        with pytest.raises(ValueError, match="Only continuous forward slicing on the time axis is supported"):
            _ = self.channel_frame[:, 500::2]

    def test_point_time_selection_requires_rank_preserving_slice(self) -> None:
        """Point time selection must use a slice to preserve Frame rank."""
        with pytest.raises(ValueError, match="Only slice selectors on non-channel axes are supported"):
            _ = self.channel_frame[0, 500]

    def test_fancy_time_selection_requires_slice_selector(self) -> None:
        """Fancy time selection is outside the non-channel slice-only contract."""
        with pytest.raises(ValueError, match="Only slice selectors on non-channel axes are supported"):
            _ = self.channel_frame[:, [500, 700, 900]]

    def test_positional_previous_constructor_argument_remains_compatible_after_history_removal(self) -> None:
        """previous remains compatible at its new positional slot after removing legacy history."""
        previous = ChannelFrame(self.dask_data, self.sample_rate)

        cf = ChannelFrame(self.dask_data, self.sample_rate, None, None, None, None, previous)

        assert cf.previous is previous
        np.testing.assert_array_equal(cf.source_time_offset, np.array([0.0, 0.0]))

    def test_constructor_rejects_legacy_history_provenance_kwargs(self) -> None:
        """Legacy history/provenance names are not constructor state sources."""
        with pytest.raises(TypeError, match="unexpected keyword argument 'operation_history'"):
            ChannelFrame(self.dask_data, self.sample_rate, operation_history=[])  # ty: ignore[unknown-argument]
        with pytest.raises(TypeError, match="unexpected keyword argument 'operation_graph'"):
            ChannelFrame(self.dask_data, self.sample_rate, operation_graph={})  # ty: ignore[unknown-argument]
        with pytest.raises(TypeError, match="unexpected keyword argument 'operations'"):
            ChannelFrame(self.dask_data, self.sample_rate, operations=[])  # ty: ignore[unknown-argument]

    def test_binary_op_allows_source_time_offset_mismatch_and_inherits_left_offset(self) -> None:
        """Frame-frame binary ops are index-wise and keep the left source timeline."""
        left = ChannelFrame(self.dask_data, self.sample_rate, source_time_offset=2.0)
        right = ChannelFrame(self.dask_data, self.sample_rate, source_time_offset=7.0)

        result = left + right

        np.testing.assert_array_equal(result.compute(), self.data + self.data)
        np.testing.assert_array_equal(result.source_time_offset, np.array([2.0, 2.0]))

    def test_channel_difference_allows_per_channel_source_time_offset_mismatch(self) -> None:
        """channel_difference is index-wise within one frame and keeps input offsets."""
        frame = ChannelFrame(
            self.dask_data,
            self.sample_rate,
            source_time_offset=[2.0, 7.0],
        )

        result = frame.channel_difference(other_channel=0)

        expected = self.data - self.data[0]
        np.testing.assert_array_equal(result.compute(), expected)
        np.testing.assert_array_equal(result.source_time_offset, np.array([2.0, 7.0]))

    def test_operations_are_lazy(self) -> None:
        """Test that operations don't trigger immediate computation."""
        with mock.patch.object(DaArray, "compute", return_value=self.data) as mock_compute:
            # Operations should build the graph but not compute
            result: ChannelFrame = self.channel_frame + 1
            result = result * 2
            result = result.abs()
            result = result.power(2)

            # Verify no computation happened
            mock_compute.assert_not_called()

            # Check that the result has the expected type
            assert isinstance(result, ChannelFrame)
            assert isinstance(result._data, DaArray)

    def test_operation_results(self) -> None:
        """Test that operations produce correct results when computed."""
        # Apply operations
        result: ChannelFrame = self.channel_frame + 1
        result = result * 2

        # Compute and check results
        computed: NDArrayReal = result.compute()
        expected: NDArrayReal = (self.data + 1) * 2
        # Scalar arithmetic on float64 — decimal=6 default (exact match expected)
        np.testing.assert_array_almost_equal(computed, expected)

    def test_persist(self) -> None:
        """Test that persist triggers computation but returns a new ChannelFrame."""
        with mock.patch.object(DaArray, "persist", return_value=self.dask_data) as mock_persist:
            result: ChannelFrame = self.channel_frame.persist()
            mock_persist.assert_called_once()
            assert isinstance(result, ChannelFrame)
            assert result is not self.channel_frame

    def test_channel_extraction(self) -> None:
        """Test extracting a channel works lazily."""
        with mock.patch.object(DaArray, "compute", return_value=self.data[0:1]) as mock_compute:
            channel: ChannelFrame = self.channel_frame.get_channel(0)
            mock_compute.assert_not_called()

            # Check that properties are correctly set
            assert channel.n_channels == 1
            assert channel.sampling_rate == self.sample_rate

            # Access data to trigger computation
            _: NDArrayReal = channel.data
            mock_compute.assert_called_once()

    def test_get_channel_single_int(self) -> None:
        """Test get_channel with single integer index."""
        # Test positive index
        channel = self.channel_frame.get_channel(0)
        assert isinstance(channel, ChannelFrame)
        assert channel.n_channels == 1
        assert channel.channels[0].label == "ch0"
        np.testing.assert_array_equal(channel.data, self.data[0])

        channel = self.channel_frame.get_channel(1)
        assert isinstance(channel, ChannelFrame)
        assert channel.n_channels == 1
        assert channel.channels[0].label == "ch1"
        np.testing.assert_array_equal(channel.data, self.data[1])

        # Test negative index
        channel = self.channel_frame.get_channel(-1)
        assert isinstance(channel, ChannelFrame)
        assert channel.n_channels == 1
        assert channel.channels[0].label == "ch1"
        np.testing.assert_array_equal(channel.data, self.data[-1])

        channel = self.channel_frame.get_channel(-2)
        assert isinstance(channel, ChannelFrame)
        assert channel.n_channels == 1
        assert channel.channels[0].label == "ch0"
        np.testing.assert_array_equal(channel.data, self.data[-2])

    def test_get_channel_list_of_ints(self) -> None:
        """Test get_channel with list of integer indices."""
        # Test with list of multiple indices
        channels = self.channel_frame.get_channel([0, 1])
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 2
        assert channels.channels[0].label == "ch0"
        assert channels.channels[1].label == "ch1"
        np.testing.assert_array_equal(channels.data, self.data[[0, 1]])

        # Test with reversed order
        channels = self.channel_frame.get_channel([1, 0])
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 2
        assert channels.channels[0].label == "ch1"
        assert channels.channels[1].label == "ch0"
        np.testing.assert_array_equal(channels.data, self.data[[1, 0]])

        # Test with single element list
        channels = self.channel_frame.get_channel([0])
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 1
        assert channels.channels[0].label == "ch0"
        # Single channel .data is squeezed to 1D
        np.testing.assert_array_equal(channels.data, self.data[0])

        # Test with negative indices
        channels = self.channel_frame.get_channel([-1, -2])
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 2
        assert channels.channels[0].label == "ch1"
        assert channels.channels[1].label == "ch0"
        np.testing.assert_array_equal(channels.data, self.data[[-1, -2]])

    def test_get_channel_tuple_of_ints(self) -> None:
        """Test get_channel with tuple of integer indices."""
        # Test with tuple
        channels = self.channel_frame.get_channel((0, 1))
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 2
        assert channels.channels[0].label == "ch0"
        assert channels.channels[1].label == "ch1"
        np.testing.assert_array_equal(channels.data, self.data[[0, 1]])

        # Test with single element tuple
        channels = self.channel_frame.get_channel((1,))
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 1
        assert channels.channels[0].label == "ch1"
        # Single channel .data is squeezed to 1D
        np.testing.assert_array_equal(channels.data, self.data[1])

    def test_get_channel_numpy_array(self) -> None:
        """Test get_channel with numpy array of indices."""
        # Test with numpy array
        indices = np.array([0, 1])
        channels = self.channel_frame.get_channel(indices)
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 2
        np.testing.assert_array_equal(channels.data, self.data[[0, 1]])

        # Test with single element numpy array
        indices = np.array([0])
        channels = self.channel_frame.get_channel(indices)
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 1
        # Single channel .data is squeezed to 1D
        np.testing.assert_array_equal(channels.data, self.data[0])

        # Test with negative indices in numpy array
        indices = np.array([-1, -2])
        channels = self.channel_frame.get_channel(indices)
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 2
        np.testing.assert_array_equal(channels.data, self.data[[-1, -2]])

    def test_get_channel_boolean_numpy_array(self) -> None:
        """Test get_channel with a 1-D boolean mask."""
        mask = np.array([False, True])
        channels = self.channel_frame.get_channel(mask)

        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 1
        assert channels.channels[0].label == "ch1"
        assert channels.operation_history[-1] == {
            "operation": "wandas.frame.get_channel",
            "version": 1,
            "params": {
                "channel_idx": {
                    "indexing": "boolean_mask",
                    "mask": [False, True],
                }
            },
        }
        assert isinstance(channels._data, DaArray)
        np.testing.assert_array_equal(channels.data, self.data[1])

    @pytest.mark.parametrize(
        ("mask", "match"),
        [
            (np.array([True]), "Boolean mask length"),
            (np.array([[False, True]]), "Channel selector must be 1-D"),
            (np.array([[False], [True]]), "Channel selector must be 1-D"),
        ],
    )
    def test_get_channel_boolean_numpy_array_rejects_invalid_masks(
        self,
        mask: np.ndarray[Any, Any],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            self.channel_frame.get_channel(mask)

    def test_get_channel_with_range(self) -> None:
        """Test get_channel with range object."""
        # Create a frame with more channels
        data = np.random.default_rng(42).random((4, 16000))
        dask_data = _da_from_array(data, chunks=(1, 4000))
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate, label="test_audio")

        # Test with range
        channels = frame.get_channel(list(range(3)))
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 3
        assert channels.channels[0].label == "ch0"
        assert channels.channels[1].label == "ch1"
        assert channels.channels[2].label == "ch2"
        np.testing.assert_array_equal(channels.data, data[[0, 1, 2]])

    def test_get_channel_preserves_metadata(self) -> None:
        """Test that get_channel preserves metadata correctly."""
        # Set metadata
        self.channel_frame.channels[0].label = "left"
        self.channel_frame.channels[0]["gain"] = 0.5
        self.channel_frame.channels[1].label = "right"
        self.channel_frame.channels[1]["gain"] = 0.75

        # Get single channel
        channel = self.channel_frame.get_channel(0)
        assert channel.channels[0].label == "left"
        assert channel.channels[0]["gain"] == 0.5

        # Get multiple channels
        channels = self.channel_frame.get_channel([0, 1])
        assert channels.channels[0].label == "left"
        assert channels.channels[0]["gain"] == 0.5
        assert channels.channels[1].label == "right"
        assert channels.channels[1]["gain"] == 0.75

        # Get channels in reverse order
        channels = self.channel_frame.get_channel([1, 0])
        assert channels.channels[0].label == "right"
        assert channels.channels[0]["gain"] == 0.75
        assert channels.channels[1].label == "left"
        assert channels.channels[1]["gain"] == 0.5

    def test_get_channel_is_lazy(self) -> None:
        """Test that get_channel operations remain lazy."""
        with mock.patch.object(DaArray, "compute", return_value=self.data) as mock_compute:
            # Single channel
            _ = self.channel_frame.get_channel(0)
            mock_compute.assert_not_called()

            # Multiple channels
            channels = self.channel_frame.get_channel([0, 1])
            mock_compute.assert_not_called()

            # Only accessing .data should trigger compute
            _ = channels.data
            mock_compute.assert_called_once()

    def test_plotting_triggers_compute(self) -> None:
        """Test that plotting triggers computation."""
        with mock.patch("wandas.visualization.plotting.create_operation") as mock_get_strategy:
            mock_strategy: mock.MagicMock = mock.MagicMock()
            mock_get_strategy.return_value = mock_strategy

            # Create a mock for the compute method
            with mock.patch.object(self.channel_frame, "compute", return_value=self.data) as mock_compute:
                mock_ax: mock.MagicMock = mock.MagicMock()
                _: Axes | Any = self.channel_frame.plot(plot_type="waveform", ax=mock_ax)

                # Verify compute was called
                mock_compute.assert_not_called()

                # Verify the strategy's plot method was called
                mock_strategy.plot.assert_called_once()

    def test_initialization_with_1d_data(self) -> None:
        """Test initialization with 1D data."""
        data_1d = np.random.default_rng(42).random(16000)
        dask_data_1d = _da_from_array(data_1d, chunks=4000)

        cf = ChannelFrame(dask_data_1d, self.sample_rate)

        # Check that the data was reshaped
        assert cf.shape == (16000,)
        assert cf.n_channels == 1

    def test_save_method(self) -> None:
        """Test saving audio to file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            # Test with multi-channel data
            with mock.patch("soundfile.write") as mock_write:
                with mock.patch.object(self.channel_frame, "compute", return_value=self.data):
                    self.channel_frame.to_wav(temp_filename)
                    mock_write.assert_called_once()

                    # Check that data was transposed for soundfile
                    args = mock_write.call_args[0]
                    assert args[0] == temp_filename
                    np.testing.assert_array_equal(args[1], self.data.T)
                    assert args[2] == int(self.sample_rate)

            # Test with single-channel data
            with mock.patch("soundfile.write") as mock_write:
                single_channel_data = self.data[0:1]
                channel_frame = ChannelFrame(
                    _da_from_array(single_channel_data, chunks=(1, 4000)),
                    self.sample_rate,
                )

                with mock.patch.object(channel_frame, "compute", return_value=single_channel_data):
                    channel_frame.to_wav(temp_filename)
                    mock_write.assert_called_once()

                    # Check that data was transposed and squeezed
                    args = mock_write.call_args[0]
                    np.testing.assert_array_equal(args[1], single_channel_data.T.squeeze(axis=1))
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def test_iter_method(self) -> None:
        """Test iterating over channels."""
        channels = list(self.channel_frame)

        assert len(channels) == 2
        for i, channel in enumerate(channels):
            assert isinstance(channel, ChannelFrame)
            assert channel.n_channels == 1
            assert channel.label == "test_audio"
            assert channel.sampling_rate == self.sample_rate
            assert channel.n_samples == 16000
            assert channel.channels[0].label == f"ch{i}"

    def test_array_method(self) -> None:
        """Test __array__ method for numpy conversion."""
        with mock.patch.object(self.channel_frame, "compute", return_value=self.data) as mock_compute:
            # Test with default dtype
            array = np.array(self.channel_frame)
            mock_compute.assert_called_once()
            np.testing.assert_array_equal(array, self.data)

            # Reset mock
            mock_compute.reset_mock()

            # Test with specified dtype
            array = np.array(self.channel_frame, dtype=np.float64)
            mock_compute.assert_called_once()
            assert array.dtype == np.float64

    def test_getitem_method(self) -> None:
        """Test __getitem__ method."""

        # Slice all channels
        result = self.channel_frame["ch0"]
        assert result is not self.channel_frame  # Pillar 1: immutability
        assert isinstance(result._data, DaArray)  # Pillar 1: Dask laziness
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.n_samples == 16000
        assert result.sampling_rate == self.sample_rate
        assert result.label == "test_audio"
        assert result.channels[0].label == "ch0"
        assert result.shape == (16000,)
        assert result.data.shape == (16000,)
        np.testing.assert_array_equal(result.data, self.data[0])

        result = self.channel_frame["ch1"]
        assert result is not self.channel_frame  # Pillar 1: immutability
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.n_samples == 16000
        assert result.sampling_rate == self.sample_rate
        assert result.label == "test_audio"
        assert result.channels[0].label == "ch1"


def test_init_with_3d_array_raises_value_error() -> None:
    # 3D arrays are not supported for ChannelFrame
    arr3 = np.zeros((1, 2, 3))
    dask3 = _da_from_array(arr3, chunks=(1, 1, 1))
    with pytest.raises(ValueError, match=r"Invalid data shape for ChannelFrame"):
        ChannelFrame(data=dask3, sampling_rate=16000)


def test_add_channel_align_strict_length_mismatch_raises() -> None:
    # base frame has 10 samples
    base = ChannelFrame(data=_da_from_array(np.zeros((1, 10)), chunks=(1, -1)), sampling_rate=16000)
    other = ChannelFrame(data=_da_from_array(np.zeros((1, 5)), chunks=(1, -1)), sampling_rate=16000)

    with pytest.raises(ValueError, match=r"Data length mismatch"):
        base.add_channel(other)  # default align='strict'


def test_add_channel_duplicate_label_without_suffix_raises() -> None:
    base = ChannelFrame(data=_da_from_array(np.zeros((1, 8)), chunks=(1, -1)), sampling_rate=16000)
    # add with explicit label equal to existing
    with pytest.raises(ValueError, match=r"Duplicate channel label"):
        base.add_channel(np.zeros(8), label="ch0")


def test_add_channel_returns_new_frame_without_mutating_original() -> None:
    base = ChannelFrame(data=_da_from_array(np.zeros((1, 6)), chunks=(1, -1)), sampling_rate=16000)
    orig_n = base.n_channels
    added = base.add_channel(np.zeros(6), label="new_ch")

    assert added is not base
    assert base.n_channels == orig_n
    assert base.labels == ["ch0"]
    assert base.operation_history == []
    assert added.n_channels == orig_n + 1
    assert added.labels == ["ch0", "new_ch"]
    assert added.operation_history[-1] == {
        "operation": "wandas.channel.add_channel",
        "version": 1,
        "params": {"label": "new_ch"},
    }


def test_add_channel_result_preserves_replayable_recipe_lineage() -> None:
    base = ChannelFrame(data=_da_from_array(np.zeros((1, 6)), chunks=(1, -1)), sampling_rate=16000)

    added = base.add_channel(np.ones(6), label="new_ch")

    recipe = RecipePlan.from_frame(added, input_names=("base", "new_data"))
    serialized = recipe.to_dict()
    restored = RecipePlan.from_dict(serialized)
    replayed = restored.apply(
        {
            "base": ChannelFrame(data=_da_from_array(np.zeros((1, 6)), chunks=(1, -1)), sampling_rate=16000),
            "new_data": np.ones(6),
        }
    )

    assert base.labels == ["ch0"]
    assert serialized["nodes"][0]["operation"] == "wandas.channel.add_channel"
    assert replayed.labels == ["ch0", "new_ch"]
    np.testing.assert_allclose(replayed.data, added.data)


def test_channel_update_helpers_preserve_source_time_offset() -> None:
    base = ChannelFrame(
        data=_da_from_array(np.zeros((1, 6)), chunks=(1, -1)),
        sampling_rate=16000,
        source_time_offset=2.5,
    )

    added = base.add_channel(np.zeros(6), label="new_ch")
    removed = added.remove_channel("new_ch")

    np.testing.assert_array_equal(added.source_time_offset, np.array([2.5, 0.0]))
    np.testing.assert_array_equal(removed.source_time_offset, np.array([2.5]))


def test_add_channel_numpy_raw_uses_explicit_source_time_offset() -> None:
    base = ChannelFrame(
        data=_da_from_array(np.zeros((1, 6)), chunks=(1, -1)),
        sampling_rate=16000,
        source_time_offset=2.5,
    )

    added = base.add_channel(np.zeros(6), label="new_ch", source_time_offset=5.0)

    np.testing.assert_array_equal(added.source_time_offset, np.array([2.5, 5.0]))
    np.testing.assert_array_equal(added.source_time[:, 0], np.array([2.5, 5.0]))


def test_add_channel_dask_raw_uses_explicit_source_time_offset_and_stays_lazy() -> None:
    base = ChannelFrame(
        data=_da_from_array(np.zeros((1, 6)), chunks=(1, -1)),
        sampling_rate=16000,
        source_time_offset=2.5,
    )
    raw_channel = _da_from_array(np.ones(6), chunks=3)
    history_before = list(base.operation_history)

    added = base.add_channel(raw_channel, label="new_ch", source_time_offset=[5.0])

    assert isinstance(added._data, DaArray)
    assert base.operation_history == history_before
    assert added.operation_history == [
        {
            "operation": "wandas.channel.add_channel",
            "version": 1,
            "params": {"label": "new_ch", "source_time_offset": [5.0]},
        }
    ]
    np.testing.assert_array_equal(added.source_time_offset, np.array([2.5, 5.0]))


@pytest.mark.parametrize(
    ("source_time_offset", "error_type", "match"),
    [
        ([1.0, 2.0], ValueError, "source_time_offset length must match number of channels"),
        (float("nan"), ValueError, "source_time_offset must be finite"),
        (float("inf"), ValueError, "source_time_offset must be finite"),
        ("not-a-number", TypeError, "source_time_offset must be a finite numeric value"),
        (np.zeros((1, 1)), ValueError, "source_time_offset must be a scalar or a 1D array"),
    ],
)
def test_add_channel_raw_source_time_offset_validation(
    source_time_offset: Any,
    error_type: type[Exception],
    match: str,
) -> None:
    base = ChannelFrame(data=_da_from_array(np.zeros((1, 6)), chunks=(1, -1)), sampling_rate=16000)

    with pytest.raises(error_type, match=match):
        base.add_channel(np.zeros(6), label="new_ch", source_time_offset=source_time_offset)


def test_add_channel_frame_preserves_per_channel_source_time_offsets() -> None:
    base = ChannelFrame(
        data=_da_from_array(np.zeros((1, 6)), chunks=(1, -1)),
        sampling_rate=16000,
        source_time_offset=2.5,
    )
    other = ChannelFrame(
        data=_da_from_array(np.ones((2, 6)), chunks=(1, -1)),
        sampling_rate=16000,
        channel_metadata=[{"label": "other0"}, {"label": "other1"}],
        source_time_offset=[5.0, 8.0],
    )

    added = base.add_channel(other)

    np.testing.assert_array_equal(added.source_time_offset, np.array([2.5, 5.0, 8.0]))
    np.testing.assert_array_equal(added.source_time[:, 0], np.array([2.5, 5.0, 8.0]))


def test_add_channel_frame_rejects_explicit_source_time_offset() -> None:
    base = ChannelFrame(
        data=_da_from_array(np.zeros((1, 6)), chunks=(1, -1)),
        sampling_rate=16000,
        source_time_offset=2.5,
    )
    other = ChannelFrame(
        data=_da_from_array(np.ones((1, 6)), chunks=(1, -1)),
        sampling_rate=16000,
        channel_metadata=[{"label": "other0"}],
        source_time_offset=5.0,
    )

    with pytest.raises(ValueError, match="source_time_offset cannot be used when adding a ChannelFrame"):
        base.add_channel(other, source_time_offset=9.0)


def test_add_channel_unsupported_type_raises() -> None:
    base = ChannelFrame(data=_da_from_array(np.zeros((1, 4)), chunks=(1, -1)), sampling_rate=16000)
    with pytest.raises(TypeError, match=r"data must be a ChannelFrame, NumPy array, or Dask array"):
        base.add_channel(12345)  # unsupported type  # ty: ignore[invalid-argument-type]


def test_add_channel_with_channelframe_align_pad_and_truncate() -> None:
    base = ChannelFrame(data=_da_from_array(np.zeros((1, 10)), chunks=(1, -1)), sampling_rate=16000)

    # shorter incoming frame -> pad
    other_short = ChannelFrame(data=_da_from_array(np.zeros((1, 5)), chunks=(1, -1)), sampling_rate=16000)
    # ensure labels won't collide with existing frame
    for ch in other_short._channel_metadata:
        ch.label = "other_ch"
    out = base.add_channel(other_short, align="pad")
    assert out is not base  # Pillar 1: immutability
    assert out.n_samples == base.n_samples
    assert out.n_channels == 2
    assert base.n_channels == 1  # Pillar 1: original unchanged
    assert out.operation_history[-1] == {
        "operation": "wandas.channel.add_channel",
        "version": 1,
        "params": {"align": "pad"},
    }

    # longer incoming frame -> truncate
    other_long = ChannelFrame(data=_da_from_array(np.zeros((1, 20)), chunks=(1, -1)), sampling_rate=16000)
    for ch in other_long._channel_metadata:
        ch.label = "other_ch_long"
    out2 = base.add_channel(other_long, align="truncate")
    assert out2 is not base  # Pillar 1: immutability
    assert out2.n_samples == base.n_samples
    assert out2.n_channels == 2
    assert out2.operation_history[-1] == {
        "operation": "wandas.channel.add_channel",
        "version": 1,
        "params": {"align": "truncate"},
    }


def test_remove_channel_index_out_of_range_raises() -> None:
    base = ChannelFrame(data=_da_from_array(np.zeros((2, 4)), chunks=(1, -1)), sampling_rate=16000)
    with pytest.raises(IndexError, match=r"index 5 out of range"):
        base.remove_channel(5)


def test_remove_channel_label_not_found_raises() -> None:
    base = ChannelFrame(data=_da_from_array(np.zeros((2, 4)), chunks=(1, -1)), sampling_rate=16000)
    with pytest.raises(KeyError, match=r"label no_such not found"):
        base.remove_channel("no_such")


def test_describe_plot_return_type_error() -> None:
    # Patch plotting strategy to return an unsupported type
    class FakeStrategy:
        def plot(self, *args, **kwargs):
            return 123  # invalid return type

    with mock.patch("wandas.visualization.plotting.create_operation", return_value=FakeStrategy()):
        cf = ChannelFrame(data=_da_from_array(np.zeros((1, 4)), chunks=(1, -1)), sampling_rate=16000)
        # describe should raise TypeError when plot returns invalid type
        with pytest.raises(TypeError, match=r"Unexpected type for plot result"):
            cf.describe()


class TestChannelFrameFileIO:
    """Public file and in-memory loading contracts."""

    def test_read_csv(self) -> None:
        """Test read_csv method."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_filename = temp_file.name
            # 時間と2つの値を持つCSVファイルを作成
            header = "time,value1,value2\n"
            data = "\n".join([f"{i / 16000},{1.1},{2.2}" for i in range(16000)])
            temp_file.write(header.encode())
            temp_file.write(data.encode())
            temp_file.flush()
            temp_file.seek(0)
            # Close the file to ensure it's written
            temp_file.close()

        try:
            # Read the CSV file into a ChannelFrame
            cf = ChannelFrame.read_csv(temp_filename)

            # Check properties
            assert cf.sampling_rate == 16000
            assert cf.n_channels == 2
            assert cf.n_samples == 16000
            assert cf.label == Path(temp_filename).stem

            # Check data
            expected_data = np.loadtxt(temp_filename, delimiter=",", skiprows=1).T
            np.testing.assert_array_equal(cf.data, expected_data[1:])

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def test_from_file_bytes_wav(self) -> None:
        """Test from_file with in-memory WAV bytes."""
        sampling_rate = 8000
        duration = 0.1
        num_samples = int(sampling_rate * duration)
        data_left = np.full(num_samples, 0.25, dtype=np.float32)
        data_right = np.full(num_samples, 0.75, dtype=np.float32)
        stereo_data = np.column_stack((data_left, data_right))

        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, sampling_rate, stereo_data)
        wav_bytes = wav_buffer.getvalue()

        cf = ChannelFrame.from_file(wav_bytes, file_type=".wav")

        assert cf.sampling_rate == sampling_rate
        assert cf.n_channels == 2
        computed_data = cf.compute()
        np.testing.assert_allclose(computed_data[0], data_left, rtol=1e-5)  # int16->float32 normalization rounding
        np.testing.assert_allclose(computed_data[1], data_right, rtol=1e-5)  # int16->float32 normalization rounding

    def test_from_file_bytes_csv(self) -> None:
        """Test from_file with in-memory CSV bytes."""
        header = "time,value1,value2\n"
        data = "\n".join([f"{i / 16000},{1.1},{2.2}" for i in range(160)])
        csv_bytes = (header + data).encode()

        cf = ChannelFrame.from_file(csv_bytes, file_type=".csv", time_column=0)

        assert cf.sampling_rate == 16000
        assert cf.n_channels == 2
        assert cf.n_samples == 160

        expected_data = np.loadtxt(io.BytesIO(csv_bytes), delimiter=",", skiprows=1).T
        np.testing.assert_array_equal(cf.data, expected_data[1:])

    def test_from_file_csv_uses_time_column_origin_for_source_time(self) -> None:
        """CSV source time starts at the selected time-column value."""
        header = "time,value1,value2\n"
        data = "\n".join([f"{10 + i / 10},{1.1},{2.2}" for i in range(20)])
        csv_bytes = (header + data).encode()

        cf = ChannelFrame.from_file(csv_bytes, file_type=".csv", time_column=0, start=0.5)

        assert cf.sampling_rate == 10
        assert cf.n_samples == 15
        np.testing.assert_array_equal(cf.source_time_offset, np.array([10.5, 10.5]))
        np.testing.assert_array_equal(cf.source_time[:, 0], np.array([10.5, 10.5]))

    def test_from_file_bytes_requires_file_type(self) -> None:
        """Test in-memory data requires file_type."""
        with pytest.raises(ValueError, match="File type is required when the extension is missing"):
            ChannelFrame.from_file(b"dummy")

    def test_from_file_stream_nonseekable(self) -> None:
        """Test from_file with non-seekable in-memory stream and file_type."""
        sampling_rate = 8000
        duration = 0.1
        num_samples = int(sampling_rate * duration)
        data_left = np.full(num_samples, 0.25, dtype=np.float32)
        data_right = np.full(num_samples, 0.75, dtype=np.float32)
        stereo_data = np.column_stack((data_left, data_right))

        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, sampling_rate, stereo_data)
        wav_bytes = wav_buffer.getvalue()

        class NonSeekableStream:
            def __init__(self, data: bytes, name: str) -> None:
                self._data = data
                self.name = name

            def read(self) -> bytes:
                return self._data

            def seek(self, *_args, **_kwargs) -> None:
                raise OSError("seek not supported")

        stream = NonSeekableStream(wav_bytes, name="memory/sample.wav")

        cf = ChannelFrame.from_file(
            cast(BinaryIO, stream),
            file_type="wav",
            source_name="source.wav",
        )

        assert cf.sampling_rate == sampling_rate
        assert cf.n_channels == 2
        assert cf.label == "source"
        computed_data = cf.compute()
        np.testing.assert_allclose(computed_data[0], data_left, rtol=1e-5)  # int16->float32 normalization rounding
        np.testing.assert_allclose(computed_data[1], data_right, rtol=1e-5)  # int16->float32 normalization rounding

    def test_from_file_wav_normalize_false(self, tmp_path: Path) -> None:
        """Test that from_file returns int16 data cast to float32 by default (normalize=False)."""
        filepath = tmp_path / "int16.wav"
        sampling_rate = 16000
        num_samples = 100
        int16_data = np.column_stack(
            [np.full(num_samples, 16384, dtype=np.int16), np.full(num_samples, -16384, dtype=np.int16)]
        )
        wavfile.write(str(filepath), sampling_rate, int16_data)

        cf = ChannelFrame.from_file(filepath)
        computed = cf.compute()

        np.testing.assert_array_equal(computed[0], np.full(num_samples, 16384, dtype=np.float32))
        np.testing.assert_array_equal(computed[1], np.full(num_samples, -16384, dtype=np.float32))
        assert computed.dtype == np.float32

    def test_from_file_wav_normalize_true(self, tmp_path: Path) -> None:
        """Test that from_file normalizes to float32 when normalize=True."""
        filepath = tmp_path / "int16_norm.wav"
        sampling_rate = 16000
        num_samples = 100
        int16_data = np.column_stack(
            [np.full(num_samples, 16384, dtype=np.int16), np.full(num_samples, -16384, dtype=np.int16)]
        )
        wavfile.write(str(filepath), sampling_rate, int16_data)

        cf = ChannelFrame.from_file(filepath, normalize=True)
        computed = cf.compute()

        np.testing.assert_allclose(computed[0], 0.5, rtol=1e-4)  # int16 quantization: 16384/32768 ~= 0.5
        np.testing.assert_allclose(computed[1], -0.5, rtol=1e-4)  # int16 quantization: 16384/32768 ~= 0.5
        assert computed.dtype == np.float32

    def test_read_wav_normalize_true(self, tmp_path: Path) -> None:
        """Test that read_wav normalizes to float32 when normalize=True."""
        filepath = tmp_path / "int16_norm.wav"
        sampling_rate = 16000
        num_samples = 100
        int16_data = np.column_stack(
            [np.full(num_samples, 16384, dtype=np.int16), np.full(num_samples, -16384, dtype=np.int16)]
        )
        wavfile.write(str(filepath), sampling_rate, int16_data)

        cf = ChannelFrame.read_wav(str(filepath), normalize=True)
        computed = cf.compute()

        np.testing.assert_allclose(computed[0], 0.5, rtol=1e-4)  # int16 quantization: 16384/32768 ~= 0.5
        np.testing.assert_allclose(computed[1], -0.5, rtol=1e-4)  # int16 quantization: 16384/32768 ~= 0.5
        assert computed.dtype == np.float32


class TestChannelFrameUtilities:
    """Debug, information, compatibility-wrapper, and graph contracts."""

    def setup_method(self) -> None:
        self.sample_rate = _SAMPLE_RATE
        self.data = _DATA_2CH
        self.channel_frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=self.sample_rate,
            label="test_audio",
        )

    def test_debug_info(self) -> None:
        """Test debug_info method."""
        with mock.patch("wandas.core.base_frame.logger") as mock_logger:
            self.channel_frame.debug_info()
            assert mock_logger.debug.call_count >= 6  # At least 6 debug messages

    def test_info_prints_basic_metadata(self, capsys: pytest.CaptureFixture[str]) -> None:
        """info() should print core frame metadata and labels."""
        self.channel_frame.info()

        captured = capsys.readouterr().out

        assert "ChannelFrame Information:" in captured
        assert f"Channels: {self.channel_frame.n_channels}" in captured
        assert f"Sampling rate: {self.channel_frame.sampling_rate} Hz" in captured
        assert f"Samples: {self.channel_frame.n_samples}" in captured
        assert f"Channel labels: {self.channel_frame.labels}" in captured

    def test_read_wav_class_method(self) -> None:
        """Test read_wav class method."""
        with mock.patch.object(ChannelFrame, "from_file") as mock_from_file:
            mock_from_file.return_value = self.channel_frame
            result = ChannelFrame.read_wav("test.wav", labels=["left", "right"])
            mock_from_file.assert_called_with(
                "test.wav", ch_labels=["left", "right"], normalize=False, file_type=None, source_name=None
            )
            assert result is self.channel_frame

    def test_read_wav_stream_source_name(self) -> None:
        """Test that read_wav propagates source_name from a file-like object's .name attribute."""
        with mock.patch.object(ChannelFrame, "from_file") as mock_from_file:
            mock_from_file.return_value = self.channel_frame

            stream = mock.MagicMock()
            stream.read = mock.MagicMock(return_value=b"")
            stream.name = "path/to/audio.wav"

            result = ChannelFrame.read_wav(stream)
            mock_from_file.assert_called_with(
                stream,
                ch_labels=None,
                normalize=False,
                file_type=".wav",
                source_name="path/to/audio.wav",
            )
            assert result is self.channel_frame

    def test_read_wav_stream_without_name(self) -> None:
        """Test that read_wav passes source_name=None when file-like has no .name."""
        with mock.patch.object(ChannelFrame, "from_file") as mock_from_file:
            mock_from_file.return_value = self.channel_frame

            stream = mock.MagicMock(spec=["read"])
            result = ChannelFrame.read_wav(stream)
            mock_from_file.assert_called_with(
                stream,
                ch_labels=None,
                normalize=False,
                file_type=".wav",
                source_name=None,
            )
            assert result is self.channel_frame

    def test_visualize_graph(self) -> None:
        """Test visualize_graph method."""
        # Test successful visualization - returns mock object from visualize()
        with mock.patch.object(DaArray, "visualize") as mock_visualize:
            mock_return_value = mock.MagicMock()
            mock_visualize.return_value = mock_return_value
            result = self.channel_frame.visualize_graph()
            mock_visualize.assert_called_once()
            # visualize_graph returns the result from _data.visualize()
            assert result is mock_return_value

        # Test with provided filename
        with mock.patch.object(DaArray, "visualize") as mock_visualize:
            mock_return_value = mock.MagicMock()
            mock_visualize.return_value = mock_return_value
            custom_filename = "test_graph.png"
            result = self.channel_frame.visualize_graph(filename=custom_filename)
            mock_visualize.assert_called_with(filename=custom_filename)
            # Returns the mock object from visualize()
            assert result is mock_return_value

        # Test handling of visualization error
        with mock.patch.object(DaArray, "visualize", side_effect=Exception("Test error")):
            result = self.channel_frame.visualize_graph()
            assert result is None


class TestChannelFrameRMS:
    """Focused 4-pillar tests for ChannelFrame.rms property."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = _SAMPLE_RATE
        self.stereo_data: NDArrayReal = _STEREO_SINE
        self.cf_stereo = ChannelFrame.from_numpy(self.stereo_data, sampling_rate=self.sample_rate)

    # ------------------------------------------------------------------
    # Pillar 1 – Immutability: computing rms must not modify the frame
    # ------------------------------------------------------------------

    def test_rms_does_not_mutate_data(self) -> None:
        """Computing rms must leave the underlying frame data unchanged."""
        original = self.stereo_data.copy()
        _ = self.cf_stereo.rms
        np.testing.assert_array_equal(self.cf_stereo.compute(), original)

    def test_rms_does_not_mutate_operation_history(self) -> None:
        """Computing rms must not append to operation_history."""
        history_before = list(self.cf_stereo.operation_history)
        _ = self.cf_stereo.rms
        assert self.cf_stereo.operation_history == history_before

    # ------------------------------------------------------------------
    # Pillar 2 – Metadata sync: return type and shape
    # ------------------------------------------------------------------

    def test_rms_returns_numpy_array(self) -> None:
        """rms must return a concrete numpy ndarray (not a dask array)."""
        result = self.cf_stereo.rms
        assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"

    def test_rms_shape_stereo(self) -> None:
        """rms of a 2-channel frame must have shape (2,)."""
        result = self.cf_stereo.rms
        assert result.shape == (2,), f"Expected (2,), got {result.shape}"

    def test_rms_shape_mono(self) -> None:
        """rms of a 1-channel frame must have shape (1,)."""
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        mono_data: NDArrayReal = np.sin(2 * np.pi * 440 * t).reshape(1, -1).astype(np.float64)
        cf_mono = ChannelFrame.from_numpy(mono_data, sampling_rate=self.sample_rate)
        result = cf_mono.rms
        assert result.shape == (1,), f"Expected (1,), got {result.shape}"

    # ------------------------------------------------------------------
    # Pillar 3 – Mathematical consistency: known reference values
    # ------------------------------------------------------------------

    def test_rms_sine_wave_equals_amplitude_over_sqrt2(self) -> None:
        """RMS of A*sin(2πft) must equal A / sqrt(2)."""
        result = self.cf_stereo.rms
        expected = np.array([1.0 / np.sqrt(2), 2.0 / np.sqrt(2)])
        # 440 Hz at 16 kHz SR produces 440 integer cycles; rtol=1e-6 for standard float64 precision.
        np.testing.assert_allclose(result, expected, rtol=1e-6)  # Pure sine RMS = amplitude / sqrt(2)

    def test_rms_dc_signal_equals_amplitude(self) -> None:
        """RMS of a constant (DC) signal must equal its absolute amplitude."""
        dc_data: NDArrayReal = np.array([[3.0] * 1000, [-5.0] * 1000])
        cf_dc = ChannelFrame.from_numpy(dc_data, sampling_rate=self.sample_rate)
        result = cf_dc.rms
        # Constant signal; no rounding error beyond float64 machine precision.
        np.testing.assert_allclose(result, [3.0, 5.0], rtol=1e-10)  # Exact: RMS of constant signal equals the constant

    def test_rms_zero_signal_is_zero(self) -> None:
        """RMS of an all-zero signal must be 0."""
        zero_data: NDArrayReal = np.zeros((2, 1000))
        cf_zero = ChannelFrame.from_numpy(zero_data, sampling_rate=self.sample_rate)
        result = cf_zero.rms
        # Exact zero input; allow for float64 subnormal rounding floor.
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-15)  # Near-zero: RMS of all-zeros signal

    def test_rms_integer_input_matches_float64_reference(self) -> None:
        """RMS must cast integer input to float64 before squaring."""
        int_data: NDArrayReal = np.array([[-32768, 0], [30000, -30000]], dtype=np.int16)
        cf_int = ChannelFrame.from_numpy(int_data, sampling_rate=self.sample_rate)

        result = cf_int.rms
        reference = np.sqrt(np.mean(int_data.astype(np.float64) ** 2, axis=1))

        np.testing.assert_allclose(result, reference, rtol=1e-12)  # Reference: direct np.sqrt(mean(x**2))

    def test_rms_mixed_samples(self) -> None:
        """RMS of [1, 2, 3, 4] must equal sqrt(mean([1,4,9,16])) = sqrt(7.5)."""
        data: NDArrayReal = np.array([[1.0, 2.0, 3.0, 4.0]])
        cf = ChannelFrame.from_numpy(data, sampling_rate=100)
        result = cf.rms
        expected = np.sqrt(np.mean(np.array([1.0, 4.0, 9.0, 16.0])))
        # Exact rational inputs; float64 precision gives near bit-identical result.
        np.testing.assert_allclose(result, [expected], rtol=1e-10)  # Single-channel pure sine RMS

    # ------------------------------------------------------------------
    # Pillar 4 – Reference-based verification: boolean channel selection
    # ------------------------------------------------------------------

    def test_rms_boolean_indexing_filters_channels(self) -> None:
        """Boolean mask from rms must correctly select channels."""
        data: NDArrayReal = np.array([[1.0] * 1000, [2.0] * 1000, [3.0] * 1000])
        cf = ChannelFrame.from_numpy(data, sampling_rate=self.sample_rate)

        mask = cf.rms > 1.5
        expected_mask = np.array([False, True, True])
        np.testing.assert_array_equal(mask, expected_mask)

        filtered = cf[mask]
        assert filtered.n_channels == 2
        # Constant-signal DC channels; float64 precision gives near bit-identical result.
        np.testing.assert_allclose(filtered.rms, [2.0, 3.0], rtol=1e-10)  # Exact: RMS of constant signal

    def test_rms_matches_numpy_reference(self) -> None:
        """rms must match the explicit numpy reference calculation."""
        rng = np.random.default_rng(42)
        data: NDArrayReal = rng.standard_normal((3, 8000)).astype(np.float64)
        cf = ChannelFrame.from_numpy(data, sampling_rate=self.sample_rate)

        result = cf.rms
        reference = np.sqrt(np.mean(data**2, axis=1))
        # Same algorithm and dtype; results should be bit-identical (rtol=1e-10 guards rounding).
        np.testing.assert_allclose(result, reference, rtol=1e-10)  # Reference: direct np.sqrt(mean(x**2))

    def test_rms_n_channel_consistency(self) -> None:
        """len(cf.rms) must equal cf.n_channels for any frame."""
        for n_channels in [1, 3, 5]:
            data: NDArrayReal = np.ones((n_channels, 500))
            cf = ChannelFrame.from_numpy(data, sampling_rate=self.sample_rate)
            assert len(cf.rms) == cf.n_channels


class TestChannelFrameValidationMessages:
    """ChannelFrame validation errors explain what failed and how to fix it."""

    def setup_method(self) -> None:
        """テストフィクスチャのセットアップ"""
        self.sample_rate = _SAMPLE_RATE
        self.data = _DATA_2CH
        self.dask_data = _DASK_2CH
        self.channel_frame = ChannelFrame(data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio")

    def test_invalid_data_shape_error_message(self) -> None:
        """Test that invalid data shape provides helpful error message."""
        data_3d = np.random.default_rng(42).random((2, 3, 4))  # 3D array
        dask_data_3d = _da_from_array(data_3d)

        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data=dask_data_3d, sampling_rate=16000)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid data shape" in error_msg
        assert "(2, 3, 4)" in error_msg
        assert "3D" in error_msg
        # Check WHY
        assert "Expected: 1D" in error_msg
        assert "2D" in error_msg
        # Check HOW
        assert "reshape" in error_msg.lower()
        assert "Example:" in error_msg

    @pytest.mark.parametrize("sampling_rate", [-44_100, 0])
    def test_nonpositive_sampling_rate_error_message(self, sampling_rate: int) -> None:
        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data=self.dask_data, sampling_rate=sampling_rate)

        error_message = str(exc_info.value)
        assert "Invalid sampling_rate" in error_message
        assert f"Got: {sampling_rate} Hz" in error_message
        assert "Expected: Positive value > 0" in error_message
        assert "Sampling rate represents samples per second and must be positive." in error_message
        assert "Common values: 8000, 16000, 22050, 44100, 48000 Hz" in error_message

    def test_file_not_found_error_message(self) -> None:
        """Test that missing file provides helpful error message."""
        fake_path = "/nonexistent/path/to/audio.wav"

        with pytest.raises(FileNotFoundError) as exc_info:
            ChannelFrame.from_file(fake_path)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Audio file not found" in error_msg
        # Cross-platform: check for key path components instead of exact path
        assert "nonexistent" in error_msg and "audio.wav" in error_msg
        # Check WHY (context)
        assert "Current directory:" in error_msg
        # Check HOW
        assert "check" in error_msg.lower()
        assert "File path is correct" in error_msg
        assert "File exists" in error_msg


class TestFadeIntegration:
    """Integration tests for fade functionality with other operations."""

    def setup_method(self) -> None:
        """Set up test fixtures for fade integration tests."""
        self.sample_rate = _SAMPLE_RATE
        self.data = _SINE_CH0.reshape(1, -1)  # Single channel, 440Hz sine
        self.dask_data = _da_from_array(self.data, chunks=(1, 4000))
        self.channel_frame = ChannelFrame(data=self.dask_data, sampling_rate=self.sample_rate, label="test_sine")

    def test_fade_preserves_rms_calculation(self) -> None:
        """Test that fade operation allows RMS calculation to work correctly."""
        # Apply fade
        faded = self.channel_frame.fade(fade_ms=100.0)

        # RMS should be calculable without errors
        rms_values = faded.rms
        assert len(rms_values) == 1
        assert rms_values[0] > 0  # RMS should be positive

        # Faded signal should have lower RMS than original (due to fade-in/out)
        original_rms = self.channel_frame.rms
        assert rms_values[0] < original_rms[0]

    def test_fade_with_normalize_chain(self) -> None:
        """Test fade operation chained with normalize."""
        # Chain fade and normalize operations
        processed = self.channel_frame.fade(fade_ms=50.0).normalize()

        # Should complete without errors
        assert isinstance(processed, ChannelFrame)
        assert processed.sampling_rate == self.sample_rate
        assert processed.n_channels == 1
        assert processed.n_samples == self.channel_frame.n_samples

        # Check the lineage-derived operation history view
        assert len(processed.operation_history) == 2
        assert processed.operation_history[0]["operation"] == "wandas.audio.fade"
        assert processed.operation_history[1]["operation"] == "wandas.audio.normalize"

        # Normalized signal should have max amplitude of 1.0
        max_amplitude = np.max(np.abs(processed.data))
        np.testing.assert_almost_equal(max_amplitude, 1.0, decimal=6)

    def test_fade_with_filter_chain(self) -> None:
        """Test fade operation chained with filtering."""
        # Chain fade and low-pass filter
        processed = self.channel_frame.fade(fade_ms=50.0).low_pass_filter(cutoff=1000)

        # Should complete without errors
        assert isinstance(processed, ChannelFrame)
        assert processed.sampling_rate == self.sample_rate

        # Check the operation history view
        assert len(processed.operation_history) == 2
        assert processed.operation_history[0]["operation"] == "wandas.audio.fade"
        assert processed.operation_history[1]["operation"] == "wandas.audio.lowpass_filter"

    def test_fade_with_multiple_operations_chain(self) -> None:
        """Test fade in a complex operation chain."""

        # Create a more complex processing chain
        processed = (
            self.channel_frame.fade(fade_ms=25.0).normalize().low_pass_filter(cutoff=2000).high_pass_filter(cutoff=100)
        )

        # Should complete without errors
        assert isinstance(processed, ChannelFrame)
        assert processed.sampling_rate == self.sample_rate

        # Check that all operations are recorded
        assert len(processed.operation_history) == 4
        operations = [op["operation"] for op in processed.operation_history]
        assert operations == [
            "wandas.audio.fade",
            "wandas.audio.normalize",
            "wandas.audio.lowpass_filter",
            "wandas.audio.highpass_filter",
        ]

    def test_fade_with_channel_operations(self) -> None:
        """Test fade with channel selection and operations."""
        # Create multi-channel signal
        multi_data = np.vstack([self.data[0], self.data[0] * 0.5])  # 2 channels
        multi_dask = _da_from_array(multi_data, chunks=(1, 4000))
        multi_frame = ChannelFrame(data=multi_dask, sampling_rate=self.sample_rate, label="multi_test")

        # Apply fade to all channels, then select one channel
        processed = multi_frame.fade(fade_ms=50.0).get_channel(0)

        # Should work correctly
        assert isinstance(processed, ChannelFrame)
        assert processed.n_channels == 1
        assert [record["operation"] for record in processed.operation_history] == [
            "wandas.audio.fade",
            "wandas.frame.get_channel",
        ]

    def test_fade_with_arithmetic_operations(self) -> None:
        """Test fade with arithmetic operations."""
        # Apply fade, then add a constant
        processed = self.channel_frame.fade(fade_ms=50.0) + 0.1

        # Should complete without errors
        assert isinstance(processed, ChannelFrame)
        assert processed.operation_history[0]["operation"] == "wandas.audio.fade"

        # Check that arithmetic operation is recorded
        assert processed.operation_history[1]["operation"] == "wandas.operator.add"

    def test_fade_preserves_metadata_and_labels(self) -> None:
        """Test that fade preserves channel metadata and labels."""
        # Set custom labels and metadata
        self.channel_frame.channels[0].label = "test_channel"
        self.channel_frame.channels[0]["gain"] = 0.8
        self.channel_frame.metadata["test_key"] = "test_value"

        # Apply fade
        faded = self.channel_frame.fade(fade_ms=50.0)

        # Check that label is updated to reflect the operation
        assert faded.channels[0].label == "fade(test_channel)"
        # Check that other metadata is preserved
        assert faded.channels[0]["gain"] == 0.8
        assert faded.metadata["test_key"] == "test_value"

        # Check the operation history view
        assert faded.operation_history[0]["operation"] == "wandas.audio.fade"
        assert faded.operation_history[0]["params"]["fade_ms"] == 50.0

    def test_fade_with_file_io_roundtrip(self) -> None:
        """Test fade operation with file save/load roundtrip."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            # Apply fade and save
            faded = self.channel_frame.fade(fade_ms=50.0)
            faded.to_wav(temp_filename)

            # Load back and verify
            loaded = ChannelFrame.from_file(temp_filename)

            # Should be able to load and have same basic properties
            assert loaded.sampling_rate == self.sample_rate
            assert loaded.n_channels == 1
            assert loaded.n_samples == self.channel_frame.n_samples

            # Data should be different (faded) but same shape
            assert loaded.data.shape == faded.data.shape

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def test_fade_with_different_fade_durations(self) -> None:
        """Test fade with different fade durations and verify effects."""
        # Test with very short fade
        short_fade = self.channel_frame.fade(fade_ms=1.0)
        short_rms = short_fade.rms[0]

        # Test with longer fade
        long_fade = self.channel_frame.fade(fade_ms=100.0)
        long_rms = long_fade.rms[0]

        # Longer fade should result in lower RMS (more signal attenuated)
        assert long_rms < short_rms

        # Both should be less than original
        original_rms = self.channel_frame.rms[0]
        assert short_rms < original_rms
        assert long_rms < original_rms

    def test_fade_with_multi_channel_signal(self) -> None:
        """Test fade with multi-channel signal."""
        # Create 3-channel signal
        multi_data = np.vstack(
            [
                self.data[0],  # Original
                self.data[0] * 0.7,  # Scaled down
                self.data[0] * 1.3,  # Scaled up
            ]
        )
        multi_dask = _da_from_array(multi_data, chunks=(1, 4000))
        multi_frame = ChannelFrame(data=multi_dask, sampling_rate=self.sample_rate, label="multi_test")

        # Apply fade
        faded = multi_frame.fade(fade_ms=50.0)

        # Should work for all channels
        assert faded.n_channels == 3
        assert faded.operation_history[0]["operation"] == "wandas.audio.fade"

        # Each channel should have different RMS due to different amplitudes
        rms_values = faded.rms
        assert len(rms_values) == 3
        assert rms_values[0] > rms_values[1]  # Original > scaled down
        assert rms_values[2] > rms_values[0]  # Scaled up > original

    def test_fade_lazy_evaluation_preserved(self) -> None:
        """Test that fade preserves lazy evaluation."""
        # Apply fade without computing
        faded = self.channel_frame.fade(fade_ms=50.0)

        # Should still be lazy (dask array)
        assert isinstance(faded._data, DaArray)

        # Operation history should be updated without computation
        assert len(faded.operation_history) == 1
        assert faded.operation_history[0]["operation"] == "wandas.audio.fade"

        # Only when we access .data should computation happen
        with mock.patch.object(DaArray, "compute", return_value=self.data) as mock_compute:
            _ = faded.data
            mock_compute.assert_called_once()

    def test_fade_with_visualization(self) -> None:
        """Test that faded signal can be visualized."""
        # Apply fade
        faded = self.channel_frame.fade(fade_ms=50.0)

        # Should be able to create plots without errors
        with (
            mock.patch("matplotlib.pyplot.figure"),
            mock.patch("matplotlib.pyplot.subplot"),
            mock.patch("matplotlib.axes.Axes.plot"),
            mock.patch("matplotlib.pyplot.tight_layout"),
            mock.patch("matplotlib.pyplot.show"),
        ):
            # Basic plot should work
            faded.plot()

            # RMS plot should work
            faded.rms_plot()

    def test_fade_error_handling_integration(self) -> None:
        """Test error handling in fade operation within processing chains."""
        # Test with invalid fade_ms (too long)
        # Create very short signal
        short_data = self.data[:, :100]  # Only 100 samples
        short_dask = _da_from_array(short_data, chunks=(1, 50))
        short_frame = ChannelFrame(data=short_dask, sampling_rate=self.sample_rate, label="short")

        # Apply fade with duration longer than signal
        # Should not fail immediately due to lazy eval
        faded_short = short_frame.fade(fade_ms=10.0)

        # Error should occur when we try to compute the result
        with pytest.raises(ValueError, match="Fade length too long"):
            _ = faded_short.data

        # Test with negative fade_ms - fails during operation creation
        with pytest.raises(ValueError, match="fade_ms must be non-negative"):
            self.channel_frame.fade(fade_ms=-1.0)


class TestChannelFrameValidation:
    """Test validation error paths in ChannelFrame."""

    def test_channel_labels_count_mismatch(self) -> None:
        """Test error when channel label count doesn't match channels."""
        # Create multi-channel data
        data = np.random.default_rng(42).random((2, 1000))

        # Try to create frame with wrong number of labels
        # This should trigger line 668
        with pytest.raises(ValueError, match="Number of channel labels does not match"):
            ChannelFrame.from_numpy(
                data,
                sampling_rate=16000,
                ch_labels=["Ch1", "Ch2", "Ch3"],  # 3 labels for 2 channels
            )

    def test_channel_units_count_mismatch(self) -> None:
        """Test error when channel unit count doesn't match channels."""
        # Create multi-channel data
        data = np.random.default_rng(42).random((2, 1000))

        # Try to create frame with wrong number of units
        # This should trigger line 678
        with pytest.raises(ValueError, match="Number of channel units does not match"):
            ChannelFrame.from_numpy(
                data,
                sampling_rate=16000,
                ch_units=["Pa"],  # 1 unit for 2 channels
            )


class TestChannelFrameBinaryFallback:
    """Fallback behavior for custom binary operands."""

    def test_add_fallback_to_type_name(self) -> None:
        """An unrecognized operand uses its type name without changing data."""
        signal = wd.generate_sin(freqs=[440], duration=1.0, sampling_rate=16000)

        # Create a custom class with __radd__ that can handle the operation
        class CustomNumeric:
            def __init__(self, value):
                self.value = value

            def __radd__(self, other):
                # Return the other object unchanged (just for testing)
                return other

        custom_obj = CustomNumeric(0.5)

        result = signal + custom_obj
        assert isinstance(result, ChannelFrame)
        assert isinstance(result._data, DaArray)
        assert result.label == f"({signal.label} + CustomNumeric)"
        np.testing.assert_array_equal(result.compute(), signal.compute())


class TestChannelFrameCrestFactor:
    """Focused 4-pillar tests for ChannelFrame.crest_factor property."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = _SAMPLE_RATE
        self.stereo_data: NDArrayReal = _STEREO_SINE
        self.cf_stereo = ChannelFrame.from_numpy(self.stereo_data, sampling_rate=self.sample_rate)

    # ------------------------------------------------------------------
    # Pillar 1 – Immutability: computing crest_factor must not modify the frame
    # ------------------------------------------------------------------

    def test_crest_factor_does_not_mutate_data(self) -> None:
        """Computing crest_factor must leave the underlying frame data unchanged."""
        original = self.stereo_data.copy()
        _ = self.cf_stereo.crest_factor
        np.testing.assert_array_equal(self.cf_stereo.compute(), original)

    def test_crest_factor_does_not_mutate_operation_history(self) -> None:
        """Computing crest_factor must not append to operation_history."""
        history_before = list(self.cf_stereo.operation_history)
        _ = self.cf_stereo.crest_factor
        assert self.cf_stereo.operation_history == history_before

    # ------------------------------------------------------------------
    # Pillar 2 – Metadata sync: return type and shape
    # ------------------------------------------------------------------

    def test_crest_factor_returns_numpy_array(self) -> None:
        """crest_factor must return a concrete numpy ndarray (not a dask array)."""
        result = self.cf_stereo.crest_factor
        assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"

    def test_crest_factor_shape_stereo(self) -> None:
        """crest_factor of a 2-channel frame must have shape (2,)."""
        result = self.cf_stereo.crest_factor
        assert result.shape == (2,), f"Expected (2,), got {result.shape}"

    def test_crest_factor_shape_mono(self) -> None:
        """crest_factor of a 1-channel frame must have shape (1,)."""
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        mono_data: NDArrayReal = np.sin(2 * np.pi * 440 * t).reshape(1, -1).astype(np.float64)
        cf_mono = ChannelFrame.from_numpy(mono_data, sampling_rate=self.sample_rate)
        result = cf_mono.crest_factor
        assert result.shape == (1,), f"Expected (1,), got {result.shape}"

    # ------------------------------------------------------------------
    # Pillar 3 – Mathematical consistency: known reference values
    # ------------------------------------------------------------------

    def test_crest_factor_sine_wave_equals_sqrt2(self) -> None:
        """Crest factor of A*sin(2πft) must equal sqrt(2) for any amplitude A."""
        result = self.cf_stereo.crest_factor
        expected = np.array([np.sqrt(2), np.sqrt(2)])
        # 440 Hz at 16 kHz SR produces 440 integer cycles; rtol=1e-6 for standard float64 precision.
        np.testing.assert_allclose(result, expected, rtol=1e-6)  # Pure sine RMS = amplitude / sqrt(2)

    def test_crest_factor_dc_signal_equals_one(self) -> None:
        """Crest factor of a constant (DC) signal must be 1.0."""
        dc_data: NDArrayReal = np.array([[3.0] * 1000, [-5.0] * 1000])
        cf_dc = ChannelFrame.from_numpy(dc_data, sampling_rate=self.sample_rate)
        result = cf_dc.crest_factor
        # Constant signal: peak == |amplitude| == RMS, so ratio == 1.0 exactly.
        np.testing.assert_allclose(result, [1.0, 1.0], rtol=1e-10)  # Exact: crest factor of constant signal = 1.0

    def test_crest_factor_zero_signal_is_one(self) -> None:
        """Crest factor of an all-zero signal must be 1.0 (no division by zero)."""
        import warnings

        zero_data: NDArrayReal = np.zeros((2, 1000))
        cf_zero = ChannelFrame.from_numpy(zero_data, sampling_rate=self.sample_rate)
        # Computing crest_factor on a zero-RMS channel must not emit RuntimeWarnings.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = cf_zero.crest_factor
        # Both channels are all-zeros → RMS == 0 → crest_factor returns 1.0.
        assert result.shape == (2,)
        np.testing.assert_array_equal(result, [1.0, 1.0])

    def test_crest_factor_known_values(self) -> None:
        """Crest factor matches explicit numpy reference calculation."""
        rng = np.random.default_rng(0)
        data: NDArrayReal = rng.standard_normal((3, 8000)).astype(np.float64)
        cf = ChannelFrame.from_numpy(data, sampling_rate=self.sample_rate)

        result = cf.crest_factor
        peak = np.max(np.abs(data), axis=1)
        rms_ref = np.sqrt(np.mean(data**2, axis=1))
        reference = peak / rms_ref
        # Same algorithm and dtype; results should be bit-identical (rtol=1e-10 guards rounding).
        np.testing.assert_allclose(result, reference, rtol=1e-10)  # Reference: direct np.sqrt(mean(x**2))

    def test_crest_factor_integer_input_matches_float64_reference(self) -> None:
        """crest_factor must cast integer input to float64 before abs/squaring."""
        int_data: NDArrayReal = np.array([[-32768, 0], [30000, -30000]], dtype=np.int16)
        cf_int = ChannelFrame.from_numpy(int_data, sampling_rate=self.sample_rate)

        result = cf_int.crest_factor
        float_data = int_data.astype(np.float64)
        peak = np.max(np.abs(float_data), axis=1)
        rms_ref = np.sqrt(np.mean(float_data**2, axis=1))
        reference = peak / rms_ref

        np.testing.assert_allclose(result, reference, rtol=1e-12)  # Reference: direct np.sqrt(mean(x**2))

    # ------------------------------------------------------------------
    # Pillar 4 – Reference-based verification: boolean channel selection
    # ------------------------------------------------------------------

    def test_crest_factor_boolean_indexing_filters_channels(self) -> None:
        """Boolean mask from crest_factor must correctly select channels."""
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        # Two channels: sine (CF≈√2≈1.414) and DC 1.0 (CF=1.0)
        sine: NDArrayReal = np.sin(2 * np.pi * 440 * t).astype(np.float64)
        dc_one: NDArrayReal = np.ones(self.sample_rate, dtype=np.float64)
        data: NDArrayReal = np.array([sine, dc_one])
        cf = ChannelFrame.from_numpy(data, sampling_rate=self.sample_rate)

        mask = cf.crest_factor > 1.1
        # Only the sine channel should have CF > 1.1
        assert mask.shape == (2,)
        assert bool(mask[0]) is True
        assert bool(mask[1]) is False

        filtered = cf[mask]
        assert filtered.n_channels == 1

    def test_crest_factor_n_channel_consistency(self) -> None:
        """len(cf.crest_factor) must equal cf.n_channels for any frame."""
        for n_channels in [1, 3, 5]:
            data: NDArrayReal = np.ones((n_channels, 500))
            cf = ChannelFrame.from_numpy(data, sampling_rate=self.sample_rate)
            assert len(cf.crest_factor) == cf.n_channels
