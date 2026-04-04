"""Tests for RoughnessFrame class."""

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.roughness import RoughnessFrame

# --- Module-level deterministic roughness test data ---
_N_BARK: int = 47
_N_TIME: int = 10
_SAMPLING_RATE: float = 10.0  # 10 Hz (typical for overlap=0.5)
_OVERLAP: float = 0.5
_rng = np.random.default_rng(42)
_DATA_MONO = da.from_array(_rng.random((_N_BARK, _N_TIME)), chunks=(47, 5))
_DATA_STEREO = da.from_array(_rng.random((2, _N_BARK, _N_TIME)), chunks=(1, 47, 5))
_BARK_AXIS = np.linspace(0.5, 23.5, _N_BARK)


class TestRoughnessFrame:
    """Test RoughnessFrame class."""

    def test_initialization_mono(self) -> None:
        """Test initialization with mono data."""
        frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )

        assert frame.data.shape == (_N_BARK, _N_TIME)
        assert frame.sampling_rate == _SAMPLING_RATE
        assert frame.n_bark_bands == 47
        assert frame.n_time_points == _N_TIME
        assert frame.overlap == _OVERLAP
        assert len(frame.bark_axis) == 47

    def test_initialization_stereo(self) -> None:
        """Test initialization with stereo data."""
        frame = RoughnessFrame(
            data=_DATA_STEREO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )

        assert frame.data.shape == (2, _N_BARK, _N_TIME)
        assert frame._n_channels == 2

    def test_initialization_validates_dimensions(self) -> None:
        """Test that initialization validates data dimensions."""
        # 1D data should fail
        with pytest.raises(ValueError, match="Data must be 2D or 3D"):
            RoughnessFrame(
                data=da.from_array(np.random.default_rng(42).random(10), chunks=5),
                sampling_rate=_SAMPLING_RATE,
                bark_axis=_BARK_AXIS,
                overlap=_OVERLAP,
            )

        # Wrong number of Bark bands should fail
        wrong_bark_data = da.from_array(np.random.default_rng(42).random((30, _N_TIME)), chunks=(15, 5))
        with pytest.raises(ValueError, match="Expected 47 Bark bands"):
            RoughnessFrame(
                data=wrong_bark_data,
                sampling_rate=_SAMPLING_RATE,
                bark_axis=_BARK_AXIS,
                overlap=_OVERLAP,
            )

    def test_initialization_validates_bark_axis(self) -> None:
        """Test that initialization validates bark_axis length."""
        wrong_bark_axis = np.linspace(0.5, 23.5, 30)

        with pytest.raises(ValueError, match="bark_axis must have 47 elements"):
            RoughnessFrame(
                data=_DATA_MONO,
                sampling_rate=_SAMPLING_RATE,
                bark_axis=wrong_bark_axis,
                overlap=_OVERLAP,
            )

    def test_initialization_validates_overlap(self) -> None:
        """Test that initialization validates overlap parameter."""
        with pytest.raises(ValueError, match="overlap must be in"):
            RoughnessFrame(
                data=_DATA_MONO,
                sampling_rate=_SAMPLING_RATE,
                bark_axis=_BARK_AXIS,
                overlap=1.5,
            )

        with pytest.raises(ValueError, match="overlap must be in"):
            RoughnessFrame(
                data=_DATA_MONO,
                sampling_rate=_SAMPLING_RATE,
                bark_axis=_BARK_AXIS,
                overlap=-0.1,
            )

    def test_time_property(self) -> None:
        """Test time property calculation."""
        frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )

        time = frame.time
        assert len(time) == _N_TIME
        assert time[0] == 0.0
        assert time[-1] == pytest.approx((_N_TIME - 1) / _SAMPLING_RATE)

    def test_metadata_storage(self) -> None:
        """Test that overlap is stored in metadata."""
        frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )

        assert "overlap" in frame.metadata
        assert frame.metadata["overlap"] == _OVERLAP

    def test_default_label(self) -> None:
        """Test default label is set correctly."""
        frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )

        assert frame.label == "roughness_spec"

    def test_custom_label(self) -> None:
        """Test custom label can be set."""
        frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
            label="custom_roughness",
        )

        assert frame.label == "custom_roughness"

    def test_plot_mono(self) -> None:
        """Test plot method with mono data."""
        import matplotlib.pyplot as plt

        frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )

        ax = frame.plot()
        assert ax is not None
        assert ax.get_xlabel() == "Time [s]"
        assert ax.get_ylabel() == "Frequency [Bark]"

        plt.close("all")

    def test_plot_stereo(self) -> None:
        """Test plot method with stereo data (should plot mean)."""
        import matplotlib.pyplot as plt

        frame = RoughnessFrame(
            data=_DATA_STEREO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )

        # Should plot mean across channels without error
        ax = frame.plot()
        assert ax is not None

        plt.close("all")

    def test_plot_with_custom_parameters(self) -> None:
        """Test plot method with custom parameters."""
        import matplotlib.pyplot as plt

        frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )

        ax = frame.plot(
            title="Custom Title",
            cmap="hot",
            vmin=0.0,
            vmax=1.0,
            xlabel="Custom X",
            ylabel="Custom Y",
        )

        assert ax.get_title() == "Custom Title"
        assert ax.get_xlabel() == "Custom X"
        assert ax.get_ylabel() == "Custom Y"

        plt.close("all")

    def test_plot_default_title(self) -> None:
        """Test that plot generates appropriate default title."""
        import matplotlib.pyplot as plt

        frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=0.5,
        )

        ax = frame.plot()
        assert "overlap=0.5" in ax.get_title()

        plt.close("all")

    def test_bark_axis_range(self) -> None:
        """Test that bark_axis has correct range."""
        frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )

        assert frame.bark_axis[0] == pytest.approx(0.5, abs=0.1)
        assert frame.bark_axis[-1] == pytest.approx(23.5, abs=0.1)
        assert len(frame.bark_axis) == 47

    def test_n_channels_property(self) -> None:
        """Test _n_channels property."""
        frame_mono = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )
        assert frame_mono._n_channels == 1

        frame_stereo = RoughnessFrame(
            data=_DATA_STEREO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )
        assert frame_stereo._n_channels == 2

    def test_to_dataframe_raises_not_implemented_error(self) -> None:
        """Test to_dataframe raises NotImplementedError for 2D roughness data."""
        # RoughnessFrameの作成（モノラル）
        roughness_frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )

        # DataFrame変換がNotImplementedErrorを投げることを確認
        with pytest.raises(NotImplementedError, match="not supported"):
            roughness_frame.to_dataframe()

    def test_get_dataframe_index_not_implemented(self) -> None:
        """Test _get_dataframe_index raises NotImplementedError."""
        frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )
        with pytest.raises(NotImplementedError, match="DataFrame index is not supported"):
            frame._get_dataframe_index()

    def test_apply_operation_not_implemented(self) -> None:
        """Test _apply_operation_impl raises NotImplementedError."""
        frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )
        with pytest.raises(NotImplementedError, match="Operation .* is not supported"):
            frame._apply_operation_impl("some_operation", param=1.0)

    def test_binary_op_with_scalar(self) -> None:
        """Test binary operations with scalar values."""
        frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )

        # Test addition with scalar
        original_data = frame.data.copy()
        result = frame + 1.0
        assert result is not frame  # Pillar 1: immutability
        assert isinstance(result._data, DaArray)  # Pillar 1: Dask laziness
        assert isinstance(result, RoughnessFrame)
        assert result.sampling_rate == frame.sampling_rate
        assert result.overlap == frame.overlap
        assert np.allclose(result.data, original_data + 1.0)
        assert np.allclose(frame.data, original_data)  # Pillar 1: original unchanged

    def test_binary_op_with_roughness_frame(self) -> None:
        """Test binary operations between RoughnessFrame instances."""
        frame1 = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )
        frame2 = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )

        # Test addition
        original_data1 = frame1.data.copy()
        result = frame1 + frame2
        assert result is not frame1  # Pillar 1: immutability
        assert result is not frame2
        assert isinstance(result._data, DaArray)  # Pillar 1: Dask laziness
        assert isinstance(result, RoughnessFrame)
        assert np.allclose(result.data, original_data1 + frame2.data)
        assert np.allclose(frame1.data, original_data1)  # Pillar 1: original unchanged

    def test_binary_op_sampling_rate_mismatch(self) -> None:
        """Test binary operation raises error on sampling rate mismatch."""
        frame1 = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=10.0,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )
        frame2 = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=20.0,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )

        with pytest.raises(ValueError, match="Sampling rates do not match"):
            _ = frame1 + frame2

    def test_binary_op_shape_mismatch(self) -> None:
        """Test binary operation raises error on shape mismatch."""
        frame1 = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )
        # Create data with different time dimension
        different_time_data = da.from_array(np.random.default_rng(42).random((_N_BARK, 5)), chunks=(47, 5))
        frame2 = RoughnessFrame(
            data=different_time_data,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )

        with pytest.raises(ValueError, match="Shape mismatch"):
            _ = frame1 + frame2

    def test_binary_op_with_numpy_array(self) -> None:
        """Test binary operation with numpy array."""
        frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )
        # Add numpy array
        array = np.ones((_N_BARK, _N_TIME))
        original_data = frame.data.copy()
        result = frame + array
        assert result is not frame  # Pillar 1: immutability
        assert isinstance(result._data, DaArray)  # Pillar 1: Dask laziness
        assert isinstance(result, RoughnessFrame)
        assert np.allclose(result.data, original_data + array)
        assert np.allclose(frame.data, original_data)  # Pillar 1: original unchanged

    def test_get_additional_init_kwargs(self) -> None:
        """Test _get_additional_init_kwargs returns correct parameters."""
        frame = RoughnessFrame(
            data=_DATA_MONO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )
        kwargs = frame._get_additional_init_kwargs()
        assert "bark_axis" in kwargs
        assert "overlap" in kwargs
        assert np.array_equal(kwargs["bark_axis"], _BARK_AXIS)
        assert kwargs["overlap"] == _OVERLAP

    def test_get_dataframe_columns(self) -> None:
        """Test _get_dataframe_columns returns channel labels."""
        frame = RoughnessFrame(
            data=_DATA_STEREO,
            sampling_rate=_SAMPLING_RATE,
            bark_axis=_BARK_AXIS,
            overlap=_OVERLAP,
        )
        columns = frame._get_dataframe_columns()
        assert isinstance(columns, list)
        assert len(columns) == 2
        assert all(isinstance(col, str) for col in columns)
