"""Tests for RoughnessFrame class."""

import dask.array as da
import numpy as np
import pytest

from wandas.frames.roughness import RoughnessFrame


class TestRoughnessFrame:
    """Test RoughnessFrame class."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        # Create sample roughness data (47 Bark bands, 10 time points)
        self.n_bark = 47
        self.n_time = 10
        self.sampling_rate = 10.0  # 10 Hz (typical for overlap=0.5)
        self.overlap = 0.5

        # Create mock specific roughness data
        self.data_mono = da.from_array(
            np.random.random((self.n_bark, self.n_time)), chunks=(47, 5)
        )
        self.data_stereo = da.from_array(
            np.random.random((2, self.n_bark, self.n_time)), chunks=(1, 47, 5)
        )

        # Create bark axis (0.5 to 23.5 Bark)
        self.bark_axis = np.linspace(0.5, 23.5, self.n_bark)

    def test_initialization_mono(self) -> None:
        """Test initialization with mono data."""
        frame = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )

        assert frame.data.shape == (self.n_bark, self.n_time)
        assert frame.sampling_rate == self.sampling_rate
        assert frame.n_bark_bands == 47
        assert frame.n_time_points == self.n_time
        assert frame.overlap == self.overlap
        assert len(frame.bark_axis) == 47

    def test_initialization_stereo(self) -> None:
        """Test initialization with stereo data."""
        frame = RoughnessFrame(
            data=self.data_stereo,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )

        assert frame.data.shape == (2, self.n_bark, self.n_time)
        assert frame._n_channels == 2

    def test_initialization_validates_dimensions(self) -> None:
        """Test that initialization validates data dimensions."""
        # 1D data should fail
        with pytest.raises(ValueError, match="Data must be 2D or 3D"):
            RoughnessFrame(
                data=da.from_array(np.random.random(10), chunks=5),
                sampling_rate=self.sampling_rate,
                bark_axis=self.bark_axis,
                overlap=self.overlap,
            )

        # Wrong number of Bark bands should fail
        wrong_bark_data = da.from_array(
            np.random.random((30, self.n_time)), chunks=(15, 5)
        )
        with pytest.raises(ValueError, match="Expected 47 Bark bands"):
            RoughnessFrame(
                data=wrong_bark_data,
                sampling_rate=self.sampling_rate,
                bark_axis=self.bark_axis,
                overlap=self.overlap,
            )

    def test_initialization_validates_bark_axis(self) -> None:
        """Test that initialization validates bark_axis length."""
        wrong_bark_axis = np.linspace(0.5, 23.5, 30)

        with pytest.raises(ValueError, match="bark_axis must have 47 elements"):
            RoughnessFrame(
                data=self.data_mono,
                sampling_rate=self.sampling_rate,
                bark_axis=wrong_bark_axis,
                overlap=self.overlap,
            )

    def test_initialization_validates_overlap(self) -> None:
        """Test that initialization validates overlap parameter."""
        with pytest.raises(ValueError, match="overlap must be in"):
            RoughnessFrame(
                data=self.data_mono,
                sampling_rate=self.sampling_rate,
                bark_axis=self.bark_axis,
                overlap=1.5,
            )

        with pytest.raises(ValueError, match="overlap must be in"):
            RoughnessFrame(
                data=self.data_mono,
                sampling_rate=self.sampling_rate,
                bark_axis=self.bark_axis,
                overlap=-0.1,
            )

    def test_time_property(self) -> None:
        """Test time property calculation."""
        frame = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )

        time = frame.time
        assert len(time) == self.n_time
        assert time[0] == 0.0
        assert time[-1] == pytest.approx((self.n_time - 1) / self.sampling_rate)

    def test_metadata_storage(self) -> None:
        """Test that overlap is stored in metadata."""
        frame = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )

        assert "overlap" in frame.metadata
        assert frame.metadata["overlap"] == self.overlap

    def test_default_label(self) -> None:
        """Test default label is set correctly."""
        frame = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )

        assert frame.label == "roughness_spec"

    def test_custom_label(self) -> None:
        """Test custom label can be set."""
        frame = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
            label="custom_roughness",
        )

        assert frame.label == "custom_roughness"

    def test_plot_mono(self) -> None:
        """Test plot method with mono data."""
        import matplotlib.pyplot as plt

        frame = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
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
            data=self.data_stereo,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )

        # Should plot mean across channels without error
        ax = frame.plot()
        assert ax is not None

        plt.close("all")

    def test_plot_with_custom_parameters(self) -> None:
        """Test plot method with custom parameters."""
        import matplotlib.pyplot as plt

        frame = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
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
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=0.5,
        )

        ax = frame.plot()
        assert "overlap=0.5" in ax.get_title()

        plt.close("all")

    def test_bark_axis_range(self) -> None:
        """Test that bark_axis has correct range."""
        frame = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )

        assert frame.bark_axis[0] == pytest.approx(0.5, abs=0.1)
        assert frame.bark_axis[-1] == pytest.approx(23.5, abs=0.1)
        assert len(frame.bark_axis) == 47

    def test_n_channels_property(self) -> None:
        """Test _n_channels property."""
        frame_mono = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )
        assert frame_mono._n_channels == 1

        frame_stereo = RoughnessFrame(
            data=self.data_stereo,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )
        assert frame_stereo._n_channels == 2

    def test_to_dataframe_raises_not_implemented_error(self) -> None:
        """Test to_dataframe raises NotImplementedError for 2D roughness data."""
        # RoughnessFrameの作成（モノラル）
        roughness_frame = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )

        # DataFrame変換がNotImplementedErrorを投げることを確認
        with pytest.raises(NotImplementedError, match="not supported"):
            roughness_frame.to_dataframe()

    def test_get_dataframe_index_not_implemented(self) -> None:
        """Test _get_dataframe_index raises NotImplementedError."""
        frame = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )
        with pytest.raises(
            NotImplementedError, match="DataFrame index is not supported"
        ):
            frame._get_dataframe_index()

    def test_apply_operation_not_implemented(self) -> None:
        """Test _apply_operation_impl raises NotImplementedError."""
        frame = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )
        with pytest.raises(NotImplementedError, match="Operation .* is not supported"):
            frame._apply_operation_impl("some_operation", param=1.0)

    def test_binary_op_with_scalar(self) -> None:
        """Test binary operations with scalar values."""
        frame = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )

        # Test addition with scalar
        result = frame + 1.0
        assert isinstance(result, RoughnessFrame)
        assert result.sampling_rate == frame.sampling_rate
        assert result.overlap == frame.overlap
        assert np.allclose(result.data, frame.data + 1.0)

    def test_binary_op_with_roughness_frame(self) -> None:
        """Test binary operations between RoughnessFrame instances."""
        frame1 = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )
        frame2 = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )

        # Test addition
        result = frame1 + frame2
        assert isinstance(result, RoughnessFrame)
        assert np.allclose(result.data, frame1.data + frame2.data)

    def test_binary_op_sampling_rate_mismatch(self) -> None:
        """Test binary operation raises error on sampling rate mismatch."""
        frame1 = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=10.0,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )
        frame2 = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=20.0,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )

        with pytest.raises(ValueError, match="Sampling rates do not match"):
            _ = frame1 + frame2

    def test_binary_op_shape_mismatch(self) -> None:
        """Test binary operation raises error on shape mismatch."""
        frame1 = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )
        # Create data with different time dimension
        different_time_data = da.from_array(
            np.random.random((self.n_bark, 5)), chunks=(47, 5)
        )
        frame2 = RoughnessFrame(
            data=different_time_data,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )

        with pytest.raises(ValueError, match="Shape mismatch"):
            _ = frame1 + frame2

    def test_binary_op_with_numpy_array(self) -> None:
        """Test binary operation with numpy array."""
        frame = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )
        # Add numpy array
        array = np.ones((self.n_bark, self.n_time))
        result = frame + array
        assert isinstance(result, RoughnessFrame)
        assert np.allclose(result.data, frame.data + array)

    def test_get_additional_init_kwargs(self) -> None:
        """Test _get_additional_init_kwargs returns correct parameters."""
        frame = RoughnessFrame(
            data=self.data_mono,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )
        kwargs = frame._get_additional_init_kwargs()
        assert "bark_axis" in kwargs
        assert "overlap" in kwargs
        assert np.array_equal(kwargs["bark_axis"], self.bark_axis)
        assert kwargs["overlap"] == self.overlap

    def test_get_dataframe_columns(self) -> None:
        """Test _get_dataframe_columns returns channel labels."""
        frame = RoughnessFrame(
            data=self.data_stereo,
            sampling_rate=self.sampling_rate,
            bark_axis=self.bark_axis,
            overlap=self.overlap,
        )
        columns = frame._get_dataframe_columns()
        assert isinstance(columns, list)
        assert len(columns) == 2
        assert all(isinstance(col, str) for col in columns)
