from typing import Any
from unittest import mock

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from dask.array.core import Array as DaArray

from wandas.core.metadata import ChannelMetadata
from wandas.frames.noct import NOctFrame, _center_freq
from wandas.processing.spectral import noct_band_labels
from wandas.utils.types import NDArrayReal

# Reference to dask array functions
_da_from_array = da.from_array

# --- Module-level deterministic NOct constants ---
_SAMPLING_RATE: int = 44100
_N: int = 3  # 1/3 octave
_G: int = 10
_FR: int = 1000
_FMIN: float = 20.0
_FMAX: float = 20000.0
# 20 Hz lies below MoSQITo's nominal table, so the frame has one more band than f_nom.
_CENTER_FREQS = noct_band_labels(*_center_freq(fmin=_FMIN, fmax=_FMAX, n=_N, G=_G, fr=_FR))
_N_FREQ_BINS: int = len(_CENTER_FREQS)
_SHAPE: tuple[int, int] = (2, _N_FREQ_BINS)


def create_real_data(shape: tuple[int, ...]) -> NDArrayReal:
    """Create real test data with the given shape."""
    return np.random.default_rng(42).random(shape).astype(np.float32)


class TestNOctFrame:
    """Tests for the NOctFrame class"""

    def setup_method(self) -> None:
        """Set up test fixtures for each test"""
        self.real_data: NDArrayReal = create_real_data(_SHAPE)
        # 遅延実行に対応したデータ構造の使用
        self.data: DaArray = _da_from_array(self.real_data, chunks=(1, -1))

        # Create channel metadata
        self.channel_metadata: list[ChannelMetadata] = [
            ChannelMetadata(label="ch1", ref=1.0),
            ChannelMetadata(label="ch2", ref=1.0),
        ]

        # Create NOctFrame instance
        self.frame: NOctFrame = NOctFrame(
            data=self.data,
            sampling_rate=_SAMPLING_RATE,
            fmin=_FMIN,
            fmax=_FMAX,
            n=_N,
            G=_G,
            fr=_FR,
            label="test_frame",
            metadata={"test": "metadata"},
            channel_metadata=self.channel_metadata,
        )

    def test_constructor_retains_default_and_explicit_analysis_parameters(self) -> None:
        """Test initialization with different parameters"""
        # Test with minimal required parameters
        minimal_frame: NOctFrame = NOctFrame(
            data=self.data,
            sampling_rate=_SAMPLING_RATE,
        )
        assert minimal_frame.sampling_rate == _SAMPLING_RATE
        assert isinstance(minimal_frame._data, DaArray)  # Dask laziness preserved
        assert minimal_frame.n == 3  # Default value
        assert minimal_frame.G == 10  # Default value
        assert minimal_frame.fr == 1000  # Default value
        assert minimal_frame.fmin == 0  # Default value
        assert minimal_frame.fmax == 0  # Default value

        # Test with all parameters
        assert self.frame.sampling_rate == _SAMPLING_RATE
        assert self.frame.n == _N
        assert self.frame.G == _G
        assert self.frame.fr == _FR
        assert self.frame.fmin == _FMIN
        assert self.frame.fmax == _FMAX
        assert self.frame.label == "test_frame"
        assert self.frame.metadata == {"test": "metadata"}

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"fmin": np.nan}, "fmin must be finite and non-negative"),
            ({"fmin": -1.0}, "fmin must be finite and non-negative"),
            ({"fmax": np.inf}, "Invalid frequency bounds for NOctFrame"),
            ({"fmin": 20.0, "fmax": 0.0}, "Invalid frequency bounds for NOctFrame"),
            ({"G": 0}, "n, G, and fr must be positive"),
            ({"fr": -1}, "n, G, and fr must be positive"),
        ],
    )
    def test_constructor_rejects_unrepresentable_analysis_state(
        self,
        kwargs: dict[str, Any],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            NOctFrame(
                data=self.data,
                sampling_rate=_SAMPLING_RATE,
                **kwargs,
            )

    def test_reverse_scalar_binary_operation_remains_unsupported(self) -> None:
        """Reverse scalar operators should respect NOctFrame binary-op policy."""
        with pytest.raises(TypeError, match="unsupported operand type"):
            2.0 + self.frame
        with pytest.raises(TypeError):
            np.float64(2.0) + self.frame

    def test_single_channel_2d_input_has_1d_public_shape(self) -> None:
        """A one-channel 2D input has a one-dimensional public shape."""
        # Build the one-channel 2D input from 1D real values.
        shape_1d: tuple[int] = (_N_FREQ_BINS,)
        real_data_1d: NDArrayReal = create_real_data(shape_1d)
        data_1d: DaArray = _da_from_array(real_data_1d.reshape(1, -1), chunks=(1, -1))

        frame_1d: NOctFrame = NOctFrame(
            data=data_1d,
            sampling_rate=_SAMPLING_RATE,
        )
        assert frame_1d.n_channels == 1
        assert frame_1d.shape == (_N_FREQ_BINS,)

    # def test_reject_high_dim_data(self) -> None:
    #     """Test that >2D data raises ValueError"""
    #     # Create 3D real data
    #     shape_3d: tuple[int, int, int] = (2, 3, _N_FREQ_BINS)
    #     real_data_3d: NDArrayReal = create_real_data(shape_3d)
    #     data_3d: DaArray = _da_from_array(real_data_3d, chunks=-1)

    #     # Check that creating frame with 3D data raises ValueError
    #     with pytest.raises(ValueError):
    #         NOctFrame(
    #             data=data_3d,
    #             sampling_rate=_SAMPLING_RATE,
    #         )

    def test_property_db(self) -> None:
        """Test dB property"""
        # dBプロパティのモックテスト
        # 実際の実装をテスト
        db: NDArrayReal = self.frame.dB
        ref_values: NDArrayReal = np.array([ch.ref for ch in self.channel_metadata])
        expected: NDArrayReal = 20 * np.log10(np.maximum(self.real_data / ref_values[:, np.newaxis], 1e-12))
        # Direct numpy equivalent — default rtol=1e-7 (exact match expected)
        np.testing.assert_allclose(db, expected)

    def test_property_dba(self) -> None:
        """Test dBA property"""
        with mock.patch("wandas.frames.noct.a_weighting_db") as mock_a_weighting:
            # モックの周波数重み付け係数を設定
            mock_weights: NDArrayReal = np.ones_like(self.frame.freqs)
            mock_a_weighting.return_value = mock_weights

            # dBAプロパティを取得
            dba: NDArrayReal = self.frame.dBA

            # A-weighting helper is called with expected arguments
            mock_a_weighting.assert_called_once()
            np.testing.assert_array_equal(mock_a_weighting.call_args[1]["frequencies"], self.frame.freqs)

            # 期待される結果（dB + A重み付け）と比較
            expected: NDArrayReal = self.frame.dB + mock_weights
            # Direct numpy equivalent — default rtol=1e-7 (exact match expected)
            np.testing.assert_allclose(dba, expected)

    def test_property_n_channels(self) -> None:
        """Test _n_channels property"""
        assert self.frame._n_channels == 2

    def test_property_freqs(self) -> None:
        """Test freqs property"""
        with mock.patch("wandas.frames.noct._center_freq") as mock_center_freq:
            # モックの中心周波数を設定
            mock_center_bands = np.array([1, 2, 3])
            mock_center_freqs = np.array([100, 125, 160])
            mock_center_freq.return_value = (mock_center_bands, mock_center_freqs)

            # freqsプロパティを取得
            freqs: NDArrayReal = self.frame.freqs

            # _center_freqが期待される引数で呼び出されたことを確認
            mock_center_freq.assert_called_once_with(
                fmax=_FMAX,
                fmin=_FMIN,
                n=_N,
                G=_G,
                fr=_FR,
            )

            # 返される値が期待通りであることを確認
            np.testing.assert_array_equal(freqs, mock_center_freqs)

    def test_freqs_not_numpy_array(self) -> None:
        """Test freqs property when _center_freq doesn't return numpy array"""
        with mock.patch("wandas.frames.noct._center_freq") as mock_center_freq:
            # _center_freqが非Numpy配列を返すケース
            mock_center_freq.return_value = (None, "not_an_array")

            # ValueError が発生することを確認
            with pytest.raises(ValueError, match="freqs is not numpy array"):
                _ = self.frame.freqs

    def test_binary_op_not_implemented(self) -> None:
        """Test that public binary operations are unsupported."""
        with pytest.raises(NotImplementedError, match="Operation \\+ is not implemented for NOctFrame"):
            _ = self.frame + 2.0

    def test_plot_delegates_default_and_custom_arguments_to_strategy(self) -> None:
        """The plot strategy is mocked; this checks delegation and its returned axis."""
        with mock.patch("wandas.visualization.plotting.create_operation") as mock_create_op:
            mock_plot_strategy: Any = mock.MagicMock()
            mock_create_op.return_value = mock_plot_strategy
            mock_ax: Any = mock.MagicMock()
            mock_plot_strategy.plot.return_value = mock_ax

            # Test with default parameters
            result: Any = self.frame.plot()
            mock_create_op.assert_called_once_with("noct")
            # Check that plot was called with the frame and the new explicit parameters
            call_kwargs = mock_plot_strategy.plot.call_args[1]
            assert call_kwargs["ax"] is None
            assert call_kwargs["title"] is None
            assert call_kwargs["overlay"] is False
            assert call_kwargs["Aw"] is False
            assert result is mock_ax

            # Reset mocks and test with custom parameters
            mock_create_op.reset_mock()
            mock_plot_strategy.plot.reset_mock()

            custom_ax: Any = mock.MagicMock()
            kwargs: dict[str, Any] = {"param1": "value1", "param2": "value2"}
            result = self.frame.plot("custom_plot", ax=custom_ax, **kwargs)

            mock_create_op.assert_called_once_with("custom_plot")
            # Verify that custom parameters are passed through
            call_kwargs = mock_plot_strategy.plot.call_args[1]
            assert call_kwargs["ax"] is custom_ax
            assert call_kwargs["param1"] == "value1"
            assert call_kwargs["param2"] == "value2"
            assert result is mock_ax

    def test_get_additional_init_kwargs(self) -> None:
        """Test _get_additional_init_kwargs method"""
        additional_kwargs = self.frame._get_additional_init_kwargs()
        assert additional_kwargs == {
            "n": _N,
            "G": _G,
            "fr": _FR,
            "fmin": _FMIN,
            "fmax": _FMAX,
        }

    def test_to_dataframe(self) -> None:
        """Test to_dataframe converts frame data to DataFrame with frequency index."""
        # NOctFrameの作成
        noct_frame = NOctFrame(
            data=self.data,
            sampling_rate=_SAMPLING_RATE,
            n=_N,
            G=_G,
            fr=_FR,
            fmin=_FMIN,
            fmax=_FMAX,
            channel_metadata=self.channel_metadata,
        )

        # DataFrame変換
        df = noct_frame.to_dataframe()

        # DataFrameの検証
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (_SHAPE[1], _SHAPE[0])  # (freq_bins, channels)
        assert df.index.name == "frequency"
        assert list(df.columns) == ["ch1", "ch2"]

        # 周波数インデックスの検証
        expected_freqs = noct_frame.freqs
        np.testing.assert_array_almost_equal(df.index.values, expected_freqs)

    def test_to_dataframe_single_channel(self) -> None:
        """Test to_dataframe with single channel."""
        # 単一チャネルのデータ作成
        single_channel_data = self.data[0:1, :]  # 最初のチャネルのみ
        single_channel_metadata = [self.channel_metadata[0]]

        noct_frame = NOctFrame(
            data=single_channel_data,
            sampling_rate=_SAMPLING_RATE,
            n=_N,
            G=_G,
            fr=_FR,
            fmin=_FMIN,
            fmax=_FMAX,
            channel_metadata=single_channel_metadata,
        )

        df = noct_frame.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (_SHAPE[1], 1)  # (freq_bins, 1)
        assert df.index.name == "frequency"
        assert list(df.columns) == ["ch1"]

    def test_plot_with_all_options(self) -> None:
        """Test plot method with all optional parameters"""
        with mock.patch("wandas.visualization.plotting.create_operation") as mock_create_op:
            mock_plot_strategy: Any = mock.MagicMock()
            mock_create_op.return_value = mock_plot_strategy
            mock_ax: Any = mock.MagicMock()
            mock_plot_strategy.plot.return_value = mock_ax

            # Test with all optional parameters
            result = self.frame.plot(
                plot_type="noct",
                ax=mock_ax,
                title="Test Title",
                overlay=True,
                xlabel="Frequency",
                ylabel="Level",
                alpha=0.5,
                xlim=(20, 20000),
                ylim=(0, 100),
                Aw=True,
                color="red",
            )

            # Verify all parameters were passed
            call_kwargs = mock_plot_strategy.plot.call_args[1]
            assert call_kwargs["ax"] is mock_ax
            assert call_kwargs["title"] == "Test Title"
            assert call_kwargs["overlay"] is True
            assert call_kwargs["xlabel"] == "Frequency"
            assert call_kwargs["ylabel"] == "Level"
            assert call_kwargs["alpha"] == 0.5
            assert call_kwargs["xlim"] == (20, 20000)
            assert call_kwargs["ylim"] == (0, 100)
            assert call_kwargs["Aw"] is True
            assert call_kwargs["color"] == "red"
            assert result is mock_ax

    def test_get_dataframe_columns(self) -> None:
        """Test _get_dataframe_columns method"""
        columns = self.frame._get_dataframe_columns()
        assert columns == ["ch1", "ch2"]

    def test_get_dataframe_index(self) -> None:
        """Test _get_dataframe_index method"""
        index = self.frame._get_dataframe_index()
        assert isinstance(index, pd.Index)
        assert index.name == "frequency"
        np.testing.assert_array_equal(index.values, self.frame.freqs)


class TestNOctFrameBandSlicing:
    """A band slice labels the selected bands with their own center frequencies."""

    def setup_method(self) -> None:
        self.real_data: NDArrayReal = create_real_data(_SHAPE)
        self.frame = NOctFrame(
            data=_da_from_array(self.real_data, chunks=(1, -1)),
            sampling_rate=_SAMPLING_RATE,
            fmin=_FMIN,
            fmax=_FMAX,
            n=_N,
            G=_G,
            fr=_FR,
            channel_metadata=[ChannelMetadata(label="ch1", ref=1.0), ChannelMetadata(label="ch2", ref=1.0)],
        )

    @pytest.mark.parametrize(
        "band_slice",
        [slice(3, 6), slice(0, 1), slice(None, 4), slice(-3, None), slice(10, None)],
    )
    def test_band_slice_keeps_selected_labels_and_data(self, band_slice: slice) -> None:
        sliced = self.frame[:, band_slice]

        np.testing.assert_array_equal(sliced.freqs, _CENTER_FREQS[band_slice])
        np.testing.assert_array_equal(sliced.data, self.real_data[:, band_slice])
        assert sliced.fmin == _CENTER_FREQS[band_slice][0]
        assert sliced.fmax == _CENTER_FREQS[band_slice][-1]

    def test_band_slice_outside_nominal_table_keeps_exact_label(self) -> None:
        # The first band (about 19.95 Hz) is outside MoSQITo's nominal table.
        sliced = self.frame[:, 0:2]

        np.testing.assert_array_equal(sliced.freqs, _CENTER_FREQS[0:2])
        assert sliced.freqs[0] != 20.0

    def test_band_slice_a_weighting_follows_selected_bands(self) -> None:
        sliced = self.frame[:, 3:6]

        np.testing.assert_allclose(sliced.dBA, self.frame.dBA[:, 3:6])

    def test_band_slice_of_slice_composes(self) -> None:
        sliced = self.frame[:, 2:12][:, 3:6]

        np.testing.assert_array_equal(sliced.freqs, _CENTER_FREQS[5:8])

    def test_channel_and_band_slice(self) -> None:
        sliced = self.frame["ch2", 3:6]

        np.testing.assert_array_equal(sliced.freqs, _CENTER_FREQS[3:6])
        np.testing.assert_array_equal(sliced.data, self.real_data[1, 3:6])

    def test_channel_only_selection_keeps_bounds(self) -> None:
        selected = self.frame[0, :]

        assert (selected.fmin, selected.fmax) == (_FMIN, _FMAX)
        np.testing.assert_array_equal(selected.freqs, _CENTER_FREQS)

    def test_band_slice_round_trips_through_wdf(self, tmp_path: Any) -> None:
        import wandas as wd

        path = tmp_path / "bands.wdf"
        self.frame[:, 3:6].save(path)
        loaded = wd.load(path)

        assert isinstance(loaded, NOctFrame)
        np.testing.assert_array_equal(loaded.freqs, _CENTER_FREQS[3:6])

    def test_stepped_band_slice_raises(self) -> None:
        with pytest.raises(ValueError, match="Stepped band slices"):
            self.frame[:, ::2]

    def test_empty_band_slice_raises(self) -> None:
        with pytest.raises(ValueError, match="selects no band"):
            self.frame[:, 5:5]

    def test_band_slice_replays_through_recipe(self) -> None:
        import wandas as wd
        from wandas.pipeline import RecipePlan

        signal = wd.generate_sin(freqs=[1000], duration=1, sampling_rate=16000)
        sliced = signal.noct_spectrum(fmin=100, fmax=4000, n=3)[:, 3:6]
        plan = RecipePlan.from_dict(RecipePlan.from_frame(sliced, input_names=("signal",)).to_dict())

        replayed = plan.apply({"signal": wd.generate_sin(freqs=[500], duration=1, sampling_rate=16000)})

        assert isinstance(replayed, NOctFrame)
        np.testing.assert_array_equal(replayed.freqs, [200.0, 250.0, 315.0])
