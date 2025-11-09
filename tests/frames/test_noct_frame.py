from typing import Any, Optional
from unittest import mock

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from dask.array.core import Array as DaArray

from wandas.core.metadata import ChannelMetadata
from wandas.frames.noct import NOctFrame
from wandas.utils.types import NDArrayReal

# Reference to dask array functions
_da_from_array = da.from_array  # type: ignore [unused-ignore]


# Helper function to create test data
def create_real_data(shape: tuple[int, ...]) -> NDArrayReal:
    """Create real test data with the given shape."""
    return np.random.rand(*shape).astype(np.float32)


def create_dask_array(data: NDArrayReal, chunks: Optional[tuple[int, ...]]) -> DaArray:
    """Convert NumPy array to Dask array with specified chunks."""
    return _da_from_array(data, chunks=chunks)


class TestNOctFrame:
    """Tests for the NOctFrame class"""

    def setup_method(self) -> None:
        """Set up test fixtures for each test"""
        self.sampling_rate: int = 44100
        self.n: int = 3  # 1/3 octave
        self.G: int = 10
        self.fr: int = 1000
        self.fmin: float = 20.0
        self.fmax: float = 20000.0

        # NOctFrameデータ形状: (channels, frequency_bins)
        # 適当な周波数ビン数を定義（_center_freqの結果に依存するので、実際は計算が必要）
        self.n_freq_bins: int = 30  # 仮の値
        self.shape: tuple[int, int] = (2, self.n_freq_bins)
        self.real_data: NDArrayReal = create_real_data(self.shape)
        # 遅延実行に対応したデータ構造の使用
        self.data: DaArray = _da_from_array(self.real_data, chunks=-1)

        # Create channel metadata
        self.channel_metadata: list[ChannelMetadata] = [
            ChannelMetadata(label="ch1", ref=1.0),
            ChannelMetadata(label="ch2", ref=1.0),
        ]

        # Create NOctFrame instance
        self.frame: NOctFrame = NOctFrame(
            data=self.data,
            sampling_rate=self.sampling_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
            label="test_frame",
            metadata={"test": "metadata"},
            channel_metadata=self.channel_metadata,
        )

    def test_initialization(self) -> None:
        """Test initialization with different parameters"""
        # Test with minimal required parameters
        minimal_frame: NOctFrame = NOctFrame(
            data=self.data,
            sampling_rate=self.sampling_rate,
        )
        assert minimal_frame.sampling_rate == self.sampling_rate
        assert minimal_frame.n == 3  # Default value
        assert minimal_frame.G == 10  # Default value
        assert minimal_frame.fr == 1000  # Default value
        assert minimal_frame.fmin == 0  # Default value
        assert minimal_frame.fmax == 0  # Default value

        # Test with all parameters
        assert self.frame.sampling_rate == self.sampling_rate
        assert self.frame.n == self.n
        assert self.frame.G == self.G
        assert self.frame.fr == self.fr
        assert self.frame.fmin == self.fmin
        assert self.frame.fmax == self.fmax
        assert self.frame.label == "test_frame"
        assert self.frame.metadata == {"test": "metadata"}

    def test_reshape_1d_data(self) -> None:
        """Test that 1D data is reshaped to 2D"""
        # Create 1D real data
        shape_1d: tuple[int] = (self.n_freq_bins,)
        real_data_1d: NDArrayReal = create_real_data(shape_1d)
        data_1d: DaArray = _da_from_array(real_data_1d, chunks=-1)

        frame_1d: NOctFrame = NOctFrame(
            data=data_1d,
            sampling_rate=self.sampling_rate,
        )
        assert frame_1d.n_channels == 1
        assert frame_1d.shape == (self.n_freq_bins,)

    # def test_reject_high_dim_data(self) -> None:
    #     """Test that >2D data raises ValueError"""
    #     # Create 3D real data
    #     shape_3d: tuple[int, int, int] = (2, 3, self.n_freq_bins)
    #     real_data_3d: NDArrayReal = create_real_data(shape_3d)
    #     data_3d: DaArray = _da_from_array(real_data_3d, chunks=-1)

    #     # Check that creating frame with 3D data raises ValueError
    #     with pytest.raises(ValueError):
    #         NOctFrame(
    #             data=data_3d,
    #             sampling_rate=self.sampling_rate,
    #         )

    def test_property_db(self) -> None:
        """Test dB property"""
        # dBプロパティのモックテスト
        # 実際の実装をテスト
        db: NDArrayReal = self.frame.dB
        ref_values: NDArrayReal = np.array([ch.ref for ch in self.channel_metadata])
        expected: NDArrayReal = 20 * np.log10(
            np.maximum(self.real_data / ref_values[:, np.newaxis], 1e-12)
        )
        np.testing.assert_allclose(db, expected)

    def test_property_dba(self) -> None:
        """Test dBA property"""
        with mock.patch("librosa.A_weighting") as mock_a_weighting:
            # モックの周波数重み付け係数を設定
            mock_weights: NDArrayReal = np.ones_like(self.frame.freqs)
            mock_a_weighting.return_value = mock_weights

            # dBAプロパティを取得
            dba: NDArrayReal = self.frame.dBA

            # librosのA_weightingが期待される引数で呼び出されたことを確認
            mock_a_weighting.assert_called_once()
            np.testing.assert_array_equal(
                mock_a_weighting.call_args[1]["frequencies"], self.frame.freqs
            )

            # 期待される結果（dB + A重み付け）と比較
            expected: NDArrayReal = self.frame.dB + mock_weights
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
                fmax=self.fmax,
                fmin=self.fmin,
                n=self.n,
                G=self.G,
                fr=self.fr,
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
        """Test that _binary_op raises NotImplementedError"""
        with pytest.raises(
            NotImplementedError, match="Operation \\+ is not implemented for NOctFrame"
        ):

            def add_op(a: Any, b: Any) -> Any:
                return a + b

            self.frame._binary_op(2.0, add_op, "+")

    def test_apply_operation_impl_not_implemented(self) -> None:
        """Test that _apply_operation_impl raises NotImplementedError"""
        with pytest.raises(
            NotImplementedError,
            match="Operation test_op is not implemented for NOctFrame",
        ):
            self.frame._apply_operation_impl("test_op", param1="value1")

    def test_plot(self) -> None:
        """Test plot method"""
        with mock.patch(
            "wandas.visualization.plotting.create_operation"
        ) as mock_create_op:
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
            "n": self.n,
            "G": self.G,
            "fr": self.fr,
            "fmin": self.fmin,
            "fmax": self.fmax,
        }

    def test_to_dataframe(self) -> None:
        """Test to_dataframe converts frame data to DataFrame with frequency index."""
        # NOctFrameの作成
        noct_frame = NOctFrame(
            data=self.data,
            sampling_rate=self.sampling_rate,
            n=self.n,
            G=self.G,
            fr=self.fr,
            fmin=self.fmin,
            fmax=self.fmax,
            channel_metadata=self.channel_metadata,
        )

        # DataFrame変換
        df = noct_frame.to_dataframe()

        # DataFrameの検証
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (self.shape[1], self.shape[0])  # (freq_bins, channels)
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
            sampling_rate=self.sampling_rate,
            n=self.n,
            G=self.G,
            fr=self.fr,
            fmin=self.fmin,
            fmax=self.fmax,
            channel_metadata=single_channel_metadata,
        )

        df = noct_frame.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (self.shape[1], 1)  # (freq_bins, 1)
        assert df.index.name == "frequency"
        assert list(df.columns) == ["ch1"]
