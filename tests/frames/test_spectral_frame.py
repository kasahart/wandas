from typing import Any, Optional
from unittest import mock

# filepath: wandas/core/test_spectral_frame.py
import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.core.metadata import ChannelMetadata
from wandas.frames.spectral import SpectralFrame
from wandas.utils.types import NDArrayComplex, NDArrayReal

# Reference to dask array functions
_da_from_array = da.from_array  # type: ignore [unused-ignore]


# Helper function to create complex test data
def create_complex_data(shape: tuple[int, ...]) -> NDArrayComplex:
    """Create complex test data with the given shape."""
    real_part = np.random.rand(*shape)
    imag_part = np.random.rand(*shape)
    return real_part + 1j * imag_part


def create_dask_array(
    data: NDArrayComplex, chunks: Optional[tuple[int, ...]]
) -> DaArray:
    """Convert NumPy array to Dask array with specified chunks."""
    return _da_from_array(data, chunks=chunks)


class TestSpectralFrame:
    """Tests for the SpectralFrame class"""

    def setup_method(self) -> None:
        """Set up test fixtures for each test"""
        self.sampling_rate: int = 44100
        self.n_fft: int = 1024
        self.window: str = "hann"

        # Create complex test data for 2 channels
        self.shape: tuple[int, int] = (2, self.n_fft // 2 + 1)
        self.complex_data: NDArrayComplex = create_complex_data(self.shape)
        # 遅延実行に対応したデータ構造の使用
        self.data: DaArray = _da_from_array(self.complex_data, chunks=-1)

        # Create channel metadata
        self.channel_metadata: list[ChannelMetadata] = [
            ChannelMetadata(label="ch1", ref=1.0),
            ChannelMetadata(label="ch2", ref=1.0),
        ]

        # Create SpectralFrame instance
        self.frame: SpectralFrame = SpectralFrame(
            data=self.data,
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            window=self.window,
            label="test_frame",
            metadata={"test": "metadata"},
            channel_metadata=self.channel_metadata,
        )

    def test_initialization(self) -> None:
        """Test initialization with different parameters"""
        # Test with minimal required parameters
        minimal_frame: SpectralFrame = SpectralFrame(
            data=self.data, sampling_rate=self.sampling_rate, n_fft=self.n_fft
        )
        assert minimal_frame.sampling_rate == self.sampling_rate
        assert minimal_frame.n_fft == self.n_fft
        assert minimal_frame.window == "hann"  # Default value

        # Test with all parameters
        assert self.frame.sampling_rate == self.sampling_rate
        assert self.frame.n_fft == self.n_fft
        assert self.frame.window == self.window
        assert self.frame.label == "test_frame"
        assert self.frame.metadata == {"test": "metadata"}

    def test_reshape_1d_data(self) -> None:
        """Test that 1D data is reshaped to 2D"""
        # Create 1D complex data
        shape_1d: tuple[int] = (self.n_fft // 2 + 1,)
        complex_data_1d: NDArrayComplex = create_complex_data(shape_1d)
        data_1d: DaArray = _da_from_array(complex_data_1d, chunks=-1)

        # Create frame with 1D data
        frame_1d: SpectralFrame = SpectralFrame(
            data=data_1d, sampling_rate=self.sampling_rate, n_fft=self.n_fft
        )

        # Check that shape is (1, n_fft//2+1)
        assert frame_1d.shape == (1, self.n_fft // 2 + 1)

    def test_reject_high_dim_data(self) -> None:
        """Test that >2D data raises ValueError"""
        # Create 3D complex data
        shape_3d: tuple[int, int, int] = (2, 3, self.n_fft // 2 + 1)
        complex_data_3d: NDArrayComplex = create_complex_data(shape_3d)
        data_3d: DaArray = _da_from_array(complex_data_3d, chunks=-1)

        # Check that creating frame with 3D data raises ValueError
        with pytest.raises(ValueError):
            SpectralFrame(
                data=data_3d, sampling_rate=self.sampling_rate, n_fft=self.n_fft
            )

    def test_property_magnitude(self) -> None:
        """Test magnitude property"""
        magnitude: NDArrayReal = self.frame.magnitude
        # 結果を評価するために .compute() を呼び出している
        expected: NDArrayReal = np.abs(self.data.compute())
        np.testing.assert_allclose(magnitude, expected)

    def test_property_phase(self) -> None:
        """Test phase property"""
        phase: NDArrayReal = self.frame.phase
        expected: NDArrayReal = np.angle(self.data.compute())
        np.testing.assert_allclose(phase, expected)

    def test_property_power(self) -> None:
        """Test power property"""
        power: NDArrayReal = self.frame.power
        expected: NDArrayReal = np.abs(self.data.compute()) ** 2
        np.testing.assert_allclose(power, expected)

    def test_property_db(self) -> None:
        """Test dB property"""
        db: NDArrayReal = self.frame.dB
        mag: NDArrayReal = np.abs(self.data.compute())
        ref_values: NDArrayReal = np.array([ch.ref for ch in self.channel_metadata])
        expected: NDArrayReal = 20 * np.log10(
            np.maximum(mag / ref_values[:, np.newaxis], 1e-12)
        )
        np.testing.assert_allclose(db, expected)

    def test_property_dba(self) -> None:
        """Test dBA property"""
        with mock.patch("librosa.A_weighting") as mock_a_weighting:
            mock_weights: NDArrayReal = np.ones_like(self.frame.freqs)
            mock_a_weighting.return_value = mock_weights

            dba: NDArrayReal = self.frame.dBA

            mock_a_weighting.assert_called_once()
            np.testing.assert_array_equal(
                mock_a_weighting.call_args[1]["frequencies"], self.frame.freqs
            )

            expected: NDArrayReal = self.frame.dB + mock_weights
            np.testing.assert_allclose(dba, expected)

    def test_property_n_channels(self) -> None:
        """Test _n_channels property"""
        assert self.frame._n_channels == 2

    def test_property_freqs(self) -> None:
        """Test freqs property"""
        freqs: NDArrayReal = self.frame.freqs
        expected: NDArrayReal = np.fft.rfftfreq(self.n_fft, 1.0 / self.sampling_rate)
        np.testing.assert_allclose(freqs, expected)

    def test_binary_op_with_spectral_frame(self) -> None:
        """Test _binary_op with another SpectralFrame"""
        other_data: DaArray = _da_from_array(create_complex_data(self.shape), chunks=-1)
        other_frame: SpectralFrame = SpectralFrame(
            data=other_data,
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            window=self.window,
            label="other_frame",
            channel_metadata=self.channel_metadata,
        )

        # Test binary operation
        def add_op(a: Any, b: Any) -> Any:
            return a + b

        symbol: str = "+"
        result: SpectralFrame = self.frame._binary_op(other_frame, add_op, symbol)

        assert isinstance(result, SpectralFrame)
        assert result.sampling_rate == self.sampling_rate
        assert result.n_fft == self.n_fft
        assert result.window == self.window
        assert result.label == f"({self.frame.label} {symbol} {other_frame.label})"

        # 結果を評価するために .compute() を呼び出している
        expected_data: NDArrayComplex = add_op(self.data, other_data).compute()
        np.testing.assert_allclose(result.data, expected_data)

    def test_binary_op_with_scalar(self) -> None:
        """Test _binary_op with a scalar"""
        scalar: float = 2.0

        def multiply_op(a: Any, b: Any) -> Any:
            return a * b

        symbol: str = "*"
        result: SpectralFrame = self.frame._binary_op(scalar, multiply_op, symbol)

        assert isinstance(result, SpectralFrame)
        assert result.label == f"({self.frame.label} {symbol} {scalar})"

        expected_data: NDArrayComplex = multiply_op(self.data, scalar).compute()
        np.testing.assert_allclose(result.data, expected_data)

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
            mock_create_op.assert_called_once_with("frequency")
            mock_plot_strategy.plot.assert_called_once_with(self.frame, ax=None)
            assert result is mock_ax

            # Reset mocks and test with custom parameters
            mock_create_op.reset_mock()
            mock_plot_strategy.plot.reset_mock()

            custom_ax: Any = mock.MagicMock()
            kwargs: dict[str, Any] = {"param1": "value1", "param2": "value2"}
            result = self.frame.plot("custom_plot", ax=custom_ax, **kwargs)

            mock_create_op.assert_called_once_with("custom_plot")
            mock_plot_strategy.plot.assert_called_once_with(
                self.frame, ax=custom_ax, **kwargs
            )
            assert result is mock_ax

    def test_plot_matrix(self) -> None:
        """Test plot_matrix method"""
        with mock.patch(
            "wandas.visualization.plotting.create_operation"
        ) as mock_create_op:
            mock_plot_strategy: Any = mock.MagicMock()
            mock_create_op.return_value = mock_plot_strategy
            mock_ax: Any = mock.MagicMock()
            mock_plot_strategy.plot.return_value = mock_ax

            # テスト実行（デフォルトパラメータ）
            result: Any = self.frame.plot_matrix()

            # プロットストラテジーの作成を検証
            mock_create_op.assert_called_once_with("matrix")

            # プロットメソッドの呼び出しを検証
            mock_plot_strategy.plot.assert_called_once_with(self.frame)

            # 戻り値を検証
            assert result is mock_ax

            # モックをリセット
            mock_create_op.reset_mock()
            mock_plot_strategy.plot.reset_mock()

            # カスタムパラメータでテスト
            kwargs: dict[str, Any] = {
                "vmin": -10,
                "vmax": 10,
                "cmap": "viridis",
                "title": "Test Matrix Plot",
            }
            result = self.frame.plot_matrix(plot_type="custom_matrix", **kwargs)

            # カスタムプロットタイプでの呼び出しを検証
            mock_create_op.assert_called_once_with("custom_matrix")

            # カスタムパラメータの渡し方を検証
            mock_plot_strategy.plot.assert_called_once_with(self.frame, **kwargs)

            # 戻り値を検証
            assert result is mock_ax

    def test_ifft(self) -> None:
        """Test ifft method"""
        with (
            mock.patch("wandas.frames.channel.ChannelFrame") as mock_channel_frame,
            mock.patch(
                "wandas.processing.time_series.create_operation"
            ) as mock_create_op,
        ):
            mock_ifft_op: Any = mock.MagicMock()
            mock_create_op.return_value = mock_ifft_op
            mock_time_series: DaArray = mock.MagicMock(spec=DaArray)
            mock_ifft_op.process.return_value = mock_time_series
            mock_result: Any = mock.MagicMock()
            mock_channel_frame.return_value = mock_result

            result = self.frame.ifft()

            mock_create_op.assert_called_once_with(
                "ifft", self.sampling_rate, n_fft=self.n_fft, window=self.window
            )
            mock_ifft_op.process.assert_called_once_with(self.data)

            mock_channel_frame.assert_called_once_with(
                data=mock_time_series,
                sampling_rate=self.sampling_rate,
                label=f"ifft({self.frame.label})",
                metadata=self.frame.metadata,
                operation_history=self.frame.operation_history,
                channel_metadata=self.frame._channel_metadata,
            )

            assert result is mock_result

    def test_mismatch_sampling_rate_error(self) -> None:
        """Test that operations with mismatched sampling rates raise ValueError"""
        other_data: DaArray = _da_from_array(create_complex_data(self.shape), chunks=-1)
        other_frame: SpectralFrame = SpectralFrame(
            data=other_data,
            sampling_rate=22050,  # Different sampling rate
            n_fft=self.n_fft,
            window=self.window,
        )

        with pytest.raises(ValueError, match="サンプリングレートが一致していません"):

            def add_op(a: Any, b: Any) -> Any:
                return a + b

            self.frame._binary_op(other_frame, add_op, "+")

    def test_apply_operation_impl(self) -> None:
        """Test _apply_operation_impl method"""
        with mock.patch(
            "wandas.processing.time_series.create_operation"
        ) as mock_create_op:
            mock_op: Any = mock.MagicMock()
            mock_create_op.return_value = mock_op
            mock_processed_data: DaArray = mock.MagicMock(spec=DaArray)
            mock_op.process.return_value = mock_processed_data

            # 適切な型を持つモックオブジェクトを作成
            mock_result = mock.MagicMock(spec=SpectralFrame)

            with mock.patch.object(
                self.frame, "_create_new_instance", return_value=mock_result
            ) as mock_create_new_instance:
                operation_name: str = "test_op"
                params: dict[str, Any] = {"param1": "value1"}
                result: SpectralFrame = self.frame._apply_operation_impl(
                    operation_name, **params
                )

            mock_create_op.assert_called_once_with(
                operation_name, self.sampling_rate, **params
            )
            mock_op.process.assert_called_once_with(self.data)

            expected_metadata: dict[str, Any] = {
                **self.frame.metadata,
                operation_name: params,
            }
            expected_history: list[dict[str, Any]] = self.frame.operation_history.copy()
            expected_history.append({"operation": operation_name, "params": params})
            mock_create_new_instance.assert_called_once_with(
                data=mock_processed_data,
                metadata=expected_metadata,
                operation_history=expected_history,
            )

            # 戻り値の検証
            assert result is mock_result

    def test_noct_synthesis_sampling_rate_error(self) -> None:
        """
        Test that noct_synthesis raises ValueError when sampling rate is not 48000Hz
        """
        # SpectralFrameのsampling_rateは44100Hzで設定されているので、
        # noct_synthesisを呼び出すとValueErrorが発生するはず
        with pytest.raises(
            ValueError,
            match="noct_synthesisは48000Hzのサンプリングレートでのみ使用できます",
        ):
            self.frame.noct_synthesis(fmin=125.0, fmax=8000.0, n=3)

    def test_noct_synthesis(self) -> None:
        """Test noct_synthesis method"""
        # 正しいサンプリングレートでSpectralFrameを作成
        correct_sr_frame = SpectralFrame(
            data=self.data,
            sampling_rate=48000,  # 正しいサンプリングレート
            n_fft=self.n_fft,
            window=self.window,
            label="test_frame",
            metadata={"test": "metadata"},
            channel_metadata=self.channel_metadata,
        )

        with (
            mock.patch("wandas.frames.noct.NOctFrame") as mock_noct_frame,
            mock.patch(
                "wandas.processing.time_series.create_operation"
            ) as mock_create_op,
        ):
            # NOctSynthesisオペレーションのモック設定
            mock_noct_op: Any = mock.MagicMock()
            mock_create_op.return_value = mock_noct_op
            mock_spectrum_data: DaArray = mock.MagicMock(spec=DaArray)
            mock_noct_op.process.return_value = mock_spectrum_data

            # NOctFrameのモック設定
            mock_result: Any = mock.MagicMock()
            mock_noct_frame.return_value = mock_result

            # テスト実行
            fmin: float = 125.0
            fmax: float = 8000.0
            n: int = 3
            G: int = 10  # noqa: N806
            fr: int = 1000

            result = correct_sr_frame.noct_synthesis(
                fmin=fmin, fmax=fmax, n=n, G=G, fr=fr
            )

            # オペレーション作成の検証
            mock_create_op.assert_called_once_with(
                "noct_synthesis", 48000, fmin=fmin, fmax=fmax, n=n, G=G, fr=fr
            )

            # プロセスの呼び出し検証
            mock_noct_op.process.assert_called_once_with(correct_sr_frame._data)

            # NOctFrameの作成検証
            mock_noct_frame.assert_called_once_with(
                data=mock_spectrum_data,
                sampling_rate=48000,
                fmin=fmin,
                fmax=fmax,
                n=n,
                G=G,
                fr=fr,
                label=f"1/{n}Oct of {correct_sr_frame.label}",
                metadata={
                    **correct_sr_frame.metadata,
                    "fmin": fmin,
                    "fmax": fmax,
                    "n": n,
                    "G": G,
                    "fr": fr,
                },
                operation_history=[
                    *correct_sr_frame.operation_history,
                    {
                        "operation": "noct_synthesis",
                        "params": {
                            "fmin": fmin,
                            "fmax": fmax,
                            "n": n,
                            "G": G,
                            "fr": fr,
                        },
                    },
                ],
                channel_metadata=correct_sr_frame._channel_metadata,
                previous=correct_sr_frame,
            )

            # 結果の検証
            assert result is mock_result
