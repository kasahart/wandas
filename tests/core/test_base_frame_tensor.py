"""Tests for BaseFrame tensor conversion methods."""

import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame
from wandas.utils.dask_helpers import da_from_array


class TestToTensorPyTorch:
    """Test to_tensor() method with PyTorch."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.data = np.linspace(0.1, 1.0, 32000, dtype=np.float32).reshape(2, 16000)
        self.dask_data: DaArray = da_from_array(self.data, chunks=(1, -1))
        self.channel_frame = ChannelFrame(data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio")

    def test_to_tensor_pytorch_default_cpu_correct_shape(self) -> None:
        """Test to_tensor() with PyTorch default device."""
        torch = pytest.importorskip("torch")

        tensor = self.channel_frame.to_tensor(framework="torch")

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == self.data.shape
        np.testing.assert_allclose(tensor.cpu().numpy(), self.data, rtol=1e-6)  # float32 precision tolerance

    def test_to_tensor_pytorch_cpu_explicit_correct_values(self) -> None:
        """Test to_tensor() with PyTorch CPU device."""
        torch = pytest.importorskip("torch")

        tensor = self.channel_frame.to_tensor(framework="torch", device="cpu")

        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"
        assert tensor.shape == self.data.shape
        np.testing.assert_allclose(tensor.cpu().numpy(), self.data, rtol=1e-6)  # float32 precision tolerance

    def test_to_tensor_pytorch_cuda_skips_if_unavailable(self) -> None:
        """Test to_tensor() with PyTorch CUDA device if available."""
        torch = pytest.importorskip("torch")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tensor = self.channel_frame.to_tensor(framework="torch", device="cuda")

        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cuda"
        assert tensor.shape == self.data.shape
        np.testing.assert_allclose(tensor.cpu().numpy(), self.data, rtol=1e-6)  # float32 precision tolerance

    def test_to_tensor_pytorch_cuda_device_index_preserved(self) -> None:
        """Test to_tensor() with specific PyTorch CUDA device if available."""
        torch = pytest.importorskip("torch")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tensor = self.channel_frame.to_tensor(framework="torch", device="cuda:0")

        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cuda"
        assert tensor.device.index == 0
        assert tensor.shape == self.data.shape

    def test_to_tensor_pytorch_missing_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test to_tensor() raises ImportError when PyTorch is not installed."""

        def raise_missing_torch(key: str, *, feature: str) -> None:
            assert key == "torch"
            assert feature == "tensor conversion with framework='torch'"
            raise ImportError(
                f"{feature} requires optional dependency {key!r}.\n"
                'Install it with: pip install "wandas[ml]"'
            )

        monkeypatch.setattr("wandas.core.base_frame.require_dependency", raise_missing_torch)
        with pytest.raises(ImportError, match=r'(?s)torch.*pip install "wandas\[ml\]"'):
            self.channel_frame.to_tensor(framework="torch")


class TestToTensorTensorFlow:
    """Test to_tensor() method with TensorFlow."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.data = np.linspace(0.1, 1.0, 32000, dtype=np.float32).reshape(2, 16000)
        self.dask_data: DaArray = da_from_array(self.data, chunks=(1, -1))
        self.channel_frame = ChannelFrame(data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio")

    def test_to_tensor_tensorflow_default_correct_shape(self) -> None:
        """Test to_tensor() with TensorFlow default device."""
        tf = pytest.importorskip("tensorflow")

        tensor = self.channel_frame.to_tensor(framework="tensorflow")

        assert isinstance(tensor, tf.Tensor)
        assert tensor.shape == self.data.shape
        np.testing.assert_allclose(tensor.numpy(), self.data, rtol=1e-6)  # float32 precision tolerance

    def test_to_tensor_tensorflow_cpu_explicit_correct_values(self) -> None:
        """Test to_tensor() with TensorFlow CPU device."""
        tf = pytest.importorskip("tensorflow")

        tensor = self.channel_frame.to_tensor(framework="tensorflow", device="/CPU:0")

        assert isinstance(tensor, tf.Tensor)
        assert tensor.shape == self.data.shape
        np.testing.assert_allclose(tensor.numpy(), self.data, rtol=1e-6)  # float32 precision tolerance

    def test_to_tensor_tensorflow_gpu_skips_if_unavailable(self) -> None:
        """Test to_tensor() with TensorFlow GPU device if available."""
        tf = pytest.importorskip("tensorflow")

        # Check if GPU is available
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            pytest.skip("GPU not available")

        tensor = self.channel_frame.to_tensor(framework="tensorflow", device="/GPU:0")

        assert isinstance(tensor, tf.Tensor)
        assert tensor.shape == self.data.shape
        np.testing.assert_allclose(tensor.numpy(), self.data, rtol=1e-6)  # float32 precision tolerance

    def test_to_tensor_tensorflow_missing_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test to_tensor() raises ImportError when TensorFlow is not installed."""

        def raise_missing_tensorflow(key: str, *, feature: str) -> None:
            assert key == "tensorflow"
            assert feature == "tensor conversion with framework='tensorflow'"
            raise ImportError(
                f"{feature} requires optional dependency {key!r}.\n"
                'Install it with: pip install "wandas[ml]"'
            )

        monkeypatch.setattr("wandas.core.base_frame.require_dependency", raise_missing_tensorflow)
        with pytest.raises(ImportError, match=r'(?s)tensorflow.*pip install "wandas\[ml\]"'):
            self.channel_frame.to_tensor(framework="tensorflow")


class TestToTensorErrorHandling:
    """Test to_tensor() error handling."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.data = np.linspace(0.1, 1.0, 32000, dtype=np.float32).reshape(2, 16000)
        self.dask_data: DaArray = da_from_array(self.data, chunks=(1, -1))
        self.channel_frame = ChannelFrame(data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio")

    def test_to_tensor_unsupported_framework_raises_value_error(self) -> None:
        """Test to_tensor() raises ValueError for unsupported framework."""
        with pytest.raises(ValueError, match="(?s)Unsupported framework.*jax"):
            self.channel_frame.to_tensor(framework="jax")

    def test_to_tensor_invalid_framework_string_raises_value_error(self) -> None:
        """Test to_tensor() with invalid framework type."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            self.channel_frame.to_tensor(framework="invalid")


class TestToNumpy:
    """Test to_numpy() method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.data = np.linspace(0.1, 1.0, 32000, dtype=np.float32).reshape(2, 16000)
        self.dask_data: DaArray = da_from_array(self.data, chunks=(1, -1))
        self.channel_frame = ChannelFrame(data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio")

    def test_to_numpy_returns_correct_shape_and_values(self) -> None:
        """Test to_numpy() returns correct NumPy array."""
        result = self.channel_frame.to_numpy()

        assert isinstance(result, np.ndarray)
        assert result.shape == self.data.shape
        np.testing.assert_allclose(result, self.data, rtol=1e-6)  # float32 precision tolerance

    def test_to_numpy_single_channel_squeezes_to_1d(self) -> None:
        """Test to_numpy() with single channel."""
        single_data = np.linspace(0.1, 1.0, 16000, dtype=np.float32).reshape(1, 16000)
        single_dask: DaArray = da_from_array(single_data, chunks=(1, -1))
        single_frame = ChannelFrame(data=single_dask, sampling_rate=self.sample_rate, label="single")

        result = single_frame.to_numpy()

        assert isinstance(result, np.ndarray)
        # Single channel should be squeezed to 1D
        assert result.ndim == 1
        assert result.shape == (16000,)

    def test_to_numpy_multi_channel_preserves_2d(self) -> None:
        """Test to_numpy() with multiple channels."""
        multi_data = np.linspace(0.1, 1.0, 64000, dtype=np.float32).reshape(4, 16000)
        multi_dask: DaArray = da_from_array(multi_data, chunks=(1, -1))
        multi_frame = ChannelFrame(data=multi_dask, sampling_rate=self.sample_rate, label="multi")

        result = multi_frame.to_numpy()

        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 16000)
        np.testing.assert_allclose(result, multi_data, rtol=1e-6)  # float32 precision tolerance

    def test_to_numpy_int32_dtype_preserved(self) -> None:
        """Test to_numpy() maintains data type."""
        int_data = np.arange(32000, dtype=np.int32).reshape(2, 16000) % 100
        int_dask: DaArray = da_from_array(int_data, chunks=(1, -1))
        int_frame = ChannelFrame(data=int_dask, sampling_rate=self.sample_rate, label="int")

        result = int_frame.to_numpy()

        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, int_data)
