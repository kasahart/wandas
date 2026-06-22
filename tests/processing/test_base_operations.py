import abc
import copy
from unittest import mock

import cloudpickle
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from dask.base import tokenize

from wandas.processing.base import (
    _OPERATION_MODULES,
    _OPERATION_REGISTRY,
    AudioOperation,
    FrozenDict,
    _freeze_config_value,
    create_operation,
    get_operation,
    register_lazy_operation,
    register_operation,
)
from wandas.processing.filters import HighPassFilter
from wandas.processing.spectral import STFT
from wandas.utils.dask_helpers import da_from_array
from wandas.utils.types import NDArrayReal


class TestOperationRegistry:
    """Test registry-related functions."""

    def test_get_operation_normal(self) -> None:
        """Test get_operation returns a registered operation."""
        # Test for existing operations
        assert "highpass_filter" in _OPERATION_REGISTRY
        assert "lowpass_filter" in _OPERATION_REGISTRY

    def test_get_operation_imports_lazy_registered_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class LazyTestOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "lazy_test_op"

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x

        def fake_import_module(module_name: str) -> object:
            assert module_name == "tests.fake_lazy_module"
            register_operation(LazyTestOperation)
            return object()

        monkeypatch.setattr("wandas.processing.base.importlib.import_module", fake_import_module)
        register_lazy_operation("lazy_test_op", "tests.fake_lazy_module")

        try:
            assert get_operation("lazy_test_op") is LazyTestOperation
        finally:
            _OPERATION_MODULES.pop("lazy_test_op", None)
            _OPERATION_REGISTRY.pop("lazy_test_op", None)

    def test_get_operation_error(self) -> None:
        """Test get_operation raises ValueError for unknown operations."""
        with pytest.raises(ValueError, match="Unknown operation type:"):
            get_operation("nonexistent_operation")

    def test_register_operation_normal(self) -> None:
        """Test registering a valid operation."""

        # Create a test operation class
        class TestOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "test_register_op"

            def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
                return input_shape

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x

        # Register and verify
        register_operation(TestOperation)
        assert get_operation("test_register_op") == TestOperation

        # Clean up
        if "test_register_op" in _OPERATION_REGISTRY:
            del _OPERATION_REGISTRY["test_register_op"]

    def test_register_operation_same_class_is_idempotent(self) -> None:
        class IdempotentOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "idempotent_op"

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x

        try:
            register_operation(IdempotentOperation)
            register_operation(IdempotentOperation)

            assert _OPERATION_REGISTRY["idempotent_op"] is IdempotentOperation
        finally:
            _OPERATION_REGISTRY.pop("idempotent_op", None)

    def test_register_operation_error(self) -> None:
        """Test registering an invalid class raises TypeError."""

        # Create a non-AudioOperation class
        class InvalidClass:
            pass

        with pytest.raises(TypeError, match="Strategy class must inherit from AudioOperation."):
            register_operation(InvalidClass)

    def test_create_operation_with_different_types(self) -> None:
        """Test creating operations of different types."""
        # Create a highpass filter operation
        hpf_op = create_operation("highpass_filter", 16000, cutoff=150.0, order=6)

        assert isinstance(hpf_op, HighPassFilter)
        assert hpf_op.cutoff == 150.0
        assert hpf_op.order == 6


class TestAudioOperation:
    """Test AudioOperation base class."""

    @staticmethod
    def _make_test_op_class() -> type:
        class _TestOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "test_op"

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x * 2

            def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
                return input_shape

        return _TestOp

    def test_process_doubles_input(self) -> None:
        """process() applies _process_array and returns correct DaskArray result."""
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000)
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dask_data = da_from_array(data, chunks=(1, -1))

        result = op.process(dask_data)

        assert isinstance(result, DaArray)
        np.testing.assert_array_equal(result.compute(), data * 2)

    def test_process_preserves_immutability(self) -> None:
        """Pillar 1: process() does not mutate the input DaskArray."""
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000)
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dask_data = da_from_array(data, chunks=(1, -1))
        data_copy = data.copy()

        _ = op.process(dask_data).compute()

        np.testing.assert_array_equal(dask_data.compute(), data_copy)

    def test_same_immutable_operation_instance_can_process_multiple_inputs(self) -> None:
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000)
        first = da_from_array(np.array([[1.0, 2.0]]), chunks=(1, -1))
        second = da_from_array(np.array([[3.0, 4.0]]), chunks=(1, -1))

        np.testing.assert_array_equal(op.process(first).compute(), np.array([[2.0, 4.0]]))
        np.testing.assert_array_equal(op.process(second).compute(), np.array([[6.0, 8.0]]))

    def test_delayed_execution_not_computed_early(self) -> None:
        """Pillar 1: process() preserves Dask lazy evaluation; no premature compute()."""
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000)
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dask_data = da_from_array(data, chunks=(1, -1))

        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = op.process(dask_data)
            mock_compute.assert_not_called()
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_validate_params_raises_on_invalid(self) -> None:
        """validate_params() raises ValueError for invalid parameters in __init__."""

        class ValidatedOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "validated_op"

            def __init__(self, sampling_rate: float, value: int):
                self.value = value
                super().__init__(sampling_rate, value=value)

            def validate_params(self) -> None:
                if self.value < 0:
                    raise ValueError("Value must be non-negative")

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x

            def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
                return input_shape

        with pytest.raises(ValueError, match="Value must be non-negative"):
            _ = ValidatedOp(16000, -1)

    def test_pure_parameter_default_true(self) -> None:
        """Default pure parameter is True."""
        test_op_cls = self._make_test_op_class()
        assert test_op_cls(16000).pure is True

    def test_pure_parameter_explicit_true(self) -> None:
        """Explicit pure=True is stored correctly."""
        test_op_cls = self._make_test_op_class()
        assert test_op_cls(16000, pure=True).pure is True

    def test_pure_parameter_explicit_false(self) -> None:
        """Explicit pure=False is stored correctly."""
        test_op_cls = self._make_test_op_class()
        assert test_op_cls(16000, pure=False).pure is False

    def test_pure_parameter_forwarded_to_delayed(self) -> None:
        """pure parameter is forwarded to dask.delayed() call."""
        test_op_cls = self._make_test_op_class()
        test_array = np.array([[1.0, 2.0, 3.0]])

        for pure_val in (True, False):
            op = test_op_cls(16000, pure=pure_val)
            with mock.patch("wandas.processing.base.delayed") as mock_delayed:
                mock_delayed.return_value = lambda x: mock.MagicMock()
                op.process_array(test_array)
                _, kwargs = mock_delayed.call_args
                assert kwargs["pure"] is pure_val

    def test_get_metadata_updates_returns_empty_dict(self) -> None:
        """Base class get_metadata_updates() returns empty dict by default."""
        test_op_cls = self._make_test_op_class()
        assert test_op_cls(16000).get_metadata_updates() == {}

    def test_operation_attribute_reassignment_is_blocked_after_init(self) -> None:
        op = HighPassFilter(16000, cutoff=500)

        with pytest.raises(AttributeError):
            op.cutoff = 1000

    def test_operation_params_are_read_only_after_init(self) -> None:
        op = HighPassFilter(16000, cutoff=500)

        with pytest.raises(TypeError):
            op.params["cutoff"] = 1000  # type: ignore[index]

    def test_frozen_dict_behaves_like_read_only_mapping(self) -> None:
        params = FrozenDict({"gain": 2.0})

        assert len(params) == 1
        assert list(params) == ["gain"]
        assert repr(params) == "{'gain': 2.0}"
        assert params == {"gain": 2.0}
        assert params != object()
        assert params.copy() == {"gain": 2.0}
        assert copy.copy(params) is params
        assert copy.deepcopy(params) is params

    def test_freeze_config_value_recursively_freezes_container_variants(self) -> None:
        readonly = np.array([1.0])
        readonly.flags.writeable = False

        assert _freeze_config_value(FrozenDict({"x": 1})) == {"x": 1}
        assert _freeze_config_value(readonly) is readonly

        frozen = _freeze_config_value(
            {
                "tuple": ({"x": 1},),
                "list": [{"y": 2}],
                "set": {1, 2},
                "frozenset": frozenset({3, 4}),
            }
        )

        assert isinstance(frozen["tuple"][0], FrozenDict)
        assert isinstance(frozen["list"], tuple)
        assert isinstance(frozen["list"][0], FrozenDict)
        assert frozen["set"] == frozenset({1, 2})
        assert frozen["frozenset"] == frozenset({3, 4})

    def test_operation_tokenize_is_stable_after_freeze(self) -> None:
        op = HighPassFilter(16000, cutoff=500)

        assert tokenize(op) == tokenize(op)

    def test_cloudpickle_serializes_frozen_operations(self) -> None:
        def identity(x: NDArrayReal) -> NDArrayReal:
            return x

        from wandas.processing.custom import CustomOperation
        from wandas.processing.effects import Normalize

        operations = [
            HighPassFilter(16000, cutoff=500),
            Normalize(16000),
            STFT(16000),
            CustomOperation(16000, func=identity),
        ]

        for operation in operations:
            assert cloudpickle.loads(cloudpickle.dumps(operation)).params == operation.params

    def test_get_display_name_returns_none(self) -> None:
        """Base class get_display_name() returns None by default."""
        test_op_cls = self._make_test_op_class()
        assert test_op_cls(16000).get_display_name() is None

    def test_process_array_not_implemented_raises(self) -> None:
        """Calling _process_array on base class raises NotImplementedError."""

        class IncompleteOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "incomplete_op"

        op = IncompleteOp(16000)
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            op._process_array(np.array([[1.0, 2.0, 3.0]]))

    def test_calculate_output_shape_default_returns_input(self) -> None:
        """Default calculate_output_shape() returns input shape unchanged."""

        class SimpleOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "simple_op"

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x * 2

        op = SimpleOp(16000)
        assert op.calculate_output_shape(()) == ()
        assert op.calculate_output_shape((2, 100)) == (2, 100)
        assert op.calculate_output_shape((1, 1025)) == (1, 1025)

    def test_register_abstract_class_raises(self) -> None:
        """Registering an abstract AudioOperation subclass raises TypeError."""

        class AbstractOp(AudioOperation[NDArrayReal, NDArrayReal], abc.ABC):
            name = "abstract_op"

            @abc.abstractmethod
            def abstract_method(self) -> None:
                pass

        with pytest.raises(TypeError, match="Cannot register abstract"):
            register_operation(AbstractOp)
