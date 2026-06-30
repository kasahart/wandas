import abc
from collections import Counter, defaultdict, namedtuple
from typing import Any
from unittest import mock

import cloudpickle
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from dask.base import tokenize

from wandas.frames.channel import ChannelFrame
from wandas.processing import base as base_module
from wandas.processing.base import (
    _OPERATION_MODULES,
    _OPERATION_REGISTRY,
    AudioOperation,
    BinaryOperation,
    _config_values_equal,
    _operand_descriptor,
    _snapshot_config_value,
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

            def _process(self, x: NDArrayReal) -> NDArrayReal:
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

            def _process(self, x: NDArrayReal) -> NDArrayReal:
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

            def _process(self, x: NDArrayReal) -> NDArrayReal:
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


def test_operand_descriptor_handles_scalar_and_shape_only_values() -> None:
    class ShapeOnly:
        shape = (2, 3)

    assert _operand_descriptor(1 + 2j) == {"type": "complex", "real": 1.0, "imag": 2.0}
    assert _operand_descriptor(True) == {"type": "bool", "value": True}
    assert _operand_descriptor(ShapeOnly()) == {"type": "ShapeOnly", "shape": [2, 3]}
    assert _operand_descriptor(object()) == {"type": "object"}


def test_binary_operation_params_delegate_to_params() -> None:
    operation = BinaryOperation(symbol="+", operand_kind="scalar", operand=2.0)

    assert operation.params == {
        "symbol": "+",
        "operand_kind": "scalar",
        "operand": {"type": "float", "value": 2.0},
    }


class TestAudioOperation:
    """Test AudioOperation base class."""

    @staticmethod
    def _make_test_op_class() -> type:
        class _TestOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "test_op"

            def __init__(self, sampling_rate: float, *, pure: bool = True, **params: Any) -> None:
                self._test_params = {key: _snapshot_config_value(value) for key, value in params.items()}
                super().__init__(sampling_rate, pure=pure, **params)

            def to_params(self) -> dict[str, Any]:
                return {key: _snapshot_config_value(value) for key, value in self._test_params.items()}

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x * 2

            def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
                return input_shape

        return _TestOp

    def test_process_doubles_input(self) -> None:
        """process() applies _process and returns correct DaskArray result."""
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000)
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dask_data = da_from_array(data, chunks=(1, -1))

        result = op.process(dask_data)

        assert isinstance(result, DaArray)
        np.testing.assert_array_equal(result.compute(), data * 2)

    def test_process_uses_calculate_output_dtype(self) -> None:
        class FloatOutputOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "float_output_op"

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x.astype(np.float64)

            def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
                return np.dtype(np.float64)

        data = da_from_array(np.array([[1, 2, 3]], dtype=np.int16), chunks=(1, -1))
        result = FloatOutputOperation(16000).process(data)

        assert result.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(result.compute(), np.array([[1.0, 2.0, 3.0]], dtype=np.float64))

    def test_process_array_removed_from_audio_operation(self) -> None:
        class SimpleOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "simple_removed_process_array_op"

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x

        op = SimpleOp(16000)

        assert not hasattr(op, "process_array")

    def test_process_rejects_1d_direct_input(self) -> None:
        """process() requires Frame-internal ch-first Dask arrays."""
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000)
        data = da_from_array(np.array([1.0, 2.0, 3.0]), chunks=(-1,))

        with pytest.raises(ValueError, match=r"AudioOperation.process requires channel-first data"):
            op.process(data)

    def test_process_rejects_1d_additional_dask_input(self) -> None:
        """Multi-input lazy operations require ch-first Dask arrays for all inputs."""

        class AddOtherOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "add_other_op"
            _expected_input_count = 2

            def _process(self, x: NDArrayReal, other: NDArrayReal) -> NDArrayReal:
                return x + other

        data = da_from_array(np.array([[1.0, 2.0, 3.0]]), chunks=(1, -1))
        other = da_from_array(np.array([0.1, 0.2, 0.3]), chunks=(-1,))
        op = AddOtherOperation(16000)

        with pytest.raises(ValueError, match=r"AudioOperation.process requires channel-first data"):
            op.process(data, other)

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

    def test_sampling_rate_is_read_only_after_initialization(self) -> None:
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000)

        with pytest.raises(AttributeError):
            setattr(op, "sampling_rate", 8000)

        assert op.sampling_rate == 16000

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

    def test_super_only_config_operation_exposes_base_params_through_lineage_and_compute(self) -> None:
        class SuperOnlyGainOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "super_only_gain_op"

            def __init__(self, sampling_rate: float, gain: float):
                super().__init__(sampling_rate, gain=gain)

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x * self._config_snapshot()["gain"]

        try:
            register_operation(SuperOnlyGainOperation)
            op = create_operation("super_only_gain_op", 16000, gain=2.0)
            data = da_from_array(np.array([[1.0, 2.0]]), chunks=(1, -1))
            frame = ChannelFrame(data, sampling_rate=16000)
            result = ChannelFrame(
                op.process(data),
                sampling_rate=16000,
                lineage=frame._lineage_with_operation(op, frame.lineage),
            )

            assert op.params == {"gain": 2.0}
            assert op.to_params() == {"gain": 2.0}
            assert result.operation_history == [{"operation": "super_only_gain_op", "params": {"gain": 2.0}}]
            np.testing.assert_array_equal(result.compute(), np.array([[2.0, 4.0]]))
        finally:
            _OPERATION_REGISTRY.pop("super_only_gain_op", None)

    def test_base_config_snapshots_constructor_inputs_for_params_compute_and_history(self) -> None:
        class BaseConfigOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "base_config_snapshot_op"

            def __init__(self, sampling_rate: float, config: dict[str, float]):
                super().__init__(sampling_rate, config=config)

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x * self._config_snapshot()["config"]["gain"]

        config = {"gain": 2.0}
        op = BaseConfigOperation(16000, config=config)
        data = da_from_array(np.array([[1.0, 2.0]]), chunks=(1, -1))
        frame = ChannelFrame(data, sampling_rate=16000)
        result = ChannelFrame(
            op.process(data),
            sampling_rate=16000,
            lineage=frame._lineage_with_operation(op, frame.lineage),
        )

        config["gain"] = 99.0

        assert op.params["config"] == {"gain": 2.0}
        assert op.to_params()["config"] == {"gain": 2.0}
        assert result.operation_history[-1]["params"] == {"config": {"gain": 2.0}}
        np.testing.assert_array_equal(result.compute(), np.array([[2.0, 4.0]]))

    def test_base_config_returned_nested_values_do_not_change_pending_compute(self) -> None:
        class BaseConfigOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "base_config_pending_op"

            def __init__(self, sampling_rate: float, config: dict[str, float]):
                super().__init__(sampling_rate, config=config)

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x * self._config_snapshot()["config"]["gain"]

        op = BaseConfigOperation(16000, config={"gain": 2.0})
        data = da_from_array(np.array([[1.0, 2.0]]), chunks=(1, -1))
        result = op.process(data)

        op.params["config"]["gain"] = 99.0
        op.to_params()["config"]["gain"] = 99.0

        assert op.params["config"] == {"gain": 2.0}
        np.testing.assert_array_equal(result.compute(), np.array([[2.0, 4.0]]))

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

            def _process(self, x: NDArrayReal) -> NDArrayReal:
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
                mock_delayed.return_value = lambda *args: mock.MagicMock()
                op.process(da_from_array(test_array, chunks=(1, -1)))
                _, kwargs = mock_delayed.call_args
                assert kwargs["pure"] is pure_val
                assert kwargs["name"] == op.name

    def test_get_metadata_updates_returns_empty_dict(self) -> None:
        """Base class get_metadata_updates() returns empty dict by default."""
        test_op_cls = self._make_test_op_class()
        assert test_op_cls(16000).get_metadata_updates() == {}

    def test_concrete_config_properties_are_read_only(self) -> None:
        op = HighPassFilter(16000, cutoff=500)

        with pytest.raises(AttributeError):
            setattr(op, "cutoff", 1000)
        with pytest.raises(AttributeError):
            del op.cutoff

        assert op.cutoff == 500
        assert op.params["cutoff"] == 500

    def test_subclass_to_params_defines_operation_params(self) -> None:
        class SeededConfigOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "seeded_config_op"

            def __init__(self, sampling_rate: float, gain: str):
                self._gain = float(gain)
                super().__init__(sampling_rate, gain=self._gain)

            @property
            def gain(self) -> float:
                return self._gain

            def to_params(self) -> dict[str, float]:
                return {"gain": self._gain}

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x * self._gain

        op = SeededConfigOperation(16000, gain="2.0")
        data = da_from_array(np.array([[1.0, 2.0]]), chunks=(1, -1))

        assert op.gain == 2.0
        assert op.params["gain"] == 2.0
        np.testing.assert_array_equal(op.process(data).compute(), np.array([[2.0, 4.0]]))

    def test_params_are_read_only_after_operation_creation(self) -> None:
        op = HighPassFilter(16000, cutoff=500)

        with pytest.raises(TypeError):
            op.params["cutoff"] = 1000  # ty: ignore[invalid-assignment]
        with pytest.raises(AttributeError):
            op.params = {"cutoff": 1000}  # ty: ignore[invalid-assignment]

        assert op.params["cutoff"] == 500
        assert op.cutoff == 500

    def test_params_copy_mutation_does_not_change_operation(self) -> None:
        op = HighPassFilter(16000, cutoff=500)

        params = op.params.copy()
        params["cutoff"] = 1000

        assert op.params["cutoff"] == 500
        assert op.cutoff == 500

    def test_params_view_exposes_read_only_mapping_helpers(self) -> None:
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000, gain=2.0)
        params = op.params

        assert len(params) == 1
        assert repr(params) == "{'gain': 2.0}"
        assert params.copy() == {"gain": 2.0}
        assert params != [("gain", 2.0)]

        with pytest.raises(TypeError):
            del params["gain"]  # type: ignore[attr-defined]

        assert op.params == {"gain": 2.0}

    def test_operation_params_nested_values_remain_defensive(self) -> None:
        class NestedParamsOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "nested_params_op"

            def __init__(self, sampling_rate: float, config: dict[str, float]):
                self._test_config = _snapshot_config_value(config)
                super().__init__(sampling_rate, config=config)

            def to_params(self) -> dict[str, Any]:
                return {"config": self._test_config}

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x

        op = NestedParamsOperation(16000, config={"gain": 2.0})

        copied_params = op.params.copy()
        copied_params["config"]["gain"] = 99.0
        nested_config = op.params["config"]
        nested_config["gain"] = 99.0

        assert op.params["config"]["gain"] == 2.0

    def test_params_deletion_does_not_change_generated_params(self) -> None:
        op = HighPassFilter(16000, cutoff=500)

        with pytest.raises(TypeError):
            del op.params["cutoff"]  # ty: ignore[not-subscriptable]

        assert op.params["cutoff"] == 500
        assert op.cutoff == 500

    def test_subclasses_snapshot_mutable_config_for_to_params(self) -> None:
        class PrivateConfigOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "private_config_op"

            def __init__(self, sampling_rate: float, config: dict[str, float]):
                self._test_config = _snapshot_config_value(config)
                super().__init__(sampling_rate, config=config)

            def to_params(self) -> dict[str, Any]:
                return {"config": self._test_config}

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x * self._test_config["gain"]

        config = {"gain": 2.0}
        op = PrivateConfigOperation(16000, config=config)
        config["gain"] = 99.0

        assert op.params["config"] == {"gain": 2.0}

    def test_non_config_mutable_public_attribute_returns_live_value(self) -> None:
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000, gain=2.0)
        cache = {"last": 1.0}
        object.__setattr__(op, "cache", cache)

        assert op.cache is cache

    def test_empty_public_mapping_after_base_init_returns_live_value(self) -> None:
        class CachedOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "cached_op"

            def __init__(self, sampling_rate: float):
                super().__init__(sampling_rate)
                self.cache: dict[str, float] = {}

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x

        op = CachedOperation(16000)

        op.cache["last"] = 1.0

        assert object.__getattribute__(op, "cache") == {"last": 1.0}

    def test_public_cache_with_param_key_returns_live_value(self) -> None:
        class CachedOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "param_key_cache_op"

            def __init__(self, sampling_rate: float, gain: float):
                super().__init__(sampling_rate, gain=gain)
                self.cache: dict[str, float | None] = {"gain": None}

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                self.cache["gain"] = 2.0
                return x

        op = CachedOperation(16000, gain=2.0)

        op.cache["gain"] = 3.0

        assert object.__getattribute__(op, "cache") == {"gain": 3.0}

    def test_snapshot_config_value_preserves_defaultdict_behavior(self) -> None:
        config: defaultdict[str, float] = defaultdict(lambda: 2.0, {"gain": 3.0})

        snapshot = _snapshot_config_value(config)
        config["gain"] = 99.0

        assert isinstance(snapshot, defaultdict)
        assert snapshot["missing"] == 2.0
        assert snapshot["gain"] == 3.0

    def test_snapshot_config_value_preserves_mutable_mapping_items(self) -> None:
        config = Counter({"gain": 2})

        snapshot = _snapshot_config_value(config)

        assert isinstance(snapshot, Counter)
        assert snapshot == Counter({"gain": 2})

    def test_snapshot_config_value_falls_back_for_mutable_mapping_copy_failure(self) -> None:
        class BadMutableMapping(dict[str, float]):
            def clear(self) -> None:
                raise RuntimeError("cannot clear")

        snapshot = _snapshot_config_value(BadMutableMapping({"gain": 2.0}))

        assert type(snapshot) is dict
        assert snapshot == {"gain": 2.0}

    def test_mapping_subclass_public_config_keeps_behavior_after_snapshot(self) -> None:
        class DefaultdictConfigOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "defaultdict_config_op"

            def __init__(self, sampling_rate: float, config: defaultdict[str, float]):
                self.config = config
                super().__init__(sampling_rate, config=config)

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x * self.config["missing"]

        op = DefaultdictConfigOperation(16000, defaultdict(lambda: 2.0))
        data = da_from_array(np.array([[1.0, 2.0]]), chunks=(1, -1))

        assert isinstance(object.__getattribute__(op, "config"), defaultdict)
        np.testing.assert_array_equal(op.process(data).compute(), np.array([[2.0, 4.0]]))

    def test_operation_params_equality_handles_array_values(self) -> None:
        class ArrayParamsOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "array_params_op"

            def __init__(self, sampling_rate: float, reference: NDArrayReal):
                self._reference = _snapshot_config_value(reference)
                self._test_config = {"weights": _snapshot_config_value(reference)}
                super().__init__(sampling_rate, reference=reference, config={"weights": reference})

            def to_params(self) -> dict[str, Any]:
                return {"reference": self._reference, "config": self._test_config}

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x

        reference = np.array([1.0, 2.0])
        op = ArrayParamsOperation(16000, reference=reference)

        assert op.params == op.params
        assert op.params == {"reference": np.array([1.0, 2.0]), "config": {"weights": np.array([1.0, 2.0])}}
        assert op.params != {"reference": np.array([1.0, 3.0]), "config": {"weights": np.array([1.0, 2.0])}}

    def test_snapshot_config_value_preserves_dask_arrays(self) -> None:
        dask_array = da_from_array(np.array([1.0]), chunks=(1,))

        assert _snapshot_config_value(dask_array) is dask_array

    def test_snapshot_config_value_returns_immutable_scalars_without_deepcopy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail_deepcopy(value: object) -> object:
            raise AssertionError(f"deepcopy should not run for {value!r}")

        monkeypatch.setattr("wandas.processing.base.copy.deepcopy", fail_deepcopy)

        for value in [None, True, 1, 1.5, "gain", b"gain", 1 + 2j]:
            assert _snapshot_config_value(value) is value

    def test_config_value_snapshots_only_requested_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        test_op_cls = self._make_test_op_class()
        target = {"gain": 2.0}
        untouched = {"bias": 1.0}
        op = test_op_cls(16000, target=target, untouched=untouched)
        config = object.__getattribute__(op, "_config")
        calls: list[object] = []
        real_snapshot = base_module._snapshot_config_value

        def record_snapshot(value: object) -> object:
            calls.append(value)
            return real_snapshot(value)

        monkeypatch.setattr(base_module, "_snapshot_config_value", record_snapshot)

        snapshot = op._config_value("target")

        assert snapshot == {"gain": 2.0}
        assert calls[0] is config["target"]
        assert config["untouched"] not in calls
        snapshot["gain"] = 99.0
        assert op._config["target"] == {"gain": 2.0}

    def test_config_values_equal_handles_non_matching_containers(self) -> None:
        assert not _config_values_equal({"left": 1}, {"right": 1})
        assert not _config_values_equal([1], (1,))
        assert not _config_values_equal([1], [1, 2])
        assert _config_values_equal([{"gain": np.array([1.0])}], [{"gain": np.array([1.0])}])

    def test_config_values_equal_handles_ambiguous_or_failing_comparisons(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class EqRaises:
            def __eq__(self, other: object) -> bool:
                raise RuntimeError("no eq")

        class EqDaskArray:
            def __eq__(self, other: object) -> Any:
                return da_from_array(np.array([True]), chunks=(1,))

        class EqNumpyArray:
            def __init__(self, values: list[bool]):
                self.values = values

            def __eq__(self, other: object) -> Any:
                return np.array(self.values)

        dask_array = da_from_array(np.array([1.0]), chunks=(1,))
        monkeypatch.setattr("wandas.processing.base.np.array_equal", mock.Mock(side_effect=RuntimeError("no array")))

        assert not _config_values_equal(np.array([1.0]), np.array([1.0]))
        assert not _config_values_equal(EqRaises(), object())
        assert not _config_values_equal(dask_array, 1.0)
        assert not _config_values_equal(EqDaskArray(), object())
        assert _config_values_equal(EqNumpyArray([True, True]), object())
        assert not _config_values_equal(EqNumpyArray([True, False]), object())

    def test_operation_params_equality_handles_ambiguous_bool_results(self) -> None:
        class AmbiguousComparison:
            def __bool__(self) -> bool:
                raise ValueError("ambiguous")

        class EqAmbiguousBool:
            def __eq__(self, other: object) -> Any:
                return AmbiguousComparison()

        class AmbiguousParamsOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "ambiguous_params_op"

            def __init__(self, sampling_rate: float, config: object):
                self._test_config = config
                super().__init__(sampling_rate, config=config)

            def to_params(self) -> dict[str, object]:
                return {"config": self._test_config}

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x

        op = AmbiguousParamsOperation(16000, config=EqAmbiguousBool())

        assert op.params != op.params

    def test_snapshot_config_value_copies_container_variants(self) -> None:
        readonly = np.array([1.0])
        readonly.flags.writeable = False

        copied_array = _snapshot_config_value(readonly)
        assert copied_array is not readonly
        assert copied_array.flags.writeable

        snapshot = _snapshot_config_value(
            {
                "tuple": ({"x": 1},),
                "list": [{"y": 2}],
                "set": {1, 2},
                "frozenset": frozenset({3, 4}),
            }
        )

        assert snapshot["tuple"] == ({"x": 1},)
        assert snapshot["list"] == [{"y": 2}]
        assert snapshot["set"] == {1, 2}
        assert snapshot["frozenset"] == frozenset({3, 4})
        snapshot["tuple"][0]["x"] = 99
        snapshot["list"][0]["y"] = 99
        assert _snapshot_config_value({"tuple": ({"x": 1},), "list": [{"y": 2}]}) == {
            "tuple": ({"x": 1},),
            "list": [{"y": 2}],
        }

    def test_snapshot_config_value_preserves_ndarray_subclasses(self) -> None:
        masked = np.ma.array([1.0, 2.0], mask=[False, True])

        snapshot = _snapshot_config_value(masked)
        masked[0] = 99.0

        assert isinstance(snapshot, np.ma.MaskedArray)
        np.testing.assert_array_equal(snapshot.data, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(snapshot.mask, np.array([False, True]))

    def test_snapshot_config_value_falls_back_when_ndarray_copy_fails(self) -> None:
        class CopyFailingArray(np.ndarray):
            def copy(self, order: Any = "C") -> Any:
                raise RuntimeError("cannot copy")

        array = np.asarray([1.0, 2.0]).view(CopyFailingArray)

        snapshot = _snapshot_config_value(array)

        assert isinstance(snapshot, CopyFailingArray)
        assert snapshot is not array
        np.testing.assert_array_equal(snapshot, np.array([1.0, 2.0]))

    def test_snapshot_config_value_preserves_tuple_subclasses(self) -> None:
        Config = namedtuple("Config", ["gain"])
        config = Config(gain={"value": 2.0})

        snapshot = _snapshot_config_value(config)
        config.gain["value"] = 99.0

        assert isinstance(snapshot, Config)
        assert snapshot.gain == {"value": 2.0}

    def test_snapshot_config_value_preserves_tuple_subclasses_with_iterable_constructor(self) -> None:
        class IterableTuple(tuple):
            def __new__(cls, values: tuple[Any, ...]):
                return super().__new__(cls, values)

        config = IterableTuple(({"gain": 2.0}, {"bias": 1.0}))

        snapshot = _snapshot_config_value(config)
        config[0]["gain"] = 99.0

        assert isinstance(snapshot, IterableTuple)
        assert snapshot == ({"gain": 2.0}, {"bias": 1.0})

    def test_tuple_subclass_public_config_keeps_behavior_after_snapshot(self) -> None:
        Config = namedtuple("Config", ["gain"])

        class TupleSubclassConfigOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "tuple_subclass_config_op"

            def __init__(self, sampling_rate: float, cfg: Any):
                self.cfg = cfg
                super().__init__(sampling_rate, cfg=cfg)

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x * self.cfg.gain

        op = TupleSubclassConfigOperation(16000, Config(gain=2.0))
        data = da_from_array(np.array([[1.0, 2.0]]), chunks=(1, -1))

        assert isinstance(object.__getattribute__(op, "cfg"), Config)
        np.testing.assert_array_equal(op.process(data).compute(), np.array([[2.0, 4.0]]))

    def test_snapshot_config_value_falls_back_when_deepcopy_fails(self) -> None:
        class NonCopyable:
            def __deepcopy__(self, memo: dict[int, object]) -> object:
                raise RuntimeError("no copy")

        value = NonCopyable()

        assert _snapshot_config_value(value) is value

    def test_public_attributes_remain_ordinary_python_state(self) -> None:
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000)
        config = {"gain": 2.0}

        op.config = config
        op.config["gain"] = 3.0
        del op.config

        assert "config" not in object.__getattribute__(op, "__dict__")

    def test_params_view_equality_requires_same_keys(self) -> None:
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000, gain=2.0)

        assert op.params != {"other": 2.0}

    def test_operation_tokenize_is_stable_after_initialization(self) -> None:
        op = HighPassFilter(16000, cutoff=500)

        assert tokenize(op) == tokenize(op)

    def test_cloudpickle_serializes_operation_lineage_objects(self) -> None:
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

    def test_inherited_operation_init_is_not_wrapped_twice(self) -> None:
        class ParentOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "parent_wrapped_op"

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x

        parent_init = ParentOperation.__init__

        class ChildOperation(ParentOperation):
            name = "child_wrapped_op"

        assert ChildOperation.__init__ is parent_init

    def test_get_display_name_returns_none(self) -> None:
        """Base class get_display_name() returns None by default."""
        test_op_cls = self._make_test_op_class()
        assert test_op_cls(16000).get_display_name() is None

    def test_incomplete_operation_requires_process_kernel(self) -> None:
        class IncompleteOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "incomplete_op"

        op = IncompleteOp(16000)

        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            op._process(np.array([[1.0, 2.0, 3.0]]))

    def test_process_accepts_variadic_inputs_for_multi_input_subclass(self) -> None:
        """A subclass can process multiple Dask inputs lazily."""

        class AddInputs(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "add_inputs_op"
            _expected_input_count = 2

            def _process(self, *inputs: NDArrayReal) -> NDArrayReal:
                left, right = inputs
                return left + right

        left = da_from_array(np.array([[1.0, 2.0, 3.0]]), chunks=(1, -1))
        right = da_from_array(np.array([[0.5, 0.25, 0.125]]), chunks=(1, -1))
        op = AddInputs(16000)

        result = op.process(left, right)

        assert isinstance(result, DaArray)
        assert result.shape == left.shape
        np.testing.assert_allclose(result.compute(), np.array([[1.5, 2.25, 3.125]]))

    def test_process_uses_result_type_dtype_metadata_for_multi_input_subclass(self) -> None:
        """Default multi-input dtype metadata follows NumPy result type."""

        class AddInputs(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "add_inputs_dtype_op"
            _expected_input_count = 2

            def _process(self, *inputs: NDArrayReal) -> NDArrayReal:
                left, right = inputs
                return left + right

        left = da_from_array(np.array([[1, 2, 3]], dtype=np.int16), chunks=(1, -1))
        right = da_from_array(np.array([[0.5, 0.25, 0.125]], dtype=np.float32), chunks=(1, -1))
        op = AddInputs(16000)

        result = op.process(left, right)

        expected_dtype = np.result_type(left.dtype, right.dtype)
        assert result.dtype == expected_dtype
        assert result.compute().dtype == expected_dtype

    def test_process_rejects_extra_inputs_before_dask_compute(self) -> None:
        """Base process validates input arity when building the Dask graph."""

        class DoubleOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "double_early_reject_op"

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x * 2.0

        first = da_from_array(np.array([[1.0, 2.0, 3.0]]), chunks=(1, -1))
        second = da_from_array(np.array([[4.0, 5.0, 6.0]]), chunks=(1, -1))
        op = DoubleOp(16000)

        with pytest.raises(ValueError, match="Expected exactly one input"):
            op.process(first, second)

    def test_process_override_rejects_extra_inputs_without_manual_validation(self) -> None:
        """Subclass process overrides are wrapped with base arity validation."""

        class OverrideProcessOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "override_process_early_reject_op"

            def process(self, data: DaArray, *inputs: DaArray) -> DaArray:
                return self._mark_array(data)

        first = da_from_array(np.array([[1.0, 2.0, 3.0]]), chunks=(1, -1))
        second = da_from_array(np.array([[4.0, 5.0, 6.0]]), chunks=(1, -1))
        op = OverrideProcessOp(16000)

        with pytest.raises(ValueError, match="Expected exactly one input"):
            op.process(first, second)

    def test_process_override_wrapper_rejects_1d_direct_input(self) -> None:
        """Subclass process overrides inherit channel-first validation."""

        class NativeOverrideOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "native_override_op"

            def process(self, data: DaArray, *inputs: DaArray) -> DaArray:
                return self._mark_array(data + 1)

        data = da_from_array(np.array([1.0, 2.0, 3.0]), chunks=(-1,))
        op = NativeOverrideOperation(16000)

        with pytest.raises(ValueError, match=r"AudioOperation.process requires channel-first data"):
            op.process(data)

    def test_default_process_rejects_extra_inputs(self) -> None:
        """Single-input operations fail clearly when called with multiple inputs."""

        class DoubleOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "double_single_input_op"

            def _process(self, x: NDArrayReal) -> NDArrayReal:
                return x * 2.0

        first = da_from_array(np.array([[1.0, 2.0, 3.0]]), chunks=(1, -1))
        second = da_from_array(np.array([[4.0, 5.0, 6.0]]), chunks=(1, -1))
        op = DoubleOp(16000)

        with pytest.raises(ValueError, match="Expected exactly one input"):
            op.process(first, second).compute()

    def test_calculate_output_shape_default_returns_input(self) -> None:
        """Default calculate_output_shape() returns input shape unchanged."""

        class SimpleOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "simple_op"

            def _process(self, x: NDArrayReal) -> NDArrayReal:
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
