import abc
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Any
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
    _config_values_equal,
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

    def test_public_attribute_reassignment_does_not_mutate_captured_config(self) -> None:
        op = HighPassFilter(16000, cutoff=500)

        op.cutoff = 1000

        assert op.cutoff == 500
        assert op.params["cutoff"] == 500

    def test_subclass_can_assign_public_attributes_after_base_init(self) -> None:
        class PostInitOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "post_init_op"

            def __init__(self, sampling_rate: float, gain: float):
                super().__init__(sampling_rate, gain=gain)
                self.gain = gain

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x * self.gain

        op = PostInitOperation(16000, gain=2.0)
        data = da_from_array(np.array([[1.0, 2.0]]), chunks=(1, -1))

        assert op.gain == 2.0
        np.testing.assert_array_equal(op.process(data).compute(), np.array([[2.0, 4.0]]))

    def test_custom_operation_can_assign_params_after_base_init(self) -> None:
        class ParamsAssignmentOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "params_assignment_op"

            def __init__(self, sampling_rate: float, gain: float):
                super().__init__(sampling_rate)
                self.params = {"gain": gain}
                self.gain = gain

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x * self.gain

        op = ParamsAssignmentOperation(16000, gain=2.0)
        params = op.params
        params["gain"] = 99.0

        assert op.params == {"gain": 99.0}
        assert op.gain == 2.0

    def test_custom_operation_can_update_params_in_place_after_base_init(self) -> None:
        class ParamsUpdateOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "params_update_op"

            def __init__(self, sampling_rate: float, gain: float):
                super().__init__(sampling_rate)
                self.params.update({"gain": gain})
                self.gain = gain

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x * self.gain

        op = ParamsUpdateOperation(16000, gain=2.0)

        assert op.params == {"gain": 2.0}

    def test_params_assignment_requires_mapping(self) -> None:
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000)

        with pytest.raises(TypeError, match="Operation params must be a mapping"):
            op.params = [("gain", 2.0)]

    def test_params_assignment_before_base_init_stores_snapshot(self) -> None:
        op = object.__new__(self._make_test_op_class())

        op.params = {"gain": 2.0}

        assert op.params == {"gain": 2.0}

    def test_post_base_init_mutable_config_assignment_is_snapshotted(self) -> None:
        class PostInitConfigOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "post_init_config_op"

            def __init__(self, sampling_rate: float, config: dict[str, float]):
                super().__init__(sampling_rate, config=config)
                self.config = config

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x * object.__getattribute__(self, "config")["gain"]

        config = {"gain": 2.0}
        op = PostInitConfigOperation(16000, config=config)
        data = da_from_array(np.array([[1.0, 2.0]]), chunks=(1, -1))
        config["gain"] = 99.0

        assert op.config["gain"] == 2.0
        np.testing.assert_array_equal(op.process(data).compute(), np.array([[2.0, 4.0]]))

    def test_post_base_init_non_container_config_assignment_is_snapshotted(self) -> None:
        class NamespaceConfigOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "namespace_config_op"

            def __init__(self, sampling_rate: float, cfg: SimpleNamespace):
                super().__init__(sampling_rate, cfg=cfg)
                self.cfg = cfg

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x * self.cfg.gain

        cfg = SimpleNamespace(gain=2.0)
        op = NamespaceConfigOperation(16000, cfg=cfg)
        data = da_from_array(np.array([[1.0, 2.0]]), chunks=(1, -1))
        cfg.gain = 99.0

        assert op.cfg.gain == 2.0
        np.testing.assert_array_equal(op.process(data).compute(), np.array([[2.0, 4.0]]))

    def test_pre_base_init_mutable_config_attribute_is_snapshotted(self) -> None:
        class PreInitConfigOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "pre_init_config_op"

            def __init__(self, sampling_rate: float, config: dict[str, float]):
                self.config = config
                super().__init__(sampling_rate, config=config)

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x * object.__getattribute__(self, "config")["gain"]

        config = {"gain": 2.0}
        op = PreInitConfigOperation(16000, config=config)
        config["gain"] = 99.0

        assert object.__getattribute__(op, "config") == {"gain": 2.0}

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

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
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

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                self.cache["gain"] = 2.0
                return x

        op = CachedOperation(16000, gain=2.0)

        op.cache["gain"] = 3.0

        assert object.__getattribute__(op, "cache") == {"gain": 3.0}

    def test_operation_params_direct_assignment_updates_captured_params(self) -> None:
        op = HighPassFilter(16000, cutoff=500)

        params = op.params
        params["cutoff"] = 1000

        assert op.params["cutoff"] == 1000
        assert op.cutoff == 500

    def test_operation_params_view_exposes_mapping_helpers(self) -> None:
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000, gain=2.0)
        params = op.params

        assert len(params) == 1
        assert repr(params) == "{'gain': 2.0}"
        assert params.copy() == {"gain": 2.0}
        assert params != [("gain", 2.0)]

        del params["gain"]

        assert op.params == {}

    def test_operation_params_nested_values_remain_defensive(self) -> None:
        class NestedParamsOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "nested_params_op"

            def __init__(self, sampling_rate: float, config: dict[str, float]):
                super().__init__(sampling_rate, config=config)

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x

        op = NestedParamsOperation(16000, config={"gain": 2.0})

        copied_params = op.params.copy()
        copied_params["config"]["gain"] = 99.0
        op.params["config"]["gain"] = 99.0

        assert op.params["config"]["gain"] == 2.0

    def test_operation_params_deletion_does_not_remove_public_config_guard(self) -> None:
        op = HighPassFilter(16000, cutoff=500)

        del op.params["cutoff"]
        op.cutoff = 1000

        assert "cutoff" not in op.params
        assert op.cutoff == 500

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

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x * self.config["missing"]

        op = DefaultdictConfigOperation(16000, defaultdict(lambda: 2.0))
        data = da_from_array(np.array([[1.0, 2.0]]), chunks=(1, -1))

        assert isinstance(object.__getattribute__(op, "config"), defaultdict)
        np.testing.assert_array_equal(op.process(data).compute(), np.array([[2.0, 4.0]]))

    def test_operation_params_equality_handles_array_values(self) -> None:
        class ArrayParamsOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "array_params_op"

            def __init__(self, sampling_rate: float, reference: NDArrayReal):
                super().__init__(sampling_rate, reference=reference, config={"weights": reference})

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x

        reference = np.array([1.0, 2.0])
        op = ArrayParamsOperation(16000, reference=reference)

        assert op.params == op.params
        assert op.params == {"reference": np.array([1.0, 2.0]), "config": {"weights": np.array([1.0, 2.0])}}
        assert op.params != {"reference": np.array([1.0, 3.0]), "config": {"weights": np.array([1.0, 2.0])}}

    def test_snapshot_config_value_preserves_dask_arrays(self) -> None:
        dask_array = da_from_array(np.array([1.0]), chunks=(1,))

        assert _snapshot_config_value(dask_array) is dask_array

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

    def test_snapshot_config_value_falls_back_when_deepcopy_fails(self) -> None:
        class NonCopyable:
            def __deepcopy__(self, memo: dict[int, object]) -> object:
                raise RuntimeError("no copy")

        value = NonCopyable()

        assert _snapshot_config_value(value) is value

    def test_public_config_access_handles_missing_params_during_partial_initialization(self) -> None:
        test_op_cls = self._make_test_op_class()
        op = object.__new__(test_op_cls)
        config = {"gain": 2.0}
        object.__setattr__(op, "_wandas_initialized", True)
        object.__setattr__(op, "config", config)

        assert op.config is config

    def test_public_config_access_handles_missing_captured_attrs_during_partial_initialization(self) -> None:
        test_op_cls = self._make_test_op_class()
        op = object.__new__(test_op_cls)
        object.__setattr__(op, "_wandas_initialized", True)
        object.__setattr__(op, "_params", {"config": {"gain": 2.0}})
        object.__setattr__(op, "config", {"gain": 2.0})

        exposed = op.config
        exposed["gain"] = 99.0

        assert object.__getattribute__(op, "config") == {"gain": 2.0}

    def test_public_config_assignment_handles_missing_params_during_partial_initialization(self) -> None:
        test_op_cls = self._make_test_op_class()
        op = object.__new__(test_op_cls)
        config = {"gain": 2.0}
        object.__setattr__(op, "_wandas_initialized", True)

        op.config = config

        assert object.__getattribute__(op, "config") is config

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
