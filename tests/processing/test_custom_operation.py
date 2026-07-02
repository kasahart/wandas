import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.processing.base import _OPERATION_REGISTRY, create_operation, get_operation
from wandas.processing.custom import CustomOperation
from wandas.utils.dask_helpers import da_from_array


class TestCustomOperation:
    def test_custom_operation_applies_func_and_params(self) -> None:
        """CustomOperation applies user function with kwargs to input array."""
        data = np.array([[1.0, 2.0, 3.0]])
        dask_data = da_from_array(data, chunks=(1, -1))

        def scale_add(x: np.ndarray, gain: float) -> np.ndarray:
            return x * gain + 1.0

        op = CustomOperation(16000, func=scale_add, gain=2.0)

        result_da = op.process(dask_data)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        np.testing.assert_array_equal(result, scale_add(data, gain=2.0))

    def test_custom_operation_process_uses_operation_owned_params_copy(self) -> None:
        data = np.array([[1.0, 2.0, 3.0]])
        op = CustomOperation(16000, func=lambda x, gain: x * gain, gain=2.0)

        np.testing.assert_array_equal(op._process(data), data * 2.0)

    def test_custom_operation_copies_nested_params_for_each_delayed_execution(self) -> None:
        data = np.array([[1.0, 2.0, 3.0]])
        dask_data = da_from_array(data, chunks=(1, -1))

        def mutating_scale(x: np.ndarray, config: dict[str, float]) -> np.ndarray:
            gain = config["gain"]
            config["gain"] = 99.0
            return x * gain

        op = CustomOperation(16000, func=mutating_scale, config={"gain": 2.0})
        result = op.process(dask_data)

        np.testing.assert_array_equal(result.compute(), data * 2.0)
        np.testing.assert_array_equal(result.compute(), data * 2.0)
        assert op.params["config"] == {"gain": 2.0}

    def test_custom_operation_subclass_process_contract_uses_process_hook(self) -> None:
        class HookedCustomOperation(CustomOperation):
            def _process(self, x: np.ndarray) -> np.ndarray:
                return super()._process(x) * 3.0

        data = np.array([[1.0, 2.0, 3.0]])
        dask_data = da_from_array(data, chunks=(1, -1))
        op = HookedCustomOperation(16000, func=lambda x, gain: x * gain, gain=2.0)

        np.testing.assert_array_equal(op.process(dask_data).compute(), data * 6.0)

    def test_custom_operation_output_shape_func_overrides(self) -> None:
        """output_shape_func overrides default shape inference for Dask graph."""
        data = np.arange(8.0).reshape(1, 8)
        dask_data = da_from_array(data, chunks=(1, -1))

        def halve_samples(x: np.ndarray) -> np.ndarray:
            return x[:, ::2]

        def output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
            return (input_shape[0], input_shape[1] // 2)

        op = CustomOperation(
            16000,
            func=halve_samples,
            output_shape_func=output_shape,
        )

        processed = op.process(dask_data)
        assert op.func is halve_samples
        assert op.output_shape_func is output_shape
        np.testing.assert_array_equal(processed.compute(), halve_samples(data))

    def test_get_display_name_uses_func_name(self) -> None:
        """get_display_name() returns the function's __name__ attribute."""

        def my_func(x: np.ndarray) -> np.ndarray:
            return x

        op = CustomOperation(16000, func=my_func)
        assert op.get_display_name() == "my_func"

    def test_get_display_name_fallback_to_custom(self) -> None:
        """get_display_name() falls back to 'custom' for callable objects."""

        class CallableObj:
            def __call__(self, x: np.ndarray) -> np.ndarray:
                return x

        op = CustomOperation(16000, func=CallableObj())
        assert op.get_display_name() == "custom"

    def test_custom_operation_registered(self) -> None:
        """CustomOperation is registered in the operation registry."""
        # Ensure registration side effect ran on import
        assert _OPERATION_REGISTRY.get("custom") is CustomOperation

        created = create_operation("custom", 16000, func=lambda x: x)
        assert isinstance(created, CustomOperation)
        assert get_operation("custom") is CustomOperation

    def test_custom_operation_direct_call_raises_type_error(self) -> None:
        """Direct CustomOperation call with sampling_rate in kwargs raises TypeError.

        Note: When calling CustomOperation directly with sampling_rate in kwargs,
        Python raises TypeError during argument binding, before __init__ body runs.
        This is expected behavior and not something we can prevent at the class level.
        The user-facing validation happens in ChannelFrame.apply(), which checks
        kwargs before passing them to CustomOperation.
        """

        def my_func(x: np.ndarray, sampling_rate: float) -> np.ndarray:
            return x * sampling_rate

        # Direct call raises TypeError due to Python's argument handling
        with pytest.raises(TypeError, match="multiple values for argument"):
            CustomOperation(16000, func=my_func, sampling_rate=44100)  # ty: ignore[parameter-already-assigned]

    def test_custom_operation_nested_params_are_defensive_snapshots(self) -> None:
        def my_func(x: np.ndarray, config: dict[str, float]) -> np.ndarray:
            return x * config["gain"]

        op = CustomOperation(16000, func=my_func, config={"gain": 2.0})

        params = op.params
        params["config"]["gain"] = 3.0

        assert op.params["config"]["gain"] == 2.0

    def test_custom_operation_defaults_to_dask_pure(self) -> None:
        op = CustomOperation(16000, func=lambda x: x)

        assert op.pure is True

    def test_custom_operation_dask_pure_controls_dask_purity_not_params(self) -> None:
        op = CustomOperation(16000, func=lambda x, gain: x * gain, dask_pure=False, gain=2.0)

        assert op.pure is False
        assert op.params == {"gain": 2.0}
        assert "dask_pure" not in op.to_params()

    def test_custom_operation_summary_includes_callable_reference_for_display(self) -> None:
        def scale(x: np.ndarray, gain: float) -> np.ndarray:
            return x * gain

        operation = CustomOperation(16000, func=scale, gain=2.0)

        summary = operation.to_summary()

        assert summary["operation"] == "custom"
        assert summary["params"] == {"gain": 2.0}
        assert summary["implementation"] is not scale
        assert summary["implementation"].endswith(".scale")
        assert "portable" not in summary
        assert "schema_version" not in summary

    def test_custom_operation_does_not_expose_pure_constructor_option(self) -> None:
        def my_func(x: np.ndarray, pure: bool) -> np.ndarray:
            return x + (1.0 if pure else 2.0)

        with pytest.raises(TypeError, match="multiple values"):
            CustomOperation(16000, func=my_func, pure=False)

    def test_custom_operation_accepts_params_named_function_argument(self) -> None:
        data = np.array([[1.0, 2.0]])
        dask_data = da_from_array(data, chunks=(1, -1))

        def my_func(x: np.ndarray, params: float) -> np.ndarray:
            return x * params

        op = CustomOperation(16000, func=my_func, params=2.0)

        assert op.params["params"] == 2.0
        np.testing.assert_array_equal(op.process(dask_data).compute(), data * 2.0)

    def test_custom_operation_callables_are_read_only(self) -> None:
        def my_func(x: np.ndarray) -> np.ndarray:
            return x

        op = CustomOperation(16000, func=my_func)

        with pytest.raises(AttributeError):
            setattr(op, "func", lambda x: x * 2)

        data = np.array([[1.0, 2.0]])
        dask_data = da_from_array(data, chunks=(1, -1))
        np.testing.assert_array_equal(op.process(dask_data).compute(), data)

    def test_custom_operation_output_shape_callable_is_read_only(self) -> None:
        op = CustomOperation(16000, func=lambda x: x, output_shape_func=lambda shape: shape)

        with pytest.raises(AttributeError):
            setattr(op, "output_shape_func", lambda shape: (shape[0],))
