import dask.array as da
import numpy as np

from wandas.processing.base import _OPERATION_REGISTRY, create_operation, get_operation
from wandas.processing.custom import CustomOperation

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestCustomOperation:
    def test_custom_operation_applies_func_and_params(self) -> None:
        data = np.array([[1.0, 2.0, 3.0]])
        dask_data = _da_from_array(data, chunks=(1, -1))

        def scale_add(x: np.ndarray, gain: float) -> np.ndarray:
            return x * gain + 1.0

        op = CustomOperation(16000, func=scale_add, gain=2.0)

        result = op.process(dask_data).compute()
        np.testing.assert_array_equal(result, scale_add(data, gain=2.0))

    def test_custom_operation_output_shape_func_overrides(self) -> None:
        data = np.arange(8.0).reshape(1, 8)
        dask_data = _da_from_array(data, chunks=(1, -1))

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
        assert processed.shape == (1, 4)
        np.testing.assert_array_equal(processed.compute(), halve_samples(data))

    def test_get_display_name_uses_func_name(self) -> None:
        def my_func(x: np.ndarray) -> np.ndarray:
            return x

        op = CustomOperation(16000, func=my_func)
        assert op.get_display_name() == "my_func"

    def test_get_display_name_fallback_to_custom(self) -> None:
        class CallableObj:
            def __call__(self, x: np.ndarray) -> np.ndarray:
                return x

        op = CustomOperation(16000, func=CallableObj())
        assert op.get_display_name() == "custom"

    def test_custom_operation_registered(self) -> None:
        # Ensure registration side effect ran on import
        assert _OPERATION_REGISTRY.get("custom") is CustomOperation

        created = create_operation("custom", 16000, func=lambda x: x)
        assert isinstance(created, CustomOperation)
        assert get_operation("custom") is CustomOperation
