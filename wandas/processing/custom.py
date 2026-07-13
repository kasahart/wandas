import importlib
import inspect
import math
import numbers
from collections.abc import Callable
from typing import Any

from wandas.processing.base import (
    AudioOperation,
    InputArrayType,
    OutputArrayType,
    register_operation,
)
from wandas.processing.semantic import CustomReplay, OperationContract, frozen_params


def _callable_reference(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", type(func).__module__)
    qualname = getattr(func, "__qualname__", type(func).__qualname__)
    return f"{module}.{qualname}"


def _importable_function_path(func: Callable[..., Any] | None) -> str | None:
    """Return a stable import path for module-level functions."""
    if func is None or not inspect.isfunction(func):
        return None
    if func.__module__ == "__main__" or func.__name__ == "<lambda>":
        return None
    if func.__qualname__ != func.__name__ or func.__closure__ is not None:
        return None
    try:
        module = importlib.import_module(func.__module__)
    except Exception:
        return None
    if getattr(module, func.__name__, None) is not func:
        return None
    return f"{func.__module__}.{func.__name__}"


def _is_recipe_literal(value: Any) -> bool:
    if value is None or isinstance(value, bool | str):
        return True
    if type(value).__module__ == "numpy" and type(value).__name__ in {"bool", "bool_"}:
        return True
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return True
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return not math.isnan(float(value))
    if isinstance(value, list | tuple):
        return all(not isinstance(item, list | tuple | dict) and _is_recipe_literal(item) for item in value)
    return False


def _is_recipe_literal_mapping(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str) and _is_recipe_literal(item) for key, item in value.items()
    )


class CustomOperation(AudioOperation[InputArrayType, OutputArrayType]):
    """Custom operation defined by a user function."""

    name = "custom"

    def __init__(
        self,
        sampling_rate: float,
        func: Callable[..., OutputArrayType],
        output_shape_func: Callable[[tuple[int, ...]], tuple[int, ...]] | None = None,
        *,
        dask_pure: bool = True,
        **params: Any,
    ):
        """
        Initialize CustomOperation.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        func : Callable
            Function to apply to the data.
        output_shape_func : Callable, optional
            Function to calculate output shape from input shape.
        dask_pure : bool, default=True
            Dask execution-control flag for the delayed task. Set to ``False``
            for non-deterministic or side-effecting custom functions. This is
            not forwarded to the custom function or recorded in lineage params.
        **params : Any
            Additional parameters to pass to the function. ``pure`` and
            ``dask_pure`` are reserved for operation purity and Dask task
            semantics; use different custom parameter names such as
            ``is_pure``.
        """
        # Store callables privately so a frame lineage operation cannot alter a
        # pending Dask graph by reassigning public attributes before compute.
        self._func: Callable[..., OutputArrayType] = func
        self._output_shape_func = output_shape_func
        super().__init__(sampling_rate, pure=dask_pure, **params)

    @property
    def func(self) -> Callable[..., OutputArrayType]:
        """Function captured at operation construction time."""
        return self._func

    @property
    def output_shape_func(self) -> Callable[[tuple[int, ...]], tuple[int, ...]] | None:
        """Output shape function captured at operation construction time."""
        return self._output_shape_func

    def _process(self, x: InputArrayType) -> OutputArrayType:
        """Apply custom function."""
        return self._func(x, **self._config_snapshot())

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape."""
        if self._output_shape_func is not None:
            return self._output_shape_func(input_shape)
        return super().calculate_output_shape(input_shape)

    def get_display_name(self) -> str:
        """Get display name for the operation."""
        name = getattr(self._func, "__name__", None)
        if isinstance(name, str):
            return name
        return "custom"

    def to_summary(self) -> dict[str, Any]:
        """Return a lightweight display summary for this custom callable."""
        summary = super().to_summary()
        summary["implementation"] = _callable_reference(self.func)
        return summary

    def to_recipe_metadata(self) -> dict[str, Any] | None:
        """Return custom recipe metadata when the callables are importable."""
        function_path = _importable_function_path(self._func)
        if function_path is None:
            return None
        output_shape_function_path = _importable_function_path(self._output_shape_func)
        if self._output_shape_func is not None and output_shape_function_path is None:
            return None
        output_frame_class = getattr(self, "_recipe_output_frame_class", None)
        metadata = {
            "function": function_path,
            "output_shape_function": output_shape_function_path,
            "dask_pure": bool(self.pure),
            "output_frame_class": output_frame_class,
        }
        if output_frame_class is not None:
            output_frame_kwargs = getattr(self, "_recipe_output_frame_kwargs", {})
            if not _is_recipe_literal_mapping(output_frame_kwargs):
                return None
            metadata["output_frame_kwargs"] = output_frame_kwargs
        return metadata

    def replay_descriptor(self) -> CustomReplay:
        metadata = self.to_recipe_metadata()
        kwargs = {} if metadata is None else metadata.get("output_frame_kwargs", {})
        return CustomReplay(
            OperationContract(
                self.name,
                self.operation_version,
                bool(self.pure),
                super().replay_descriptor().contract.bindings,
            ),
            frozen_params(self.to_params(), allow_opaque=metadata is None),
            self.name,
            None if metadata is None else metadata["function"],
            None if metadata is None else metadata["output_shape_function"],
            None if metadata is None else metadata["output_frame_class"],
            frozen_params(kwargs, allow_opaque=metadata is None),
        )


register_operation(CustomOperation)
