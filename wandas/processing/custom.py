from collections.abc import Callable
from typing import Any

from wandas.processing.base import (
    AudioOperation,
    InputArrayType,
    OutputArrayType,
    register_operation,
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

    def _process_array(self, x: InputArrayType) -> OutputArrayType:
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


register_operation(CustomOperation)
