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
        **params : Any
            Additional parameters to pass to the function.
        """
        # Check for conflicting parameter names
        if "sampling_rate" in params:
            raise ValueError(
                "Parameter name conflict\n"
                "  Cannot use 'sampling_rate' as a parameter name in "
                "custom functions.\n"
                "  The sampling rate is automatically provided by the "
                "ChannelFrame.\n"
                "  Please use a different parameter name "
                "(e.g., 'sr', 'sample_rate', 'fs').\n"
                f"  Received params: {list(params.keys())}"
            )

        # Annotate the instance attribute so mypy knows the callable's return
        # type corresponds to OutputArrayType.
        self.func: Callable[..., OutputArrayType] = func
        self.output_shape_func = output_shape_func
        super().__init__(sampling_rate, **params)

    def _process_array(self, x: InputArrayType) -> OutputArrayType:
        """Apply custom function."""
        return self.func(x, **self.params)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape."""
        if self.output_shape_func:
            return self.output_shape_func(input_shape)
        return super().calculate_output_shape(input_shape)

    def get_display_name(self) -> str:
        """Get display name for the operation."""
        name = getattr(self.func, "__name__", None)
        if isinstance(name, str):
            return name
        return "custom"


register_operation(CustomOperation)
