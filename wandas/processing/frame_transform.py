from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray
from dask.delayed import delayed

from wandas.processing.base import AudioOperation

logger = logging.getLogger(__name__)


class FrameTransformOperation(AudioOperation[Any, Any]):
    """Operation wrapper for user-defined transforms used by Frame.transform.

    Notes
    -----
    - Supports optional output shape inference by running the function once on a
      small NumPy test array ("dry-run"). This never touches the real Dask data.
    - Supports explicit output dtype to handle transforms that change dtype
      (e.g., real -> complex for FFT-like operations).
    """

    name = "frame_transform"

    def __init__(
        self,
        sampling_rate: float,
        *,
        func: Callable[..., Any],
        output_shape_func: Callable[[tuple[int, ...]], tuple[int, ...]] | None = None,
        infer_output_shape: bool = True,
        infer_input_shape: tuple[int, ...] | None = None,
        output_dtype: Any | None = None,
        pure: bool = True,
        **params: Any,
    ) -> None:
        self.func = func
        self.output_shape_func = output_shape_func
        self.infer_output_shape = infer_output_shape
        self.infer_input_shape = infer_input_shape
        self.output_dtype = output_dtype
        super().__init__(sampling_rate, pure=pure, **params)

    def _process_array(self, x: Any) -> Any:
        return self.func(x, **self.params)

    def _infer_from_dry_run(
        self, input_shape: tuple[int, ...]
    ) -> tuple[tuple[int, ...], Any]:
        test_shape = self.infer_input_shape
        if test_shape is None:
            if len(input_shape) == 0:
                test_shape = input_shape
            else:
                # Keep the channel axis if present; shrink the remaining axes.
                ch = int(input_shape[0]) if len(input_shape) >= 1 else 1
                ch = 1 if ch <= 0 else min(ch, 1)
                rest: list[int] = []
                for i, s in enumerate(input_shape[1:], start=1):
                    if isinstance(s, (int, np.integer)):
                        si = int(s)
                    else:
                        raise ValueError(
                            f"Invalid dimension in input_shape at index {i}: {s!r}\n"
                            f"  Expected an integer for shape inference, got {type(s).__name__}\n"
                            "  Provide a valid input_shape or set infer_output_shape explicitly."
                        )
                    rest.append(min(max(si, 1), 64))
                test_shape = (ch, *rest)

        # Use a real dtype for the probe to support real-only ops (e.g. rfft).
        # The actual output dtype can still be forced via output_dtype.
        test_input = np.zeros(test_shape, dtype=np.float64)
        try:
            test_output = self._process_array(test_input)
        except Exception as e:
            func_name = getattr(self.func, "__name__", None)
            if not isinstance(func_name, str):
                func_name = "<callable>"
            raise RuntimeError(
                "Failed to infer output shape/dtype via dry-run\n"
                f"  What happened: user transform '{func_name}' raised "
                f"{type(e).__name__}: {e}\n"
                "  Why this matters: transform() needs output shape/dtype "
                "to build a lazy Dask graph\n"
                "  How to fix:\n"
                "    - Provide output_shape_func=... (recommended)\n"
                "    - Or: infer_output_shape=False + output_shape_func\n"
                "    - Or pass infer_input_shape=(...) to match what your "
                "function expects\n"
                "    - If dtype changes (e.g., FFT), pass output_dtype=...\n"
                "  Note: dry-run uses a small real NumPy array; "
                "functions with side effects are not suitable."
            ) from e
        if not isinstance(test_output, np.ndarray):
            raise TypeError(
                "Dry-run output must be a NumPy ndarray\n"
                f"  Got: {type(test_output).__name__}\n"
                "Provide output_shape_func or return a NumPy ndarray in dry-run."
            )
        return tuple(int(s) for s in test_output.shape), test_output.dtype

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if self.output_shape_func is not None:
            return self.output_shape_func(input_shape)

        if not self.infer_output_shape:
            raise ValueError(
                "Cannot determine output shape\n"
                "  output_shape_func was not provided and infer_output_shape=False\n"
                "Provide output_shape_func or set infer_output_shape=True."
            )

        out_shape, _ = self._infer_from_dry_run(input_shape)
        logger.warning(
            "Frame transform inferred output shape via dry-run. "
            "For performance and safety, provide output_shape_func."
        )
        return out_shape

    def _infer_output_dtype(self, input_shape: tuple[int, ...]) -> Any:
        if self.output_dtype is not None:
            return self.output_dtype

        if self.output_shape_func is not None and not self.infer_output_shape:
            # No inference available; fall back to input dtype at runtime.
            return None

        if self.infer_output_shape:
            _, out_dtype = self._infer_from_dry_run(input_shape)
            return out_dtype
        return None

    def process(self, data: DaArray) -> DaArray:
        wrapper = self._create_named_wrapper()
        delayed_func = delayed(wrapper, pure=self.pure)
        delayed_result = delayed_func(data)

        out_shape = self.calculate_output_shape(data.shape)

        out_dtype = self._infer_output_dtype(data.shape)
        if out_dtype is None:
            out_dtype = data.dtype

        return da.from_delayed(delayed_result, shape=out_shape, dtype=out_dtype)
