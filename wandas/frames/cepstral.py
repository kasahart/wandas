from __future__ import annotations

import logging
import numbers
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from dask.array.core import Array as DaArray

from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelMetadata
from wandas.frames.spectral import SpectralFrame
from wandas.utils import validate_sampling_rate
from wandas.utils.optional_imports import require_pandas
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


class CepstralFrame(BaseFrame[NDArrayReal]):
    """Frame for real cepstrum data with quefrency-aware metadata."""

    _xarray_dim_suffix = ("channel", "quefrency")

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        n_fft: int,
        window: str = "hann",
        analysis_length: int | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        channel_metadata: Sequence[ChannelMetadata | dict[str, Any]] | None = None,
        channel_ids: list[str] | None = None,
        previous: BaseFrame[Any] | None = None,
        source_time_offset: float | Sequence[float] | NDArrayReal = 0.0,
        lineage: Any | None = None,
        operation_summaries_snapshot: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim != 2:
            raise ValueError(
                f"CepstralFrame data must be 1D or 2D\n"
                f"  Got: shape {data.shape} ({data.ndim}D)\n"
                "Reshape higher-dimensional data to (channels, quefrency_bins)."
            )
        if np.issubdtype(data.dtype, np.complexfloating):
            raise TypeError("CepstralFrame requires real-valued coefficients, not complex data.")
        if isinstance(n_fft, bool) or not isinstance(n_fft, numbers.Integral):
            raise TypeError("n_fft must be a positive integer.")
        self._n_fft = int(n_fft)
        if self.n_fft <= 0:
            raise ValueError("n_fft must be a positive integer.")
        if int(data.shape[-1]) > self.n_fft:
            raise ValueError("CepstralFrame cannot contain more quefrency bins than n_fft.")
        self._window = window
        if analysis_length is not None and (
            isinstance(analysis_length, bool) or not isinstance(analysis_length, numbers.Integral)
        ):
            raise TypeError("analysis_length must be a positive integer or None.")
        self._analysis_length = self.n_fft if analysis_length is None else int(analysis_length)
        if self.analysis_length <= 0 or self.analysis_length > self.n_fft:
            raise ValueError("analysis_length must be positive and no greater than n_fft.")
        self._pending_sampling_rate = float(sampling_rate)
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            metadata=metadata,
            channel_metadata=channel_metadata,
            channel_ids=channel_ids,
            source_time_offset=source_time_offset,
            lineage=lineage,
            operation_summaries_snapshot=operation_summaries_snapshot,
            previous=previous,
        )
        del self._pending_sampling_rate

    @property
    def quefrencies(self) -> NDArrayReal:
        """Return the quefrency axis in seconds."""
        return np.asarray(self._xr.coords["quefrency"].values, dtype=float).copy()

    @property
    def n_fft(self) -> int:
        """Return the immutable period of the complete cepstrum."""
        return self._n_fft

    @property
    def window(self) -> str:
        """Return the immutable analysis window."""
        return self._window

    @property
    def analysis_length(self) -> int:
        """Return the immutable pre-padding analysis window length."""
        return self._analysis_length

    @property
    def sampling_rate(self) -> float:
        """Return the immutable sampling rate defining the quefrency axis."""
        return float(self._xr.attrs["sampling_rate"])

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        validate_sampling_rate(value)
        current = self._xr.attrs.get("sampling_rate")
        if current is not None and float(value) != float(current):
            raise AttributeError("CepstralFrame sampling_rate is immutable because it defines the quefrency axis.")
        self._xr.attrs["sampling_rate"] = float(value)

    def _xarray_coords(self, data: DaArray) -> dict[str, Any]:
        coords = super()._xarray_coords(data)
        if "quefrency" in self._xarray_dims(data):
            sampling_rate = (
                self._pending_sampling_rate if hasattr(self, "_pending_sampling_rate") else self.sampling_rate
            )
            coords["quefrency"] = np.arange(int(data.shape[-1]), dtype=float) / sampling_rate
        return coords

    def __getitem__(self, key: Any) -> CepstralFrame:
        """Slice data and its quefrency coordinates atomically."""
        normalized_key = key
        if (
            isinstance(key, tuple)
            and len(key) > 1
            and isinstance(key[1], numbers.Integral)
            and not isinstance(key[1], bool)
        ):
            raw_index = int(key[1])
            if raw_index < -len(self.quefrencies) or raw_index >= len(self.quefrencies):
                raise IndexError(f"Quefrency index out of range: {raw_index}")
            index = raw_index % len(self.quefrencies)
            normalized_key = (key[0], slice(index, index + 1), *key[2:])
        result = super().__getitem__(normalized_key)
        quefrency_key = slice(None)
        if isinstance(normalized_key, tuple) and len(normalized_key) > 1:
            quefrency_key = normalized_key[1]
        selected_quefrencies = np.atleast_1d(self.quefrencies[quefrency_key]).astype(float, copy=True)
        if "quefrency" in result._xr.dims:
            result._xr = result._xr.assign_coords(quefrency=selected_quefrencies.copy())
        return result

    def _create_new_instance(self, data: DaArray, **kwargs: Any) -> CepstralFrame:
        """Preserve quefrency coordinates for shape-preserving derivations."""
        result = super()._create_new_instance(data=data, **kwargs)
        if int(result._data.shape[-1]) == len(self.quefrencies):
            if "quefrency" in result._xr.dims:
                result._xr = result._xr.assign_coords(quefrency=self.quefrencies)
        return result

    def _binary_operand_op(self, other: Any, op: Any, symbol: str, *, reverse: bool = False) -> CepstralFrame:
        if isinstance(other, BaseFrame) and not isinstance(other, CepstralFrame):
            raise TypeError("CepstralFrame operations require another CepstralFrame or a real scalar.")
        if isinstance(other, CepstralFrame):
            if self.n_fft != other.n_fft:
                raise ValueError(f"Cepstral n_fft mismatch: {self.n_fft} != {other.n_fft}")
            if self.window != other.window or self.analysis_length != other.analysis_length:
                raise ValueError(
                    "Cepstral analysis metadata must match exactly: "
                    f"(window={self.window!r}, analysis_length={self.analysis_length}) != "
                    f"(window={other.window!r}, analysis_length={other.analysis_length})."
                )
            if not np.array_equal(self.quefrencies, other.quefrencies):
                raise ValueError("Cepstral quefrency coordinates must match exactly.")
        elif isinstance(other, bool) or not isinstance(other, numbers.Real):
            raise TypeError("CepstralFrame operations require another CepstralFrame or a real-valued scalar operand.")
        return cast(
            "CepstralFrame",
            super()._binary_operand_op(other, op, symbol, reverse=reverse),
        )

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """Allow NumPy dispatch only for the same real-scalar arithmetic contract."""
        if method != "__call__" or len(inputs) != 2 or not any(value is self for value in inputs) or kwargs:
            raise TypeError("CepstralFrame does not support NumPy ufunc coercion.")
        other = inputs[0] if inputs[1] is self else inputs[1]
        if isinstance(other, bool) or not isinstance(other, numbers.Real):
            raise TypeError("CepstralFrame operations require another CepstralFrame or a real-valued scalar operand.")
        direct_methods = {
            "add": "__add__",
            "subtract": "__sub__",
            "multiply": "__mul__",
            "divide": "__truediv__",
            "true_divide": "__truediv__",
            "power": "__pow__",
        }
        reverse_methods = {
            "add": "__radd__",
            "subtract": "__rsub__",
            "multiply": "__rmul__",
            "divide": "__rtruediv__",
            "true_divide": "__rtruediv__",
            "power": "__rpow__",
        }
        method_name = (direct_methods if inputs[0] is self else reverse_methods).get(ufunc.__name__)
        if method_name is None:
            raise TypeError(f"CepstralFrame does not support NumPy ufunc {ufunc.__name__!r}.")
        return getattr(self, method_name)(other)

    def apply_operation(self, operation_name: str, **params: Any) -> CepstralFrame:
        """Reject generic operations so domain transitions stay on typed APIs."""
        raise ValueError(
            f"Operation {operation_name!r} is not available through CepstralFrame.apply_operation(). "
            "Use lifter() or to_spectral_envelope() for cepstral-domain processing."
        )

    def lifter(self, cutoff: float, mode: str = "low") -> CepstralFrame:
        """Apply low-pass or high-pass liftering in the quefrency domain."""
        self._require_complete_quefrency_axis("lifter")
        from wandas.processing import create_operation

        operation = create_operation("lifter", self.sampling_rate, cutoff=cutoff, mode=mode)
        lineage = self._lineage_with_method("lifter", operation.to_params())
        return self._create_new_instance(
            data=operation.process(self._data),
            lineage=lineage,
            **self._operation_summaries_snapshot_kwargs(lineage),
        )

    def to_spectral_envelope(self) -> SpectralFrame:
        """Convert the cepstrum back to a smooth spectral envelope."""
        self._require_complete_quefrency_axis("to_spectral_envelope")
        from wandas.processing import SpectralEnvelope, create_operation

        operation_name = "spectral_envelope"
        operation = create_operation(
            operation_name,
            self.sampling_rate,
            window=self.window,
            window_length=self.analysis_length,
        )
        operation = cast("SpectralEnvelope", operation)
        envelope_data = operation.process(self._data)
        lineage = self._lineage_with_method(operation_name, operation.to_params())

        return SpectralFrame(
            data=envelope_data,
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            window=self.window,
            label=f"spectral_envelope({self.label})",
            metadata=self.metadata,
            channel_metadata=self.channels.to_list(),
            channel_ids=self._channel_ids,
            source_time_offset=self.source_time_offset,
            lineage=lineage,
            previous=self,
            **self._operation_summaries_snapshot_kwargs(lineage),
        )

    def _require_complete_quefrency_axis(self, operation_name: str) -> None:
        expected = np.arange(self.n_fft, dtype=float) / self.sampling_rate
        if self.shape[-1] != self.n_fft or not np.array_equal(self.quefrencies, expected):
            raise ValueError(
                f"{operation_name} requires a complete, unsliced quefrency axis. "
                "Apply the operation before slicing the CepstralFrame."
            )

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """Provide constructor kwargs needed to recreate a CepstralFrame."""
        return {
            "n_fft": self.n_fft,
            "window": self.window,
            "analysis_length": self.analysis_length,
        }

    def _get_dataframe_index(self) -> Any:
        """Get quefrency index for DataFrame conversion."""
        pd = require_pandas("CepstralFrame.to_dataframe")
        return pd.Index(self.quefrencies, name="quefrency")

    def plot(
        self,
        plot_type: str = "raw",
        ax: Axes | None = None,
        title: str | None = None,
        overlay: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        alpha: float = 1.0,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Plot cepstral data using quefrency-aware axis labeling."""
        if plot_type != "raw":
            raise ValueError("CepstralFrame.plot supports only plot_type='raw'.")
        import matplotlib.pyplot as plt

        target = ax if ax is not None else plt.subplots()[1]
        values = np.asarray(self._data.compute())
        for index, label in enumerate(self.labels):
            target.plot(self.quefrencies, values[index], label=label, alpha=alpha, **kwargs)
        if overlay or self.n_channels > 1:
            target.legend()
        target.set(
            xlabel=xlabel or "Quefrency [s]",
            ylabel=ylabel or "Cepstrum",
            title=title or self.label,
            xlim=xlim,
            ylim=ylim,
        )
        return target

    def info(self) -> None:
        """Display cepstral-domain information."""
        quefrency_resolution = 1.0 / self.sampling_rate

        print("CepstralFrame Information:")
        print(f"  Channels: {self.n_channels}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  FFT size: {self.n_fft}")
        print(f"  Quefrency range: {self.quefrencies[0]:.6f} - {self.quefrencies[-1]:.6f} s")
        print(f"  Quefrency bins: {len(self.quefrencies)}")
        print(f"  Quefrency resolution (delta_Q): {quefrency_resolution:.6f} s")
        print(f"  Channel labels: {self.labels}")
        self._print_operation_history()
