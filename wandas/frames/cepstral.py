"""Quefrency-domain frame for real-cepstrum analysis."""

from __future__ import annotations

import numbers
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from dask.array.core import Array as DaArray

from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelMetadata
from wandas.pipeline.decorators import recipe_operation
from wandas.processing.semantic import LineageNode
from wandas.utils import validate_sampling_rate
from wandas.utils.optional_imports import require_pandas
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from wandas.frames.spectral import SpectralFrame


class CepstralFrame(BaseFrame[NDArrayReal]):
    """Immutable, lazy real-cepstrum data on a quefrency axis.

    Data is rank two with dimensions ``(channel, quefrency)`` and a real dtype.
    ``n_fft`` is the circular period of the complete cepstrum; sliced frames keep
    that period so methods can reject incomplete axes. Quefrency is measured in
    seconds at spacing ``1 / sampling_rate``. The sampling rate is immutable
    because changing it would reinterpret the stored axis.

    ``lifter()`` preserves this frame family. ``to_spectral_envelope()`` returns a
    :class:`~wandas.frames.spectral.SpectralFrame`. Both operations remain Dask
    backed and preserve channel identity, metadata, source-time offsets, and
    semantic lineage.

    Parameters
    ----------
    data : dask.array.Array
        Real coefficients shaped ``(quefrency,)`` or
        ``(channels, quefrency)``.
    sampling_rate : float
        Sampling rate in Hz that defines the quefrency-bin spacing.
    n_fft : int
        Positive FFT size of the complete cepstrum. Sliced data may contain fewer
        bins but never more than this value.
    window : str, default="hann"
        Window used by the originating cepstrum analysis.
    label : str, optional
        Human-readable frame label.
    metadata : dict, optional
        User and recording metadata, copied on construction.
    channel_metadata : sequence, optional
        Metadata aligned with the channel axis.
    channel_ids : list[str], optional
        Stable identifiers aligned with the channel axis.
    previous : BaseFrame, optional
        Compatibility/debug pointer to the immediate prior frame.
    source_time_offset : float or sequence, default=0.0
        Per-channel source timeline offsets, preserved across domain changes.
    lineage : LineageNode, optional
        Authoritative runtime semantic lineage.
    operation_history_prefix : sequence, default=()
        Persisted display history for a new source frame.

    Raises
    ------
    TypeError
        If coefficients are complex, ``n_fft`` is not integral, or ``window`` is
        not a non-empty string.
    ValueError
        If rank, FFT size, or coefficient count violates the domain contract.

    Examples
    --------
    >>> cepstrum = frame.cepstrum(n_fft=2048)
    >>> envelope = cepstrum.lifter(0.002).to_spectral_envelope()
    """

    _xarray_dim_suffix = ("channel", "quefrency")

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        n_fft: int,
        window: str = "hann",
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        channel_metadata: Sequence[ChannelMetadata | dict[str, Any]] | None = None,
        channel_ids: list[str] | None = None,
        previous: BaseFrame[Any] | None = None,
        source_time_offset: float | Sequence[float] | NDArrayReal = 0.0,
        lineage: LineageNode | None = None,
        operation_history_prefix: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim != 2:
            raise ValueError(
                "Invalid data shape for CepstralFrame\n"
                f"  Got: {data.shape} ({data.ndim}D)\n"
                "  Expected: (quefrency,) or (channels, quefrency)\n"
                "Reshape higher-dimensional data before construction."
            )
        if np.issubdtype(data.dtype, np.complexfloating):
            raise TypeError(
                "CepstralFrame requires real-valued coefficients\n"
                f"  Got: {data.dtype}\n"
                "  Expected: a real numeric dtype\n"
                "Construct it from ChannelFrame.cepstrum()."
            )
        if isinstance(n_fft, bool) or not isinstance(n_fft, numbers.Integral):
            raise TypeError(
                "Invalid n_fft for CepstralFrame\n"
                f"  Got: {type(n_fft).__name__}\n"
                "  Expected: a positive integer\n"
                "Pass the FFT size used to produce these coefficients."
            )
        normalized_n_fft = int(n_fft)
        if normalized_n_fft <= 0:
            raise ValueError(
                "Invalid n_fft for CepstralFrame\n"
                f"  Got: {normalized_n_fft}\n"
                "  Expected: a positive integer\n"
                "Pass the FFT size used to produce these coefficients."
            )
        if int(data.shape[-1]) > normalized_n_fft:
            raise ValueError(
                "CepstralFrame cannot contain more quefrency bins than n_fft\n"
                f"  Got: {data.shape[-1]} bins for n_fft={normalized_n_fft}\n"
                "  Expected: coefficient count <= n_fft\n"
                "Use the original FFT size or trim the coefficient data."
            )
        if not isinstance(window, str) or not window:
            raise TypeError("CepstralFrame window must be a non-empty string.")

        self._n_fft = normalized_n_fft
        self._window = window
        self._pending_sampling_rate = float(sampling_rate)
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            metadata=metadata,
            channel_metadata=channel_metadata,
            channel_ids=channel_ids,
            previous=previous,
            source_time_offset=source_time_offset,
            lineage=lineage,
            operation_history_prefix=operation_history_prefix,
        )
        del self._pending_sampling_rate

    @property
    def n_fft(self) -> int:
        """Return the immutable circular period of the complete cepstrum."""
        return self._n_fft

    @property
    def window(self) -> str:
        """Return the immutable originating analysis-window name."""
        return self._window

    @property
    def sampling_rate(self) -> float:
        """Return the immutable rate that defines the quefrency spacing."""
        return float(self._xr.attrs["sampling_rate"])

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        validate_sampling_rate(value)
        current = self._xr.attrs.get("sampling_rate")
        if current is not None and float(current) != float(value):
            raise AttributeError("CepstralFrame sampling_rate is immutable because it defines the quefrency axis.")
        self._xr.attrs["sampling_rate"] = float(value)

    @property
    def quefrencies(self) -> NDArrayReal:
        """Return a defensive copy of the represented quefrency bins in seconds."""
        return np.asarray(self._xr.coords["quefrency"].values, dtype=float).copy()

    def _xarray_coords(self, data: DaArray) -> dict[str, Any]:
        """Build channel and quefrency coordinates without computing data."""
        coords = super()._xarray_coords(data)
        if "quefrency" in self._xarray_dims(data):
            pending_sampling_rate = getattr(self, "_pending_sampling_rate", None)
            sampling_rate = self.sampling_rate if pending_sampling_rate is None else pending_sampling_rate
            coords["quefrency"] = (
                "quefrency",
                np.arange(int(data.shape[-1]), dtype=float) / sampling_rate,
            )
        return coords

    def _create_new_instance(self, data: DaArray, **kwargs: Any) -> CepstralFrame:
        """Recreate the frame and retain represented coordinates when shape permits."""
        result = cast("CepstralFrame", super()._create_new_instance(data=data, **kwargs))
        if int(result._data.shape[-1]) == len(self.quefrencies):
            result._xr = result._xr.assign_coords(quefrency=("quefrency", self.quefrencies))
        return result

    def _handle_multidim_indexing(self, key: tuple[Any, ...]) -> CepstralFrame:
        """Preserve selected quefrency coordinate values during public slicing."""
        result = cast("CepstralFrame", super()._handle_multidim_indexing(key))
        if len(key) > 1:
            selected = np.asarray(self.quefrencies[key[1]], dtype=float)
            result._xr = result._xr.assign_coords(quefrency=("quefrency", selected))
        return result

    def _binary_operand_op(
        self,
        other: Any,
        op: Callable[[Any, Any], Any],
        symbol: str,
        *,
        reverse: bool = False,
    ) -> CepstralFrame:
        """Require matching cepstral domain state for frame-frame arithmetic."""
        if isinstance(other, BaseFrame) and not isinstance(other, CepstralFrame):
            raise TypeError("CepstralFrame arithmetic requires another CepstralFrame or a scalar/array operand.")
        if isinstance(other, CepstralFrame):
            if self.n_fft != other.n_fft:
                raise ValueError(f"Cepstral n_fft mismatch: {self.n_fft} != {other.n_fft}")
            if self.window != other.window:
                raise ValueError(f"Cepstral analysis window mismatch: {self.window!r} != {other.window!r}")
            if not np.array_equal(self.quefrencies, other.quefrencies):
                raise ValueError("Cepstral quefrency coordinates must match exactly.")
        return cast(
            "CepstralFrame",
            super()._binary_operand_op(other, op, symbol, reverse=reverse),
        )

    @recipe_operation("wandas.cepstral.lifter")
    def lifter(
        self,
        cutoff: float,
        mode: Literal["low", "high"] = "low",
    ) -> CepstralFrame:
        """Keep low- or high-quefrency coefficients.

        Parameters
        ----------
        cutoff : float
            Positive quefrency boundary in seconds. It must reach at least one
            represented bin and remain below half the complete cepstrum.
        mode : {"low", "high"}, default="low"
            ``"low"`` keeps the smooth-envelope region; ``"high"`` keeps the
            complementary fine structure.

        Returns
        -------
        CepstralFrame
            A new lazy frame with the same axes and metadata.

        Raises
        ------
        ValueError
            If this frame has been sliced on the quefrency axis or the lifter
            parameters cannot be represented.

        Notes
        -----
        The method builds a Dask graph and does not compute coefficients.

        Examples
        --------
        >>> smooth = frame.cepstrum().lifter(0.002, mode="low")
        """
        self._require_complete_quefrency_axis("lifter")
        return cast(
            "CepstralFrame",
            self._apply_named_operation("lifter", cutoff=cutoff, mode=mode),
        )

    @recipe_operation("wandas.cepstral.to_spectral_envelope")
    def to_spectral_envelope(self) -> SpectralFrame:
        """Convert a complete real cepstrum to a smooth spectral envelope.

        Returns
        -------
        SpectralFrame
            New lazy complex-valued frequency data with zero phase, the original
            ``n_fft`` and window, and preserved metadata and source-time offsets.

        Raises
        ------
        ValueError
            If this frame has been sliced on the quefrency axis. Asymmetric
            concrete coefficients raise when the lazy result is computed.

        Notes
        -----
        This method builds a Dask graph. It does not compute the envelope.

        Examples
        --------
        >>> envelope = frame.cepstrum().lifter(0.002).to_spectral_envelope()
        """
        self._require_complete_quefrency_axis("to_spectral_envelope")
        from wandas.frames.spectral import SpectralFrame
        from wandas.processing import SpectralEnvelope, create_operation

        operation = cast(
            "SpectralEnvelope",
            create_operation("spectral_envelope", self.sampling_rate),
        )
        return SpectralFrame(
            data=operation.process(self._data),
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            window=self.window,
            label=f"Spectral envelope of {self.label}",
            metadata=self.metadata,
            channel_metadata=self.channels.to_list(),
            channel_ids=self._channel_ids,
            previous=self,
            source_time_offset=self.source_time_offset,
            lineage=self._required_semantic_lineage(),
        )

    def _require_complete_quefrency_axis(self, operation_name: str) -> None:
        """Reject transforms whose circular coefficient axis was sliced."""
        expected = np.arange(self.n_fft, dtype=float) / self.sampling_rate
        if int(self._data.shape[-1]) != self.n_fft or not np.array_equal(self.quefrencies, expected):
            raise ValueError(
                f"{operation_name} requires a complete, unsliced quefrency axis. "
                "Apply the operation before slicing the CepstralFrame."
            )

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """Return domain state required by ``_create_new_instance``."""
        return {"n_fft": self.n_fft, "window": self.window}

    def _get_dataframe_index(self) -> pd.Index[Any]:
        """Return the represented quefrency bins as the DataFrame index."""
        pd = require_pandas("CepstralFrame.to_dataframe")
        return pd.Index(self.quefrencies, name="quefrency")

    def plot(
        self,
        plot_type: str = "quefrency",
        ax: Axes | None = None,
        *,
        title: str | None = None,
        xlabel: str = "Quefrency [s]",
        ylabel: str = "Real cepstrum",
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Plot real coefficients against quefrency.

        Parameters
        ----------
        plot_type : str, default="quefrency"
            Only ``"quefrency"`` is supported.
        ax : matplotlib.axes.Axes, optional
            Existing axes. A new figure and axes are created when omitted.
        title : str, optional
            Plot title; defaults to the frame label.
        xlabel, ylabel : str
            Axis labels.
        **kwargs : Any
            Keyword arguments passed to ``Axes.plot``.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing one line per channel.

        Raises
        ------
        ValueError
            If another plot type is requested.

        Notes
        -----
        Plotting is an explicit compute boundary and materializes coefficients.

        Examples
        --------
        >>> cepstrum.plot()
        """
        if plot_type != "quefrency":
            raise ValueError("CepstralFrame.plot supports only plot_type='quefrency'.")
        import matplotlib.pyplot as plt

        target = ax if ax is not None else plt.subplots()[1]
        values = self._compute()
        for channel_index, label in enumerate(self.labels):
            target.plot(self.quefrencies, values[channel_index], label=label, **kwargs)
        if self.n_channels > 1:
            target.legend()
        target.set(xlabel=xlabel, ylabel=ylabel, title=title or self.label)
        return target


__all__ = ["CepstralFrame"]
