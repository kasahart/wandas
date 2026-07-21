"""Time-varying real-cepstrum frame on quefrency and time axes."""

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
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from wandas.frames.spectrogram import SpectrogramFrame

_DEFAULT_COLOR_PERCENTILE = 99.0


class CepstrogramFrame(BaseFrame[NDArrayReal]):
    """Immutable, lazy real cepstrum evolving over STFT time frames.

    Data is rank three with dimensions ``(channel, quefrency, time)``.
    ``n_fft`` defines the complete circular quefrency axis, while
    ``hop_length`` defines the spacing of the retained STFT time frames.
    ``lifter()`` preserves this frame family and
    ``to_spectral_envelope()`` returns a
    :class:`~wandas.frames.spectrogram.SpectrogramFrame`.

    Parameters
    ----------
    data : dask.array.Array
        Real coefficients shaped ``(quefrency, time)`` or
        ``(channels, quefrency, time)``.
    sampling_rate : float
        Sampling rate in Hz defining both axis spacings.
    n_fft : int
        Positive FFT size of the complete cepstrum.
    hop_length : int
        Positive sample distance between adjacent time frames.
    win_length : int, optional
        Analysis-window length inherited from the source spectrogram. Defaults
        to ``n_fft``.
    window : str, default="hann"
        Analysis-window name inherited from the source spectrogram.
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
        Per-channel source timeline offsets.
    lineage : LineageNode, optional
        Authoritative runtime semantic lineage.
    operation_history_prefix : sequence, default=()
        Persisted display history for a new source frame.

    Raises
    ------
    TypeError
        If coefficients are complex or domain parameters have invalid types.
    ValueError
        If rank, FFT size, coefficient count, or time-analysis parameters are
        invalid.

    Examples
    --------
    >>> cepstrogram = frame.stft(n_fft=2048).cepstrum()
    >>> envelope = cepstrogram.lifter(0.002).to_spectral_envelope()
    """

    _xarray_dim_suffix = ("channel", "quefrency", "time")

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        n_fft: int,
        hop_length: int,
        win_length: int | None = None,
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
        if data.ndim == 2:
            data = data.reshape((1, *data.shape))
        elif data.ndim != 3:
            raise ValueError(
                "Invalid data shape for CepstrogramFrame\n"
                f"  Got: {data.shape} ({data.ndim}D)\n"
                "  Expected: (quefrency, time) or (channels, quefrency, time)\n"
                "Reshape data so quefrency precedes the time axis."
            )
        if np.issubdtype(data.dtype, np.complexfloating):
            raise TypeError(
                "CepstrogramFrame requires real-valued coefficients\n"
                f"  Got: {data.dtype}\n"
                "  Expected: a real numeric dtype\n"
                "Construct it from SpectrogramFrame.cepstrum()."
            )

        validate_sampling_rate(sampling_rate)
        normalized_n_fft = self._positive_integer(n_fft, name="n_fft")
        if int(data.shape[-2]) > normalized_n_fft:
            raise ValueError(
                "CepstrogramFrame cannot contain more quefrency bins than n_fft\n"
                f"  Got: {data.shape[-2]} bins for n_fft={normalized_n_fft}\n"
                "  Expected: coefficient count <= n_fft\n"
                "Use the FFT size of the source spectrogram or trim the data."
            )
        normalized_hop_length = self._positive_integer(
            hop_length,
            name="hop_length",
        )
        normalized_win_length = (
            normalized_n_fft if win_length is None else self._positive_integer(win_length, name="win_length")
        )
        if normalized_win_length > normalized_n_fft:
            raise ValueError(
                "Invalid win_length for CepstrogramFrame\n"
                f"  Got: {normalized_win_length} for n_fft={normalized_n_fft}\n"
                "  Expected: win_length <= n_fft\n"
                "Use the analysis state of the source spectrogram."
            )
        if normalized_hop_length > normalized_win_length:
            raise ValueError(
                "Invalid hop_length for CepstrogramFrame\n"
                f"  Got: {normalized_hop_length} for win_length={normalized_win_length}\n"
                "  Expected: hop_length <= win_length\n"
                "Use the analysis state of the source spectrogram."
            )
        if not isinstance(window, str) or not window:
            raise TypeError("CepstrogramFrame window must be a non-empty string.")

        self._n_fft = normalized_n_fft
        self._hop_length = normalized_hop_length
        self._win_length = normalized_win_length
        self._window = window
        self._pending_sampling_rate = float(sampling_rate)
        self._pending_hop_length = normalized_hop_length
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
        del self._pending_hop_length

    @staticmethod
    def _positive_integer(value: int, *, name: str) -> int:
        """Return a positive integer domain parameter."""
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(
                f"Invalid {name} for CepstrogramFrame\n"
                f"  Got: {type(value).__name__}\n"
                "  Expected: a positive integer\n"
                "Use the analysis state of the source spectrogram."
            )
        normalized = int(value)
        if normalized <= 0:
            raise ValueError(
                f"Invalid {name} for CepstrogramFrame\n"
                f"  Got: {normalized}\n"
                "  Expected: a positive integer\n"
                "Use the analysis state of the source spectrogram."
            )
        return normalized

    @property
    def n_fft(self) -> int:
        """Return the immutable circular period of the cepstrum."""
        return self._n_fft

    @property
    def hop_length(self) -> int:
        """Return the immutable sample spacing between time frames."""
        return self._hop_length

    @property
    def win_length(self) -> int:
        """Return the immutable originating analysis-window length."""
        return self._win_length

    @property
    def window(self) -> str:
        """Return the immutable originating analysis-window name."""
        return self._window

    @property
    def sampling_rate(self) -> float:
        """Return the immutable rate defining quefrency and time spacing."""
        return float(self._xr.attrs["sampling_rate"])

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        validate_sampling_rate(value)
        current = self._xr.attrs.get("sampling_rate")
        if current is not None and float(current) != float(value):
            raise AttributeError("CepstrogramFrame sampling_rate is immutable because it defines both axes.")
        self._xr.attrs["sampling_rate"] = float(value)

    @property
    def n_quefrency_bins(self) -> int:
        """Return the represented quefrency-bin count."""
        return int(self._data.shape[-2])

    @property
    def n_frames(self) -> int:
        """Return the time-frame count."""
        return int(self._data.shape[-1])

    @property
    def quefrencies(self) -> NDArrayReal:
        """Return a defensive copy of represented quefrencies in seconds."""
        return np.asarray(self._xr.coords["quefrency"].values, dtype=float).copy()

    @property
    def times(self) -> NDArrayReal:
        """Return a defensive copy of frame times in seconds."""
        return np.asarray(self._xr.coords["time"].values, dtype=float).copy()

    @property
    def source_times(self) -> NDArrayReal:
        """Return frame times on each channel's original source timeline."""
        return self.source_time_offset[:, None] + self.times[None, :]

    def _xarray_coords(self, data: DaArray) -> dict[str, Any]:
        """Build channel, quefrency, and time coordinates without computing."""
        coords = super()._xarray_coords(data)
        dims = self._xarray_dims(data)
        sampling_rate = getattr(self, "_pending_sampling_rate", None)
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        if "quefrency" in dims:
            coords["quefrency"] = (
                "quefrency",
                np.arange(int(data.shape[-2]), dtype=float) / sampling_rate,
            )
        if "time" in dims:
            hop_length = getattr(self, "_pending_hop_length", self.hop_length)
            coords["time"] = (
                "time",
                np.arange(int(data.shape[-1]), dtype=float) * hop_length / sampling_rate,
            )
        return coords

    def _create_new_instance(self, data: DaArray, **kwargs: Any) -> CepstrogramFrame:
        """Recreate the frame while retaining a sliced quefrency coordinate."""
        result = cast("CepstrogramFrame", super()._create_new_instance(data=data, **kwargs))
        if int(result._data.shape[-2]) == len(self.quefrencies):
            result._xr = result._xr.assign_coords(quefrency=("quefrency", self.quefrencies))
        return result

    def _handle_multidim_indexing(self, key: tuple[Any, ...]) -> CepstrogramFrame:
        """Preserve represented quefrencies while time slices reset locally."""
        result = cast("CepstrogramFrame", super()._handle_multidim_indexing(key))
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
    ) -> CepstrogramFrame:
        """Require matching time-varying cepstral state for frame arithmetic."""
        if isinstance(other, BaseFrame) and not isinstance(other, CepstrogramFrame):
            raise TypeError("CepstrogramFrame arithmetic requires another CepstrogramFrame or a scalar/array operand.")
        if isinstance(other, CepstrogramFrame):
            domain_state = (
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
            )
            other_domain_state = (
                other.n_fft,
                other.hop_length,
                other.win_length,
                other.window,
            )
            if domain_state != other_domain_state:
                raise ValueError("Cepstrogram analysis state must match exactly.")
            if not np.array_equal(self.quefrencies, other.quefrencies):
                raise ValueError("Cepstrogram quefrency coordinates must match exactly.")
            if not np.array_equal(self.times, other.times):
                raise ValueError("Cepstrogram time coordinates must match exactly.")
        return cast(
            "CepstrogramFrame",
            super()._binary_operand_op(other, op, symbol, reverse=reverse),
        )

    @recipe_operation("wandas.cepstrogram.lifter")
    def lifter(
        self,
        cutoff: float,
        mode: Literal["low", "high"] = "low",
    ) -> CepstrogramFrame:
        """Keep low- or high-quefrency coefficients at every time frame.

        Parameters
        ----------
        cutoff : float
            Positive quefrency boundary in seconds. It must reach at least one
            bin and remain below half of the complete cepstrum.
        mode : {"low", "high"}, default="low"
            ``"low"`` keeps the smooth-envelope region; ``"high"`` keeps the
            complementary fine structure.

        Returns
        -------
        CepstrogramFrame
            New lazy coefficients with unchanged time and channel axes.

        Raises
        ------
        ValueError
            If the quefrency axis was sliced or the cutoff is not representable.

        Notes
        -----
        This method only builds a Dask graph.
        """
        self._require_complete_quefrency_axis("lifter")
        return cast(
            "CepstrogramFrame",
            self._apply_named_operation(
                "lifter",
                cutoff=cutoff,
                mode=mode,
                axis=-2,
            ),
        )

    @recipe_operation("wandas.cepstrogram.to_spectral_envelope")
    def to_spectral_envelope(self) -> SpectrogramFrame:
        """Reconstruct a smooth magnitude spectrogram with zero phase.

        Returns
        -------
        SpectrogramFrame
            New lazy frequency-time data preserving the original STFT analysis
            state, channels, metadata, and source-time offsets.

        Raises
        ------
        ValueError
            If the quefrency axis was sliced. Asymmetric concrete coefficients
            raise when the lazy result is computed.

        Notes
        -----
        This method only builds a Dask graph.
        """
        self._require_complete_quefrency_axis("to_spectral_envelope")
        from wandas.frames.spectrogram import SpectrogramFrame
        from wandas.processing import SpectralEnvelope, create_operation

        operation = cast(
            "SpectralEnvelope",
            create_operation("spectral_envelope", self.sampling_rate, axis=-2),
        )
        return SpectrogramFrame(
            data=operation.process(self._data),
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            label=f"Spectral envelope of {self.label}",
            metadata=self.metadata,
            channel_metadata=self._borrowed_channel_metadata_descriptors(),
            channel_ids=self._channel_ids,
            previous=self,
            source_time_offset=self.source_time_offset,
            lineage=self._required_semantic_lineage(),
        )

    def _require_complete_quefrency_axis(self, operation_name: str) -> None:
        """Reject transforms after slicing the circular quefrency axis."""
        expected = np.arange(self.n_fft, dtype=float) / self.sampling_rate
        if self.n_quefrency_bins != self.n_fft or not np.array_equal(
            self.quefrencies,
            expected,
        ):
            raise ValueError(
                f"{operation_name} requires a complete, unsliced quefrency axis. "
                "Apply the operation before slicing the CepstrogramFrame."
            )

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """Return domain state required by ``_create_new_instance``."""
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": self.window,
        }

    def _get_dataframe_index(self) -> pd.Index[Any]:
        """Reject a lossy 3D-to-2D conversion."""
        raise NotImplementedError("DataFrame conversion is not supported for CepstrogramFrame.")

    def to_dataframe(self) -> pd.DataFrame:
        """Reject conversion because the frame has three semantic dimensions.

        Raises
        ------
        NotImplementedError
            Always raised. Materialize selected data when a tabular representation
            is required.
        """
        raise NotImplementedError("DataFrame conversion is not supported for CepstrogramFrame.")

    def plot(
        self,
        plot_type: str = "cepstrogram",
        ax: Axes | None = None,
        *,
        title: str | None = None,
        xlabel: str = "Time [s]",
        ylabel: str = "Quefrency [s]",
        cmap: str = "RdBu_r",
        qmin: float = 0.0,
        qmax: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Plot real coefficients over time and quefrency.

        Parameters
        ----------
        plot_type : str, default="cepstrogram"
            Only ``"cepstrogram"`` is supported.
        ax : matplotlib.axes.Axes, optional
            Existing axes for a single-channel frame. Multi-channel frames
            create one axes per channel when omitted.
        title : str, optional
            Plot title prefix; defaults to the frame label.
        xlabel, ylabel : str
            Axis labels.
        cmap : str, default="RdBu_r"
            Matplotlib colormap for signed real coefficients.
        qmin, qmax : float, optional
            Display range on the quefrency axis in seconds.
        vmin, vmax : float, optional
            Shared color limits. When omitted, a symmetric robust range is
            estimated from the displayed coefficients except the dominant
            zero-quefrency row.
        **kwargs : Any
            Additional keyword arguments passed to ``Axes.pcolormesh``.

        Returns
        -------
        matplotlib.axes.Axes or Iterator[matplotlib.axes.Axes]
            One axes for mono data or an iterator for multiple channels.

        Notes
        -----
        Plotting is an explicit compute boundary.
        """
        if plot_type != "cepstrogram":
            raise ValueError("CepstrogramFrame.plot supports only plot_type='cepstrogram'.")
        if ax is not None and self.n_channels != 1:
            raise ValueError(
                "An explicit axes can plot only one CepstrogramFrame channel. "
                "Select a channel first or omit ax to create separate panels."
            )

        import matplotlib.pyplot as plt

        upper = self.quefrencies[-1] if qmax is None else qmax
        represented = (self.quefrencies >= qmin) & (self.quefrencies <= upper)
        if not np.any(represented):
            raise ValueError("The requested quefrency plot range contains no bins.")

        values = self._compute()
        displayed_values = values[:, represented, :]
        scale_values = displayed_values
        if self.quefrencies[represented][0] == 0.0 and displayed_values.shape[-2] > 1:
            scale_values = displayed_values[:, 1:, :]
        finite_values = np.abs(scale_values[np.isfinite(scale_values)])
        if finite_values.size and (vmin is None or vmax is None):
            robust_limit = float(np.percentile(finite_values, _DEFAULT_COLOR_PERCENTILE))
            if robust_limit > 0:
                vmin = -robust_limit if vmin is None else vmin
                vmax = robust_limit if vmax is None else vmax

        if ax is None:
            _, subplot_grid = plt.subplots(
                self.n_channels,
                1,
                squeeze=False,
                sharex=True,
            )
            axes = list(subplot_grid[:, 0])
        else:
            axes = [ax]

        for channel_index, target in enumerate(axes):
            target.pcolormesh(
                self.times,
                self.quefrencies[represented],
                values[channel_index, represented, :],
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
            channel_title = title or self.label
            if self.n_channels > 1:
                channel_title = f"{channel_title} — {self.labels[channel_index]}"
            target.set(
                xlabel=xlabel,
                ylabel=ylabel,
                title=channel_title,
            )
        return axes[0] if self.n_channels == 1 else iter(axes)


__all__ = ["CepstrogramFrame"]
