"""Roughness analysis frame for detailed psychoacoustic analysis."""

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from dask.array.core import Array as DaArray

from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelMetadata
from wandas.utils.optional_imports import require_matplotlib_pyplot
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


class RoughnessFrame(BaseFrame[NDArrayReal]):
    """
    Frame for detailed roughness analysis with Bark-band information.

    This frame contains specific roughness (R_spec) data organized by
    Bark frequency bands over time, calculated using the Daniel & Weber (1997)
    method.

    The relationship between total roughness and specific roughness follows:
    R = 0.25 * sum(R_spec, axis=bark_bands)

    Parameters
    ----------
    data : da.Array
        Specific roughness data with shape:
        - (n_bark_bands, n_time) for mono signals
        - (n_channels, n_bark_bands, n_time) for multi-channel signals
        where n_bark_bands is always 47.
    sampling_rate : float
        Sampling rate of the roughness time series in Hz.
        For overlap=0.5, this is approximately 10 Hz (100ms hop).
        For overlap=0.0, this is approximately 5 Hz (200ms hop).
    bark_axis : NDArrayReal
        Bark frequency axis with 47 values from 0.5 to 23.5 Bark.
    overlap : float
        Overlap coefficient used in the calculation (0.0 to 1.0).
    label : str, optional
        Frame label. Defaults to "roughness_spec".
    metadata : dict, optional
        Additional metadata.
    lineage : LineageNode, optional
        Constructor override for the runtime lineage. When omitted, a source node is
        created. ``operation_history`` is its public derived projection.
    channel_metadata : list[ChannelMetadata], optional
        Metadata for each channel.
    previous : BaseFrame, optional
        Compatibility/debug pointer to the immediate prior frame; not the
        provenance source of truth.

    Attributes
    ----------
    bark_axis : NDArrayReal
        Frequency axis in Bark scale.
    n_bark_bands : int
        Number of Bark bands (always 47).
    n_time_points : int
        Number of time points.
    time : NDArrayReal
        Time axis based on sampling rate.
    overlap : float
        Overlap coefficient used (0.0 to 1.0).

    Examples
    --------
    Create a roughness frame from a signal:

    >>> import wandas as wd
    >>> signal = wd.read("motor.wav")
    >>> roughness_spec = signal.roughness_dw_spec(overlap=0.5)
    >>>
    >>> # Plot Bark-Time heatmap
    >>> roughness_spec.plot()
    >>>
    >>> # Find dominant Bark band
    >>> dominant_idx = roughness_spec.data.mean(axis=1).argmax()
    >>> dominant_bark = roughness_spec.bark_axis[dominant_idx]
    >>> print(f"Dominant frequency: {dominant_bark:.1f} Bark")
    >>>
    >>> # Extract specific Bark band
    >>> bark_10_idx = np.argmin(np.abs(roughness_spec.bark_axis - 10.0))
    >>> roughness_at_10bark = roughness_spec.data[bark_10_idx, :]

    The Daniel & Weber (1997) roughness model calculates specific roughness
    for 47 critical bands (Bark scale) over time, then integrates them to
    produce the total roughness:

    .. math::
        R = 0.25 \\sum_{i=1}^{47} R'_i

    where R'_i is the specific roughness in the i-th Bark band.

    References
    ----------
    .. [1] Daniel, P., & Weber, R. (1997). "Psychoacoustical roughness:
           Implementation of an optimized model". Acta Acustica united with
           Acustica, 83(1), 113-123.
    """

    bark_axis: NDArrayReal
    overlap: float

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        bark_axis: NDArrayReal,
        overlap: float,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        channel_metadata: Sequence[ChannelMetadata | dict[str, Any]] | None = None,
        channel_ids: list[str] | None = None,
        previous: "BaseFrame[Any] | None" = None,
        source_time_offset: float | Sequence[float] | NDArrayReal = 0.0,
        lineage: Any | None = None,
        operation_history_prefix: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Initialize a roughness tensor and its exact analysis state.

        See the class docstring for parameter descriptions. The constructor requires
        one finite Bark coordinate for each of the 47 model bands and an overlap in
        the closed interval ``[0.0, 1.0]``.
        """
        # Validate dimensions
        if data.ndim not in (2, 3):
            raise ValueError(f"Data must be 2D or 3D (mono or multi-channel), got {data.ndim}D")

        # Validate Bark bands
        if data.shape[-2] != 47:
            raise ValueError(f"Expected 47 Bark bands, got {data.shape[-2]} (data shape: {data.shape})")

        normalized_bark_axis = np.asarray(bark_axis)
        if normalized_bark_axis.ndim != 1 or len(normalized_bark_axis) != 47:
            raise ValueError(f"bark_axis must have 47 elements, got shape {normalized_bark_axis.shape}")
        if not np.all(np.isfinite(normalized_bark_axis)):
            raise ValueError("bark_axis must contain 47 finite real numbers")

        # Validate overlap
        if not np.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
            raise ValueError(f"overlap must be in [0.0, 1.0], got {overlap}")

        self.bark_axis = normalized_bark_axis.copy()
        self.overlap = overlap

        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label or "roughness_spec",
            metadata=metadata,
            channel_metadata=channel_metadata,
            channel_ids=channel_ids,
            source_time_offset=source_time_offset,
            lineage=lineage,
            operation_history_prefix=operation_history_prefix,
            previous=previous,
        )

    @property
    def data(self) -> NDArrayReal:
        """
        Returns the computed data without squeezing.

        For RoughnessFrame, even mono signals have 2D shape (47, n_time)
        so we don't squeeze the channel dimension.

        Returns
        -------
        NDArrayReal
            Computed data array.
        """
        return self.compute()

    @property
    def n_bark_bands(self) -> int:
        """
        Number of Bark bands.

        Returns
        -------
        int
            Always 47 for the Daniel & Weber model.
        """
        return 47

    @property
    def n_time_points(self) -> int:
        """
        Number of time points in the roughness time series.

        Returns
        -------
        int
            Number of time frames in the analysis.
        """
        return int(self._data.shape[-1])

    @property
    def time(self) -> NDArrayReal:
        """
        Time axis based on sampling rate.

        Returns
        -------
        NDArrayReal
            Time values in seconds for each frame.
        """
        return np.arange(self.n_time_points) / self.sampling_rate

    @property
    def source_time(self) -> NDArrayReal:
        """Return roughness analysis time points on the source timeline."""
        return self.source_time_offset[:, None] + self.time[None, :]

    def _channel_count_from_data(self, data: DaArray) -> int:
        """Return the number of channels for mono or channel-bark-time data."""
        if data.ndim == 2:
            return 1
        return int(data.shape[-3])

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """
        Provide additional initialization arguments for RoughnessFrame.

        Returns
        -------
        dict
            Dictionary containing bark_axis and overlap
        """
        return {
            "bark_axis": self.bark_axis,
            "overlap": self.overlap,
        }

    def _get_dataframe_index(self) -> "pd.Index[Any]":
        """DataFrame index is not supported for RoughnessFrame."""
        raise NotImplementedError("DataFrame index is not supported for RoughnessFrame.")

    def _source_time_slice_context(self, keys: tuple[Any, ...]) -> tuple[Any, int, float] | None:
        """Roughness time is stored on the last data axis."""
        key_index = self._data.ndim - 2
        if key_index < 0 or key_index >= len(keys):
            return None
        return keys[key_index], self._data.shape[-1], 1.0 / self.sampling_rate

    def to_dataframe(self) -> "pd.DataFrame":
        """DataFrame conversion is not supported for RoughnessFrame.

        RoughnessFrame contains 3D data (channels, bark_bands, time_frames)
        which cannot be directly converted to a 2D DataFrame.

        Raises
        ------
        NotImplementedError
            Always raised as DataFrame conversion is not supported.
        """
        raise NotImplementedError("DataFrame conversion is not supported for RoughnessFrame.")

    def _apply_operation_impl(self, operation_name: str, **params: Any) -> "RoughnessFrame":
        raise NotImplementedError(
            f"Operation '{operation_name}' is not supported for RoughnessFrame. "
            "RoughnessFrame is typically a terminal node in the processing chain."
        )

    def plot(
        self,
        plot_type: str = "heatmap",
        ax: "Axes | None" = None,
        title: str | None = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        xlabel: str = "Time [s]",
        ylabel: str = "Frequency [Bark]",
        colorbar_label: str = "Specific Roughness [Asper/Bark]",
        **kwargs: Any,
    ) -> "Axes":
        """
        Plot Bark-Time-Roughness heatmap.

        For multi-channel signals, the mean across channels is plotted.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, a new figure is created.
        title : str, optional
            Plot title. If None, a default title is used.
        cmap : str, default="viridis"
            Colormap name for the heatmap.
        vmin, vmax : float, optional
            Color scale limits. If None, automatic scaling is used.
        xlabel : str, default="Time [s]"
            Label for the x-axis.
        ylabel : str, default="Frequency [Bark]"
            Label for the y-axis.
        colorbar_label : str, default="Specific Roughness [Asper/Bark]"
            Label for the colorbar.
        **kwargs : Any
            Additional keyword arguments passed to pcolormesh.

        Returns
        -------
        Axes
            The matplotlib axes object containing the plot.

        Examples
        --------
        >>> import wandas as wd
        >>> signal = wd.read("motor.wav")
        >>> roughness_spec = signal.roughness_dw_spec(overlap=0.5)
        >>> roughness_spec.plot(cmap="hot", title="Motor Roughness Analysis")
        """
        plt = require_matplotlib_pyplot("roughness plot")

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        # Select data to plot (first channel for mono, mean for multi-channel)
        # self._data is Dask array, self.data is computed NumPy array
        computed_data = self.compute()

        # Select data to plot (first channel for mono, mean for multi-channel)
        data_to_plot = computed_data if computed_data.ndim == 2 else computed_data.mean(axis=0)

        # Create heatmap
        im = ax.pcolormesh(
            self.time,
            self.bark_axis,
            data_to_plot,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

        # Labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is None:
            title = f"Roughness Spectrogram (overlap={self.overlap})"
        ax.set_title(title)

        # Colorbar
        plt.colorbar(im, ax=ax, label=colorbar_label)

        return ax
