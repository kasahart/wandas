"""Roughness analysis frame for detailed psychoacoustic analysis."""

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import dask.array as da
import numpy as np

from wandas.core.base_frame import BaseFrame
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
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
    operation_history : list[dict], optional
        History of operations applied to this frame.
    channel_metadata : list[ChannelMetadata], optional
        Metadata for each channel.
    previous : BaseFrame, optional
        Reference to the previous frame in the processing chain.

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
    >>> signal = wd.read_wav("motor.wav")
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

    Notes
    -----
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

    def __init__(
        self,
        data: da.Array,
        sampling_rate: float,
        bark_axis: NDArrayReal,
        overlap: float,
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        operation_history: Optional[list[dict[str, Any]]] = None,
        channel_metadata: Optional[list[Any]] = None,
        previous: Optional["BaseFrame[Any]"] = None,
    ) -> None:
        """Initialize a RoughnessFrame."""
        # Validate dimensions
        if data.ndim not in (2, 3):
            raise ValueError(
                f"Data must be 2D or 3D (mono or multi-channel), got {data.ndim}D"
            )

        # Validate Bark bands
        if data.shape[-2] != 47:
            raise ValueError(
                f"Expected 47 Bark bands, got {data.shape[-2]} "
                f"(data shape: {data.shape})"
            )

        if len(bark_axis) != 47:
            raise ValueError(f"bark_axis must have 47 elements, got {len(bark_axis)}")

        # Validate overlap
        if not 0.0 <= overlap <= 1.0:
            raise ValueError(f"overlap must be in [0.0, 1.0], got {overlap}")

        # Store Bark-specific attributes
        self._bark_axis = bark_axis
        self._overlap = overlap

        # Initialize base frame
        metadata = metadata or {}
        metadata["overlap"] = overlap

        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label or "roughness_spec",
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=channel_metadata,
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
    def bark_axis(self) -> NDArrayReal:
        """
        Bark frequency axis.

        Returns
        -------
        NDArrayReal
            Array of 47 Bark values from 0.5 to 23.5 Bark.
        """
        return self._bark_axis

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
    def overlap(self) -> float:
        """
        Overlap coefficient used in the calculation.

        Returns
        -------
        float
            Overlap value between 0.0 and 1.0.
        """
        return self._overlap

    @property
    def _n_channels(self) -> int:
        """
        Return the number of channels.

        Returns
        -------
        int
            Number of channels. For 2D data (mono), returns 1.
        """
        if self._data.ndim == 2:
            return 1
        return int(self._data.shape[0])

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """
        Provide additional initialization arguments for RoughnessFrame.

        Returns
        -------
        dict
            Dictionary containing bark_axis and overlap
        """
        return {
            "bark_axis": self._bark_axis,
            "overlap": self._overlap,
        }

    def _binary_op(
        self,
        other: Union["RoughnessFrame", int, float, NDArrayReal, da.Array],
        op: "Callable[[da.Array, Any], da.Array]",
        symbol: str,
    ) -> "RoughnessFrame":
        """
        Common implementation for binary operations.

        Parameters
        ----------
        other : RoughnessFrame, int, float, NDArrayReal, or da.Array
            Right operand for the operation.
        op : Callable
            Function to execute the operation.
        symbol : str
            Symbolic representation of the operation.

        Returns
        -------
        RoughnessFrame
            A new RoughnessFrame with the operation result.

        Raises
        ------
        ValueError
            If sampling rates don't match or shapes are incompatible.
        """
        logger.debug(f"Setting up {symbol} operation (lazy)")

        # Handle metadata and operation_history
        metadata = self.metadata.copy() if self.metadata else {}
        operation_history = (
            self.operation_history.copy() if self.operation_history else []
        )

        # Check if other is a RoughnessFrame
        if isinstance(other, RoughnessFrame):
            if self.sampling_rate != other.sampling_rate:
                raise ValueError(
                    f"Sampling rates do not match: {self.sampling_rate} vs "
                    f"{other.sampling_rate}"
                )

            if self._data.shape != other._data.shape:
                raise ValueError(
                    f"Shape mismatch: {self._data.shape} vs {other._data.shape}"
                )

            # Apply operation
            result_data = op(self._data, other._data)

            # Update operation history
            operation_history.append(
                {"name": f"binary_op_{symbol}", "params": {"other": "RoughnessFrame"}}
            )

        else:
            # Scalar or array operation
            if isinstance(other, np.ndarray):
                other = da.from_array(other, chunks=self._data.chunks)

            result_data = op(self._data, other)

            operation_history.append(
                {"name": f"binary_op_{symbol}", "params": {"other": str(type(other))}}
            )

        # Create new instance
        return RoughnessFrame(
            data=result_data,
            sampling_rate=self.sampling_rate,
            bark_axis=self._bark_axis,
            overlap=self._overlap,
            label=self.label,
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=self._channel_metadata,
            previous=self,
        )

    def _apply_operation_impl(
        self, operation_name: str, **params: Any
    ) -> "RoughnessFrame":
        """
        Implementation of operation application.

        Note: RoughnessFrame is typically a terminal node in processing chains.
        Most operations are not directly applicable to spectral roughness data.

        Parameters
        ----------
        operation_name : str
            Name of the operation to apply.
        **params : Any
            Operation parameters.

        Returns
        -------
        RoughnessFrame
            A new RoughnessFrame with the operation applied.

        Raises
        ------
        NotImplementedError
            As most operations are not applicable to roughness spectrograms.
        """
        raise NotImplementedError(
            f"Operation '{operation_name}' is not supported for RoughnessFrame. "
            "RoughnessFrame is typically a terminal node in the processing chain."
        )

    def plot(
        self,
        plot_type: str = "heatmap",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
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
        >>> signal = wd.read_wav("motor.wav")
        >>> roughness_spec = signal.roughness_dw_spec(overlap=0.5)
        >>> roughness_spec.plot(cmap="hot", title="Motor Roughness Analysis")
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        # Select data to plot (first channel for mono, mean for multi-channel)
        # self._data is Dask array, self.data is computed NumPy array
        computed_data = self.compute()

        if computed_data.ndim == 2:
            # Mono: (47, n_time)
            data_to_plot = computed_data
        else:
            # Multi-channel: (n_channels, 47, n_time) -> average to (47, n_time)
            data_to_plot = computed_data.mean(axis=0)

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
            title = f"Roughness Spectrogram (overlap={self._overlap})"
        ax.set_title(title)

        # Colorbar
        plt.colorbar(im, ax=ax, label=colorbar_label)

        return ax
