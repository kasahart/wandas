from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from dask.array.core import Array as DaArray

from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelMetadata, FrameMetadata
from wandas.frames.spectral import SpectralFrame
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from wandas.frames.channel import ChannelFrame

logger = logging.getLogger(__name__)


class CepstralFrame(SpectralFrame):
    """Frame for real cepstrum data with quefrency-aware metadata."""

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        n_fft: int,
        window: str = "hann",
        label: str | None = None,
        metadata: FrameMetadata | dict[str, Any] | None = None,
        operation_history: list[dict[str, Any]] | None = None,
        channel_metadata: list[ChannelMetadata] | list[dict[str, Any]] | None = None,
        previous: BaseFrame[Any] | None = None,
    ) -> None:
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            window=window,
            label=label,
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=channel_metadata,
            previous=previous,
        )

    @property
    def quefrencies(self) -> NDArrayReal:
        """Return the quefrency axis in seconds."""
        import numpy as np

        return np.arange(self.n_fft) / self.sampling_rate

    @property
    def freqs(self) -> NDArrayReal:
        """Alias the plotting axis to quefrency values in seconds."""
        return self.quefrencies

    def lifter(self, cutoff: float, mode: str = "low") -> CepstralFrame:
        """Apply low-pass or high-pass liftering in the quefrency domain."""
        return self.apply_operation("lifter", cutoff=cutoff, mode=mode)

    def to_spectral_envelope(self) -> SpectralFrame:
        """Convert the cepstrum back to a smooth spectral envelope."""
        from wandas.processing import SpectralEnvelope, create_operation

        params = {"n_fft": self.n_fft}
        operation_name = "spectral_envelope"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("SpectralEnvelope", operation)
        envelope_data = operation.process(self._data)

        new_history = [
            *self.operation_history,
            {"operation": operation_name, "params": params},
        ]
        new_metadata = self.metadata.copy()
        new_metadata[operation_name] = params

        return SpectralFrame(
            data=envelope_data,
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            window=self.window,
            label=f"spectral_envelope({self.label})",
            metadata=new_metadata,
            operation_history=new_history,
            channel_metadata=self._relabel_channels(operation_name, operation.get_display_name()),
            previous=self,
        )

    def _apply_operation_impl(self, operation_name: str, **params: Any) -> CepstralFrame:
        """Apply cepstral-domain operations while preserving metadata and history."""
        from wandas.processing import create_operation

        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        operation = create_operation(operation_name, self.sampling_rate, **params)
        processed_data = operation.process(self._data)

        operation_metadata = {"operation": operation_name, "params": params}
        new_history = self.operation_history.copy()
        new_history.append(operation_metadata)
        new_metadata = self.metadata.copy()
        new_metadata[operation_name] = params

        creation_params: dict[str, Any] = {
            "data": processed_data,
            "metadata": new_metadata,
            "operation_history": new_history,
            "channel_metadata": self._relabel_channels(operation_name, operation.get_display_name()),
        }
        creation_params.update(operation.get_metadata_updates())
        return self._create_new_instance(**creation_params)

    def _get_dataframe_index(self) -> pd.Index[Any]:
        """Get quefrency index for DataFrame conversion."""
        return pd.Index(self.quefrencies, name="quefrency")

    def plot(
        self,
        plot_type: str = "frequency",
        ax: Axes | None = None,
        title: str | None = None,
        overlay: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        alpha: float = 1.0,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        Aw: bool = False,  # noqa: N803
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Plot cepstral data using quefrency-aware axis labeling."""
        return super().plot(
            plot_type=plot_type,
            ax=ax,
            title=title,
            overlay=overlay,
            xlabel=xlabel or "Quefrency [s]",
            ylabel=ylabel or "Cepstrum",
            alpha=alpha,
            xlim=xlim,
            ylim=ylim,
            Aw=Aw,
            **kwargs,
        )

    def ifft(self) -> ChannelFrame:
        """IFFT is not defined for cepstral data."""
        raise NotImplementedError(
            "IFFT is not supported for cepstral data. Use to_spectral_envelope() to reconstruct a smooth spectrum."
        )

    def info(self) -> None:
        """Display cepstral-domain information."""
        delta_q = 1.0 / self.sampling_rate

        print("CepstralFrame Information:")
        print(f"  Channels: {self.n_channels}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  FFT size: {self.n_fft}")
        print(f"  Quefrency range: {self.quefrencies[0]:.6f} - {self.quefrencies[-1]:.6f} s")
        print(f"  Quefrency bins: {len(self.quefrencies)}")
        print(f"  Quefrency resolution (ΔQ): {delta_q:.6f} s")
        print(f"  Channel labels: {self.labels}")
        self._print_operation_history()
