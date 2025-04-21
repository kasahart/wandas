# wandas/core/spectrums.py

from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from wandas.core.channel_access_mixin import ChannelAccessMixin
from wandas.core.frequency_channel import FrequencyChannel

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class FrequencyChannelFrame(ChannelAccessMixin["FrequencyChannel"]):
    def __init__(self, channels: list["FrequencyChannel"], label: Optional[str] = None):
        """
        Initialize a FrequencyChannelFrame object.

        Parameters
        ----------
        channels : list of FrequencyChannel
            List of FrequencyChannel objects.
        label : str, optional
            Label for the spectrum.
        """
        self._channels = channels
        self.label = label
        self._channel_dict = {ch.label: ch for ch in self.channels}
        if len(self._channel_dict) != len(self):
            raise ValueError("Channel labels must be unique.")

    def plot(
        self,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        Aw: bool = False,  # noqa: N803
        overlay: bool = True,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Plot spectrum data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to use for plotting.
        title : str, optional
            Plot title.
        Aw : bool, default=False
            If True, apply A-weighting before plotting.
        overlay : bool, default=True
            If True, plot all channels on the same axes.
        plot_kwargs : dict, optional
            Additional keyword arguments for the plot function.

        Raises
        ------
        ValueError
            If ax is provided when overlay is False.
        """
        if ax is not None and not overlay:
            raise ValueError("ax must be None when overlay is False.")

        suptitle = title or self.label or "Spectrum"

        if not overlay:
            num_channels = len(self._channels)
            fig, axs = plt.subplots(
                num_channels, 1, figsize=(10, 4 * num_channels), sharex=True
            )
            if num_channels == 1:
                axs = [axs]  # Ensure axs is iterable when there's only one channel

            for i, channel in enumerate(self._channels):
                tmp = axs[i]
                channel.plot(ax=tmp, Aw=Aw, plot_kwargs=plot_kwargs)
                leg = tmp.get_legend()
                if leg:
                    leg.remove()

            fig.suptitle(suptitle)
            plt.tight_layout()
            plt.show()
            return

        if ax is None:
            fig, tmp = plt.subplots(figsize=(10, 4))
        else:
            tmp = ax

        for channel in self._channels:
            channel.plot(ax=tmp, Aw=Aw, plot_kwargs=plot_kwargs)

        tmp.grid(True)
        tmp.legend()
        tmp.set_title(suptitle)

        if ax is None:
            plt.tight_layout()
            plt.show()

    def plot_matrix(
        self,
        title: Optional[str] = None,
        Aw: bool = False,  # noqa: N803
    ) -> tuple["Figure", "Axes"]:
        """
        Plot channels in a matrix layout.

        Parameters
        ----------
        title : str, optional
            Plot title.
        Aw : bool, default=False
            If True, apply A-weighting before plotting.

        Returns
        -------
        tuple
            Tuple containing (figure, axes).
        """

        num_channels = len(self._channels)
        num_rows = int(np.ceil(np.sqrt(num_channels)))

        fig, axes = plt.subplots(
            num_rows,
            num_rows,
            figsize=(3 * num_rows, 3 * num_rows),
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        for ch, ax in zip(self._channels, axes):
            ch.plot(ax=ax, title=title, Aw=Aw)
            ax.set_title(title or self.label)

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        return fig, axes
