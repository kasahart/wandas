from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .channel_frame import ChannelFrame
    from .matrix_frame import MatrixFrame


def trim_channel_frame(cf: "ChannelFrame", start: float, end: float) -> "ChannelFrame":
    """
    Trim each channel in the specified range and return a new ChannelFrame.

    Parameters
    ----------
    cf : ChannelFrame
        The channel frame to trim.
    start : float
        Start time for trimming (seconds).
    end : float
        End time for trimming (seconds).

    Returns
    -------
    ChannelFrame
        A new ChannelFrame containing trimmed channels.
    """
    from .channel_frame import ChannelFrame

    trimmed_channels = [ch.trim(start, end) for ch in cf._channels]
    return ChannelFrame(channels=trimmed_channels, label=cf.label)


def cut_channel_frame(
    cf: "ChannelFrame",
    point_list: Union[list[int], list[float]],
    cut_len: Union[int, float],
    taper_rate: float = 0,
    dc_cut: bool = False,
) -> list["MatrixFrame"]:
    """
    Cut each channel and convert segments into new MatrixFrame objects.
    This function calls the cut method for each channel and groups them by segment.

    Parameters
    ----------
    cf : ChannelFrame
        The channel frame to cut.
    point_list : list of int or list of float
        List of cut points.
    cut_len : int or float
        Length of each segment to cut.
    taper_rate : float, default=0
        Taper rate to apply to the cut segments.
    dc_cut : bool, default=False
        Whether to remove DC component from the segments.

    Returns
    -------
    list of MatrixFrame
        A list of MatrixFrame objects, each containing one segment from all channels.
    """
    from .channel_frame import ChannelFrame

    # Collect cut results (list of Channel) from each channel
    cut_channels = [
        ch.cut(point_list, cut_len, taper_rate, dc_cut) for ch in cf._channels
    ]
    segment_num = len(cut_channels[0])
    matrix_frames = []
    for i in range(segment_num):
        new_channels = [ch_seg[i] for ch_seg in cut_channels]
        # Delegate conversion from ChannelFrame to MatrixFrame
        # to internal processing (to_matrix_frame)
        new_cf = ChannelFrame(
            channels=new_channels, label=f"{cf.label}, Segment:{i + 1}"
        )
        matrix_frames.append(new_cf.to_matrix_frame())
    return matrix_frames
