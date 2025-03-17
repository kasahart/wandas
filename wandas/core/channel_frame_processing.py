from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .channel_frame import ChannelFrame
    from .matrix_frame import MatrixFrame


def trim_channel_frame(cf: "ChannelFrame", start: float, end: float) -> "ChannelFrame":
    """各チャンネルを指定区間でトリミングし、新しい ChannelFrame を返す。"""
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
    各チャンネルを切り出し、セグメント毎に新規の MatrixFrame オブジェクトに変換。
    ここでは各チャンネルの cut メソッドを呼び出し、セグメントごとにグループ化する。
    """
    from .channel_frame import ChannelFrame

    # 各チャンネルの cut (list of Channel) をまとめる
    cut_channels = [
        ch.cut(point_list, cut_len, taper_rate, dc_cut) for ch in cf._channels
    ]
    segment_num = len(cut_channels[0])
    matrix_frames = []
    for i in range(segment_num):
        new_channels = [ch_seg[i] for ch_seg in cut_channels]
        # ChannelFrameからMatrixFrameへの変換は内部処理（to_matrix_frame）に委譲
        new_cf = ChannelFrame(
            channels=new_channels, label=f"{cf.label}, Segment:{i + 1}"
        )
        matrix_frames.append(new_cf.to_matrix_frame())
    return matrix_frames
