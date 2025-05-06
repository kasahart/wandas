"""信号処理に関連するミックスインを提供するモジュール。"""

import logging
from typing import TYPE_CHECKING, Any, Optional, Union, cast

from .protocols import ProcessingFrameProtocol, T_Processing

if TYPE_CHECKING:
    from librosa._typing import (
        _FloatLike_co,
        _IntLike_co,
        _PadModeSTFT,
        _WindowSpec,
    )
logger = logging.getLogger(__name__)


class ChannelProcessingMixin:
    """信号処理に関連するメソッドを提供するミックスイン。

    このミックスインは、信号処理フィルタや変換操作など、
    オーディオ信号やその他の時系列データに適用される処理メソッドを提供します。
    """

    def high_pass_filter(
        self: T_Processing, cutoff: float, order: int = 4
    ) -> T_Processing:
        """信号にハイパスフィルタを適用する。

        Args:
            cutoff: フィルタのカットオフ周波数（Hz）
            order: フィルタの次数。デフォルトは4。

        Returns:
            フィルタ適用後の新しいChannelFrame
        """
        logger.debug(
            f"Setting up highpass filter: cutoff={cutoff}, order={order} (lazy)"
        )
        result = self.apply_operation("highpass_filter", cutoff=cutoff, order=order)
        return cast(T_Processing, result)

    def low_pass_filter(
        self: T_Processing, cutoff: float, order: int = 4
    ) -> T_Processing:
        """信号にローパスフィルタを適用する。

        Args:
            cutoff: フィルタのカットオフ周波数（Hz）
            order: フィルタの次数。デフォルトは4。

        Returns:
            フィルタ適用後の新しいChannelFrame
        """
        logger.debug(
            f"Setting up lowpass filter: cutoff={cutoff}, order={order} (lazy)"
        )
        result = self.apply_operation("lowpass_filter", cutoff=cutoff, order=order)
        return cast(T_Processing, result)

    def normalize(
        self: T_Processing, target_level: float = -20, channel_wise: bool = True
    ) -> T_Processing:
        """信号レベルを正規化する。

        このメソッドは、ターゲットRMSレベルに達するように信号振幅を調整します。

        Args:
            target_level: ターゲットRMSレベル（dB）。デフォルトは-20。
            channel_wise: Trueの場合、各チャネルを個別に正規化します。
                Falseの場合、すべてのチャネルに同じスケーリングを適用します。

        Returns:
            正規化された信号を含む新しいChannelFrame
        """
        logger.debug(
            f"Setting up normalize: target_level={target_level}, channel_wise={channel_wise} (lazy)"  # noqa: E501
        )
        result = self.apply_operation(
            "normalize", target_level=target_level, channel_wise=channel_wise
        )
        return cast(T_Processing, result)

    def a_weighting(self: T_Processing) -> T_Processing:
        """信号にA特性重み付けフィルタを適用する。

        A特性重み付けは、IEC 61672-1:2013規格に従って、
        人間の聴覚の知覚に近似するように周波数応答を調整します。

        Returns:
            A特性重み付けされた信号を含む新しいChannelFrame
        """
        result = self.apply_operation("a_weighting")
        return cast(T_Processing, result)

    def abs(self: T_Processing) -> T_Processing:
        """信号の絶対値を計算する。

        Returns:
            絶対値を含む新しいChannelFrame
        """
        result = self.apply_operation("abs")
        return cast(T_Processing, result)

    def power(self: T_Processing, exponent: float = 2.0) -> T_Processing:
        """信号のべき乗を計算する。

        Args:
            exponent: 信号を累乗する指数。デフォルトは2.0。

        Returns:
            べき乗された信号を含む新しいChannelFrame
        """
        result = self.apply_operation("power", exponent=exponent)
        return cast(T_Processing, result)

    def sum(self: T_Processing) -> T_Processing:
        """Sum all channels.

        Returns:
            A new ChannelFrame with summed signal.
        """
        result = self.apply_operation("sum")
        return cast(T_Processing, result)

    def mean(self: T_Processing) -> T_Processing:
        """Average all channels.

        Returns:
            A new ChannelFrame with averaged signal.
        """
        result = self.apply_operation("mean")
        return cast(T_Processing, result)

    def trim(
        self: T_Processing,
        start: float = 0,
        end: Optional[float] = None,
    ) -> T_Processing:
        """信号を指定された時間範囲にトリミングする。

        Args:
            start: 開始時間（秒）
            end: 終了時間（秒）

        Returns:
            トリミングされた信号を含む新しいChannelFrame

        Raises:
            ValueError: 終了時間が開始時間より前の場合
        """
        if end is None:
            end = self.duration
        if start > end:
            raise ValueError("start must be less than end")
        result = self.apply_operation("trim", start=start, end=end)
        return cast(T_Processing, result)

    def fix_length(
        self: T_Processing,
        length: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> T_Processing:
        """信号を指定された時間にする。

        Args:
            duration: 信号の長さ（秒）
            length: 信号の長さ（サンプル数）

        Returns:
            指定された長さに調整された信号を含む新しいChannelFrame
        """

        result = self.apply_operation("fix_length", length=length, duration=duration)
        return cast(T_Processing, result)

    def rms_trend(
        self: T_Processing,
        frame_length: int = 2048,
        hop_length: int = 512,
        dB: bool = False,  # noqa: N803
        Aw: bool = False,  # noqa: N803
    ) -> T_Processing:
        """信号のRMS傾向を計算する。

        このメソッドは、スライディングウィンドウ上で二乗平均平方根値を計算します。

        Args:
            frame_length: サンプル単位のスライディングウィンドウのサイズ。
            デフォルトは2048。
            hop_length: サンプル単位のウィンドウ間のホップ長。デフォルトは512。
            dB: RMS値をデシベルで返すかどうか。デフォルトはFalse。
            Aw: A特性重み付けを適用するかどうか。デフォルトはFalse。

        Returns:
            RMS傾向を含む新しいChannelFrame
        """
        # _channel_metadataにアクセスして参照値を取得
        frame = cast(ProcessingFrameProtocol, self)

        # _channel_metadataが存在することを確認してから参照
        ref_values = []
        if hasattr(frame, "_channel_metadata") and frame._channel_metadata:
            ref_values = [ch.ref for ch in frame._channel_metadata]

        result = self.apply_operation(
            "rms_trend",
            frame_length=frame_length,
            hop_length=hop_length,
            ref=ref_values,
            dB=dB,
            Aw=Aw,
        )

        # サンプリングレートを更新
        result_obj = cast(T_Processing, result)
        if hasattr(result_obj, "sampling_rate"):
            result_obj.sampling_rate = frame.sampling_rate / hop_length

        return result_obj

    def channel_difference(
        self: T_Processing, other_channel: Union[int, str] = 0
    ) -> T_Processing:
        """チャネル間の差分を計算する。

        Args:
            other_channel: 参照チャネルのインデックスまたはラベル。デフォルトは0。

        Returns:
            チャネル差分を含む新しいChannelFrame
        """
        # label2indexはBaseFrameのメソッド
        if isinstance(other_channel, str):
            if hasattr(self, "label2index"):
                other_channel = self.label2index(other_channel)

        result = self.apply_operation("channel_difference", other_channel=other_channel)
        return cast(T_Processing, result)

    def resampling(
        self: T_Processing,
        target_sr: float,
        **kwargs: Any,
    ) -> T_Processing:
        """オーディオデータをリサンプリングする。

        Args:
            target_sr: 目標サンプリングレート（Hz）
            **kwargs: 追加のリサンプリングパラメータ

        Returns:
            リサンプリングされたChannelFrame
        """
        return cast(
            T_Processing,
            self.apply_operation(
                "resampling",
                target_sr=target_sr,
                **kwargs,
            ),
        )

    def hpss_harmonic(
        self: T_Processing,
        kernel_size: Union[
            "_IntLike_co", tuple["_IntLike_co", "_IntLike_co"], list["_IntLike_co"]
        ] = 31,
        power: float = 2,
        margin: Union[
            "_FloatLike_co",
            tuple["_FloatLike_co", "_FloatLike_co"],
            list["_FloatLike_co"],
        ] = 1,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: "_WindowSpec" = "hann",
        center: bool = True,
        pad_mode: "_PadModeSTFT" = "constant",
    ) -> T_Processing:
        """
        Extract harmonic components using HPSS
         (Harmonic-Percussive Source Separation).
        """
        result = self.apply_operation(
            "hpss_harmonic",
            kernel_size=kernel_size,
            power=power,
            margin=margin,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
        return cast(T_Processing, result)

    def hpss_percussive(
        self: T_Processing,
        kernel_size: Union[
            "_IntLike_co", tuple["_IntLike_co", "_IntLike_co"], list["_IntLike_co"]
        ] = 31,
        power: float = 2,
        margin: Union[
            "_FloatLike_co",
            tuple["_FloatLike_co", "_FloatLike_co"],
            list["_FloatLike_co"],
        ] = 1,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: "_WindowSpec" = "hann",
        center: bool = True,
        pad_mode: "_PadModeSTFT" = "constant",
    ) -> T_Processing:
        """
        Extract percussive components using HPSS
        (Harmonic-Percussive Source Separation).

        This method separates the percussive (tonal) components from the signal.

        Args:
            kernel_size: Median filter size for HPSS.
            power: Exponent for the Weiner filter used in HPSS.
            margin: Margin size for the separation.

        Returns:
            A new ChannelFrame containing the harmonic components.
        """
        result = self.apply_operation(
            "hpss_percussive",
            kernel_size=kernel_size,
            power=power,
            margin=margin,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
        return cast(T_Processing, result)
