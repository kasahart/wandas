"""周波数変換や変換操作に関連するミックスインを提供するモジュール。"""

import logging
from typing import TYPE_CHECKING, Any, Optional, cast

from ...core.base_frame import BaseFrame
from .protocols import T_Transform

if TYPE_CHECKING:
    from wandas.frames.noct import NOctFrame
    from wandas.frames.spectral import SpectralFrame
    from wandas.frames.spectrogram import SpectrogramFrame


logger = logging.getLogger(__name__)


class ChannelTransformMixin:
    """周波数変換やその他の変換操作に関連するメソッドを提供するミックスイン。

    このミックスインは、FFT、STFT、ウェルチ法などの周波数解析と
    変換に関する操作を提供します。
    """

    def fft(
        self: T_Transform, n_fft: Optional[int] = None, window: str = "hann"
    ) -> "SpectralFrame":
        """高速フーリエ変換（FFT）を計算する。

        Args:
            n_fft: FFTポイント数。デフォルトはデータ長の次のべき乗。
            window: ウィンドウタイプ。デフォルトは"hann"。

        Returns:
            FFT結果を含むSpectralFrame
        """
        from wandas.frames.spectral import SpectralFrame
        from wandas.processing import FFT, create_operation

        params = {"n_fft": n_fft, "window": window}
        operation_name = "fft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # 操作インスタンスの作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("FFT", operation)
        # データに処理を適用
        spectrum_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"
        )

        if n_fft is None:
            is_even = spectrum_data.shape[-1] % 2 == 0
            _n_fft = (
                spectrum_data.shape[-1] * 2 - 2
                if is_even
                else spectrum_data.shape[-1] * 2 - 1
            )
        else:
            _n_fft = n_fft

        # BaseFrame型としてselfをキャスト
        base_self = cast(BaseFrame[Any], self)

        return SpectralFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            n_fft=_n_fft,
            window=operation.window,
            label=f"Spectrum of {self.label}",
            metadata={**self.metadata, "window": window, "n_fft": _n_fft},
            operation_history=[
                *self.operation_history,
                {"operation": "fft", "params": {"n_fft": _n_fft, "window": window}},
            ],
            channel_metadata=self._channel_metadata,
            previous=base_self,
        )

    def welch(
        self: T_Transform,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: int = 2048,
        window: str = "hann",
        average: str = "mean",
    ) -> "SpectralFrame":
        """ウェルチ法によるパワースペクトル密度を計算する。

        Args:
            n_fft: FFTポイント数。デフォルトは2048。
            hop_length: フレーム間のサンプル数。
                デフォルトはn_fft//4。
            win_length: ウィンドウの長さ。デフォルトはn_fft。
            window: ウィンドウの種類。デフォルトは"hann"。
            average: セグメントの平均化方法。デフォルトは"mean"。

        Returns:
            パワースペクトル密度を含むSpectralFrame
        """
        from wandas.frames.spectral import SpectralFrame
        from wandas.processing import Welch, create_operation

        params = dict(
            n_fft=n_fft or win_length,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            average=average,
        )
        operation_name = "welch"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # 操作インスタンスの作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("Welch", operation)
        # データに処理を適用
        spectrum_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"
        )

        # BaseFrame型としてselfをキャスト
        base_self = cast(BaseFrame[Any], self)

        return SpectralFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            n_fft=operation.n_fft,
            window=operation.window,
            label=f"Spectrum of {self.label}",
            metadata={**self.metadata, **params},
            operation_history=[
                *self.operation_history,
                {"operation": "welch", "params": params},
            ],
            channel_metadata=self._channel_metadata,
            previous=base_self,
        )

    def noct_spectrum(
        self: T_Transform,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ) -> "NOctFrame":
        """N-オクターブバンドスペクトルを計算する。

        Args:
            fmin: 最小中心周波数（Hz）。デフォルトは20 Hz。
            fmax: 最大中心周波数（Hz）。デフォルトは20000 Hz。
            n: バンド分割（1：オクターブ、3：1/3オクターブ）。デフォルトは3。
            G: 参照ゲイン（dB）。デフォルトは10 dB。
            fr: 参照周波数（Hz）。デフォルトは1000 Hz。

        Returns:
            N-オクターブバンドスペクトルを含むNOctFrame
        """
        from wandas.processing import NOctSpectrum, create_operation

        from ..noct import NOctFrame

        params = {"fmin": fmin, "fmax": fmax, "n": n, "G": G, "fr": fr}
        operation_name = "noct_spectrum"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # 操作インスタンスの作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("NOctSpectrum", operation)
        # データに処理を適用
        spectrum_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"
        )

        # BaseFrame型としてselfをキャスト
        base_self = cast(BaseFrame[Any], self)

        return NOctFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            fmin=fmin,
            fmax=fmax,
            n=n,
            G=G,
            fr=fr,
            label=f"1/{n}Oct of {self.label}",
            metadata={**self.metadata, **params},
            operation_history=[
                *self.operation_history,
                {
                    "operation": "noct_spectrum",
                    "params": params,
                },
            ],
            channel_metadata=self._channel_metadata,
            previous=base_self,
        )

    def stft(
        self: T_Transform,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
    ) -> "SpectrogramFrame":
        """短時間フーリエ変換を計算する。

        Args:
            n_fft: FFTポイント数。デフォルトは2048。
            hop_length: フレーム間のサンプル数。
                デフォルトはn_fft//4。
            win_length: ウィンドウの長さ。デフォルトはn_fft。
            window: ウィンドウの種類。デフォルトは"hann"。

        Returns:
            STFT結果を含むSpectrogramFrame
        """
        from wandas.processing import STFT, create_operation

        from ..spectrogram import SpectrogramFrame

        # ホップ長とウィンドウ長の設定
        _hop_length = hop_length if hop_length is not None else n_fft // 4
        _win_length = win_length if win_length is not None else n_fft

        params = {
            "n_fft": n_fft,
            "hop_length": _hop_length,
            "win_length": _win_length,
            "window": window,
        }
        operation_name = "stft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # 操作インスタンスの作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("STFT", operation)

        # データに処理を適用
        spectrogram_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectrogramFrame with operation {operation_name} added to graph"  # noqa: E501
        )

        # BaseFrame型としてselfをキャスト
        base_self = cast(BaseFrame[Any], self)

        # 新しいインスタンスの作成
        return SpectrogramFrame(
            data=spectrogram_data,
            sampling_rate=self.sampling_rate,
            n_fft=n_fft,
            hop_length=_hop_length,
            win_length=_win_length,
            window=window,
            label=f"stft({self.label})",
            metadata=self.metadata,
            operation_history=self.operation_history,
            channel_metadata=self._channel_metadata,
            previous=base_self,
        )

    def coherence(
        self: T_Transform,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        detrend: str = "constant",
    ) -> "SpectralFrame":
        """マグニチュード二乗コヒーレンスを計算する。

        Args:
            n_fft: FFTポイント数。デフォルトは2048。
            hop_length: フレーム間のサンプル数。
                デフォルトはn_fft//4。
            win_length: ウィンドウの長さ。デフォルトはn_fft。
            window: ウィンドウの種類。デフォルトは"hann"。
            detrend: トレンド除去方法。オプション："constant"、"linear"、None。

        Returns:
            マグニチュード二乗コヒーレンスを含むSpectralFrame
        """
        from wandas.core.metadata import ChannelMetadata
        from wandas.processing import Coherence, create_operation

        from ..spectral import SpectralFrame

        params = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "window": window,
            "detrend": detrend,
        }
        operation_name = "coherence"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # 操作インスタンスの作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("Coherence", operation)

        # データに処理を適用
        coherence_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"
        )

        # BaseFrame型としてselfをキャスト
        base_self = cast(BaseFrame[Any], self)

        # 新しいチャネルメタデータの作成
        channel_metadata = []
        for in_ch in self._channel_metadata:
            for out_ch in self._channel_metadata:
                meta = ChannelMetadata()
                meta.label = f"$\\gamma_{{{in_ch.label}, {out_ch.label}}}$"
                meta.unit = ""
                meta.ref = 1
                meta["metadata"] = dict(
                    in_ch=in_ch["metadata"], out_ch=out_ch["metadata"]
                )
                channel_metadata.append(meta)

        # 新しいインスタンスの作成
        return SpectralFrame(
            data=coherence_data,
            sampling_rate=self.sampling_rate,
            n_fft=operation.n_fft,
            window=operation.window,
            label=f"Coherence of {self.label}",
            metadata={**self.metadata, **params},
            operation_history=[
                *self.operation_history,
                {"operation": operation_name, "params": params},
            ],
            channel_metadata=channel_metadata,
            previous=base_self,
        )

    def csd(
        self: T_Transform,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "SpectralFrame":
        """クロススペクトル密度行列を計算する。

        Args:
            n_fft: FFTポイント数。デフォルトは2048。
            hop_length: フレーム間のサンプル数。
                デフォルトはn_fft//4。
            win_length: ウィンドウの長さ。デフォルトはn_fft。
            window: ウィンドウの種類。デフォルトは"hann"。
            detrend: トレンド除去方法。オプション："constant"、"linear"、None。
            scaling: スケーリング方法。オプション："spectrum"、"density"。
            average: セグメントの平均化方法。デフォルトは"mean"。

        Returns:
            クロススペクトル密度行列を含むSpectralFrame
        """
        from wandas.core.metadata import ChannelMetadata
        from wandas.frames.spectral import SpectralFrame
        from wandas.processing import CSD, create_operation

        params = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "window": window,
            "detrend": detrend,
            "scaling": scaling,
            "average": average,
        }
        operation_name = "csd"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # 操作インスタンスの作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("CSD", operation)

        # データに処理を適用
        csd_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"
        )

        # BaseFrame型としてselfをキャスト
        base_self = cast(BaseFrame[Any], self)

        # 新しいチャネルメタデータの作成
        channel_metadata = []
        for in_ch in self._channel_metadata:
            for out_ch in self._channel_metadata:
                meta = ChannelMetadata()
                meta.label = f"{operation_name}({in_ch.label}, {out_ch.label})"
                meta.unit = ""
                meta.ref = 1
                meta["metadata"] = dict(
                    in_ch=in_ch["metadata"], out_ch=out_ch["metadata"]
                )
                channel_metadata.append(meta)

        # 新しいインスタンスの作成
        return SpectralFrame(
            data=csd_data,
            sampling_rate=self.sampling_rate,
            n_fft=operation.n_fft,
            window=operation.window,
            label=f"$C_{{{in_ch.label}, {out_ch.label}}}$",
            metadata={**self.metadata, **params},
            operation_history=[
                *self.operation_history,
                {"operation": operation_name, "params": params},
            ],
            channel_metadata=channel_metadata,
            previous=base_self,
        )

    def transfer_function(
        self: T_Transform,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "SpectralFrame":
        """伝達関数行列を計算する。

        伝達関数は、周波数領域でのチャネル間の信号伝達特性を表し、
        システムの入力-出力関係を表します。

        Args:
            n_fft: FFTポイント数。デフォルトは2048。
            hop_length: フレーム間のサンプル数。
                デフォルトはn_fft//4。
            win_length: ウィンドウの長さ。デフォルトはn_fft。
            window: ウィンドウの種類。デフォルトは"hann"。
            detrend: トレンド除去方法。オプション："constant"、"linear"、None。
            scaling: スケーリング方法。オプション："spectrum"、"density"。
            average: セグメントの平均化方法。デフォルトは"mean"。

        Returns:
            伝達関数行列を含むSpectralFrame
        """
        from wandas.core.metadata import ChannelMetadata
        from wandas.frames.spectral import SpectralFrame
        from wandas.processing import TransferFunction, create_operation

        params = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "window": window,
            "detrend": detrend,
            "scaling": scaling,
            "average": average,
        }
        operation_name = "transfer_function"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # 操作インスタンスの作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("TransferFunction", operation)

        # データに処理を適用
        tf_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"
        )

        # BaseFrame型としてselfをキャスト
        base_self = cast(BaseFrame[Any], self)

        # 新しいチャネルメタデータの作成
        channel_metadata = []
        for in_ch in self._channel_metadata:
            for out_ch in self._channel_metadata:
                meta = ChannelMetadata()
                meta.label = f"$H_{{{in_ch.label}, {out_ch.label}}}$"
                meta.unit = ""
                meta.ref = 1
                meta["metadata"] = dict(
                    in_ch=in_ch["metadata"], out_ch=out_ch["metadata"]
                )
                channel_metadata.append(meta)

        # 新しいインスタンスの作成
        return SpectralFrame(
            data=tf_data,
            sampling_rate=self.sampling_rate,
            n_fft=operation.n_fft,
            window=operation.window,
            label=f"Transfer function of {self.label}",
            metadata={**self.metadata, **params},
            operation_history=[
                *self.operation_history,
                {"operation": operation_name, "params": params},
            ],
            channel_metadata=channel_metadata,
            previous=base_self,
        )
