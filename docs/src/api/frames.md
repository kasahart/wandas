# フレームモジュール

`wandas.frames` モジュールは、オーディオデータの操作と表現のための様々なデータフレームクラスを提供します。

## ChannelFrame

ChannelFrameは時間領域の波形データを扱うための基本的なフレームです。

::: wandas.frames.channel.ChannelFrame

## SpectralFrame

SpectralFrameは周波数領域のデータを扱うためのフレームです。

::: wandas.frames.spectral.SpectralFrame

## SpectrogramFrame

SpectrogramFrameは時間-周波数領域（スペクトログラム）のデータを扱うフレームです。

::: wandas.frames.spectrogram.SpectrogramFrame

## NOctFrame

NOctFrameはオクターブバンド解析のためのフレームクラスです。

::: wandas.frames.noct.NOctFrame

## Mixins

フレームの機能を拡張するためのミックスインです。

### ChannelProcessingMixin

::: wandas.frames.mixins.channel_processing_mixin.ChannelProcessingMixin

### ChannelTransformMixin

::: wandas.frames.mixins.channel_transform_mixin.ChannelTransformMixin
