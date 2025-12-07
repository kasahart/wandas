# Frames Module / フレームモジュール

The `wandas.frames` module provides various data frame classes for manipulating and representing audio data.
`wandas.frames` モジュールは、オーディオデータの操作と表現のための様々なデータフレームクラスを提供します。

## ChannelFrame

ChannelFrame is the basic frame for handling time-domain waveform data.
ChannelFrameは時間領域の波形データを扱うための基本的なフレームです。

::: wandas.frames.channel.ChannelFrame

## SpectralFrame

SpectralFrame is a frame for handling frequency-domain data.
SpectralFrameは周波数領域のデータを扱うためのフレームです。

::: wandas.frames.spectral.SpectralFrame

## SpectrogramFrame

SpectrogramFrame is a frame for handling time-frequency domain (spectrogram) data.
SpectrogramFrameは時間-周波数領域（スペクトログラム）のデータを扱うフレームです。

::: wandas.frames.spectrogram.SpectrogramFrame

## NOctFrame

NOctFrame is a frame class for octave-band analysis.
NOctFrameはオクターブバンド解析のためのフレームクラスです。

::: wandas.frames.noct.NOctFrame

## Mixins

Mixins for extending frame functionality.
フレームの機能を拡張するためのミックスインです。

### ChannelProcessingMixin

::: wandas.frames.mixins.channel_processing_mixin.ChannelProcessingMixin

### ChannelTransformMixin

::: wandas.frames.mixins.channel_transform_mixin.ChannelTransformMixin
