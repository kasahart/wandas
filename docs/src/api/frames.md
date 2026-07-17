# Frames Module / フレームモジュール

The `wandas.frames` module provides various data frame classes for manipulating and representing audio data.
`wandas.frames` モジュールは、オーディオデータの操作と表現のための様々なデータフレームクラスを提供します。

## ChannelFrame

ChannelFrame is the basic frame for handling time-domain waveform data.
ChannelFrameは時間領域の波形データを扱うための基本的なフレームです。

::: wandas.frames.channel.ChannelFrame

Use the <a href="../learning-path/07_per_channel_calibration.html">per-channel
calibration learning app</a> to configure known conversion factors without
modifying the source frame. Calibrated physical values are available from
`frame.data` as a NumPy array; users do not need to manage the internal array
backend.

### `get_channel(..., validate_query_keys: bool = True)` parameter

- **validate_query_keys**: When `True` (default), dict-style `query` arguments are validated against the known channel metadata fields and any existing `extra` keys. Unknown keys raise `KeyError` with the message "Unknown channel metadata key". Set to `False` to skip this pre-validation and allow queries that reference keys not present on the model; in that case, normal matching proceeds and a no-match will raise the usual `KeyError` for no results.

### Source-time offsets and index-wise operations

`source_time_offset` records where each channel's local sample axis starts on
the original source timeline. Binary frame operators such as `frame_a + frame_b`
do not use this value for automatic alignment. They operate on the current array
indices after verifying that sampling rate, channel count, and shape match.

Different `source_time_offset` values are allowed. The result inherits the left
operand's `source_time_offset`, so `frame_a + frame_b` carries `frame_a`'s
source timeline. `channel_difference()` follows the same index-wise principle
within one frame and preserves the input channel offsets.

When a workflow needs source-time alignment, trim or otherwise align frames
explicitly before applying binary operators. A dedicated source-time alignment
API may be added separately in the future.

## SpectralFrame

SpectralFrame is a frame for handling frequency-domain data.
SpectralFrameは周波数領域のデータを扱うためのフレームです。

::: wandas.frames.spectral.SpectralFrame

## CepstralFrame

CepstralFrame represents a normalized real cepstrum on a quefrency axis. Start with
the [cepstral analysis guide](../how-to/cepstral-analysis.md) for the typed
`ChannelFrame -> CepstralFrame -> SpectralFrame` workflow.
CepstralFrameは、ケフレンシー軸上の正規化された実ケプストラムを表します。型付きの
`ChannelFrame -> CepstralFrame -> SpectralFrame`ワークフローは
[ケプストラム解析ガイド](../how-to/cepstral-analysis.md)を参照してください。

::: wandas.frames.cepstral.CepstralFrame

## CepstrogramFrame

CepstrogramFrame represents a real cepstrum at every STFT time frame with dimensions
`(channel, quefrency, time)`. It is created by `SpectrogramFrame.cepstrum()` and can
reconstruct a time-varying spectral envelope. See the
[cepstral analysis guide](../how-to/cepstral-analysis.md).
CepstrogramFrameは各STFT時間フレームの実ケプストラムを
`(channel, quefrency, time)`で表します。`SpectrogramFrame.cepstrum()`から生成し、
時間変化するスペクトル包絡を再構成できます。

::: wandas.frames.cepstrogram.CepstrogramFrame

## SpectrogramFrame

SpectrogramFrame is a frame for handling time-frequency domain (spectrogram) data.
SpectrogramFrameは時間-周波数領域（スペクトログラム）のデータを扱うフレームです。

::: wandas.frames.spectrogram.SpectrogramFrame

## NOctFrame

NOctFrame is a frame class for octave-band analysis.
NOctFrameはオクターブバンド解析のためのフレームクラスです。

::: wandas.frames.noct.NOctFrame

## RoughnessFrame

RoughnessFrame is a frame class for psychoacoustic roughness analysis results.
RoughnessFrameは心理音響ラフネス解析結果のためのフレームクラスです。

::: wandas.frames.roughness.RoughnessFrame

## Mixins

Mixins for extending frame functionality.
フレームの機能を拡張するためのミックスインです。

### ChannelProcessingMixin

::: wandas.frames.mixins.channel_processing_mixin.ChannelProcessingMixin

### ChannelTransformMixin

::: wandas.frames.mixins.channel_transform_mixin.ChannelTransformMixin
