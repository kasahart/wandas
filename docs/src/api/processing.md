# Processing Module / 処理モジュール

The `wandas.processing` module provides various processing capabilities for audio data.
`wandas.processing` モジュールは、オーディオデータに対する様々な処理機能を提供します。

## Base Processing / 基本処理

Provides the public operation extension contracts. Use `AudioOperation` for
cross-channel algorithms and `ChannelIndependentAudioOperation` only when each output
channel depends exclusively on its corresponding input channel.
公開Operation拡張契約を提供します。cross-channel algorithmには`AudioOperation`を使い、
各出力channelが対応入力channelだけに依存する場合だけ
`ChannelIndependentAudioOperation`を使用します。

::: wandas.processing.AudioOperation
    options:
      show_root_heading: true

::: wandas.processing.ChannelIndependentAudioOperation
    options:
      show_root_heading: true

## Effects / エフェクト

Provides audio effect processing.
オーディオエフェクト処理を提供します。

::: wandas.processing.effects

## Filters / フィルター

Provides various audio filter processing.
様々なオーディオフィルター処理を提供します。

::: wandas.processing.filters

## Spectral Processing / スペクトル処理

Provides spectral analysis and processing capabilities.
スペクトル解析と処理機能を提供します。

::: wandas.processing.spectral

## Cepstral Processing / ケプストラム処理

Provides real-cepstrum analysis, symmetric liftering, and spectral-envelope
reconstruction. Most users should use the typed Frame methods described in the
[cepstral analysis guide](../how-to/cepstral-analysis.md).
実ケプストラム解析、対称リフタリング、スペクトル包絡再構成を提供します。通常は
[ケプストラム解析ガイド](../how-to/cepstral-analysis.md)の型付きFrameメソッドを利用してください。

::: wandas.processing.cepstral

## Statistical Processing / 統計処理

Provides statistical analysis functions for audio data.
オーディオデータの統計分析機能を提供します。

::: wandas.processing.stats

## Temporal Processing / 時間領域処理

Provides time-domain processing capabilities.
時間領域の処理機能を提供します。

::: wandas.processing.temporal
