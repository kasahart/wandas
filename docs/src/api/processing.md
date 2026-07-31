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

`RmsTrend` and `SoundLevel` distinguish linear values from levels:

- `RmsTrend(dB=False)` returns windowed linear RMS; `dB=True` applies
  `20 log10(RMS / ref)`.
- `SoundLevel(dB=False)` returns the square root of frequency-weighted,
  exponentially smoothed power. `dB=True` applies
  `10 log10(smoothed_power / ref²)`.
- For finite output, `RmsTrend` floors its amplitude ratio at `1e-12`
  (-240 dB), while `SoundLevel` floors its power ratio at `1e-20` (-200 dB).
  Silence returns the relevant floor instead of negative infinity.
- The dB implementations keep per-channel calibration scales separate from raw
  samples. Before A/C frequency weighting, each complete channel whose peak falls
  outside a conservative normal-range band is normalized by one exact power of two;
  normal-range channels retain their bit-for-bit filter input. The removed exponent
  is restored with calibration and reference terms in the logarithmic domain after
  logarithmic RMS or smoothed power is formed. Finite tiny and huge samples therefore
  do not prematurely underflow or overflow in the filter. `dB=False` retains the
  linear calibrated processing path.
- A result is dB SPL only when the input is pressure in Pa and
  `ref=2e-5 Pa`. Other references produce relative dB and must be labeled with
  that reference.
- `Aw=True` or `freq_weighting="A"` applies the implemented digital
  A-frequency-weighting curve. Fast and Slow select 125 ms and 1 s exponential
  time constants. These are numerical implementation contracts, not complete
  IEC/JIS instrument-conformance claims.

::: wandas.processing.temporal
