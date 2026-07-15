# Processing Module / 処理モジュール

The `wandas.processing` module provides various processing capabilities for audio data.
`wandas.processing` モジュールは、オーディオデータに対する様々な処理機能を提供します。

## Base Processing / 基本処理

Provides basic processing operations.
基本的な処理操作を提供します。

::: wandas.processing.base

## Calibration / 校正

Derive auditable physical calibration values from a known reference signal and
apply their channel-wise factors without materializing the target signal. Start with
the [signal calibration guide](../how-to/calibrate-signals.md) for the complete
sound-pressure and generic sensor workflows.

既知の基準信号から監査可能な物理校正値を導出し、対象信号を実体化せずチャンネルごとの倍率を
適用します。音圧および一般センサーの手順は
[信号校正ガイド](../how-to/calibrate-signals.md)を参照してください。

::: wandas.processing.calibration

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
