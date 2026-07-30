# Frames Module / フレームモジュール

The `wandas.frames` module provides various data frame classes for manipulating and representing audio data.
`wandas.frames` モジュールは、オーディオデータの操作と表現のための様々なデータフレームクラスを提供します。

## ChannelFrame

ChannelFrame is the basic frame for handling time-domain waveform data.
ChannelFrameは時間領域の波形データを扱うための基本的なフレームです。

Frame annotations are updated immutably with `with_label()`, `with_metadata()`,
or `with_channel_extra()`. Use `with_source_time_offset()`
for portable source-time intent and `rename_channels()` on any Frame family.
Direct mutation is unsupported; public state getters return detached snapshots.
String channel lookup always means a channel name. Use `frame.channels.by_id()`
for explicit stable-ID lookup.

`ChannelCalibration.with_unit()` preserves the factor and resets `ref` to the
new unit's default. Chain `.with_unit(...).with_ref(...)` for a custom reference.

::: wandas.frames.channel.ChannelFrame

### Combining channels / チャンネルの結合

Use `frame.add_channel(array, ...)` to append exactly one channel from a NumPy or
Dask array. Use `frame.concat_frame(other, ...)` to append every channel from another
`ChannelFrame` while preserving its channel metadata, calibration, and source-time
offsets. `add_channel(ChannelFrame)` is no longer supported; migrate its `label=`
prefix to `concat_frame(..., label_prefix=...)`.

NumPyまたはDask配列から1チャンネルを追加する場合は
`frame.add_channel(array, ...)`を使用します。別の`ChannelFrame`の全チャンネルを、
channel metadata、calibration、source-time offsetとともに追加する場合は
`frame.concat_frame(other, ...)`を使用します。`add_channel(ChannelFrame)`は
サポートされないため、従来の`label=` prefixは
`concat_frame(..., label_prefix=...)`へ移行してください。

Recipe replay for `wandas.channel.add_channel` version 1 is no longer supported.
Recreate saved workflows with `add_channel()` version 2 when the second input is a
raw array, or with `concat_frame()` version 1 when it is a `ChannelFrame`.

`wandas.channel.add_channel` version 1 Recipeの再生互換性は終了しました。
第2入力がraw arrayなら`add_channel()` version 2、`ChannelFrame`なら
`concat_frame()` version 1を使用して、保存済みworkflowを作り直してください。

Use the <a href="../learning-path/07_per_channel_calibration.html">per-channel
calibration learning app</a> to configure known conversion factors without
modifying the source frame. Calibrated physical values are available from
`frame.data` as a NumPy array; users do not need to manage the internal array
backend.

For a recorded reference event, use exactly one known scalar target:

```python
microphone_calibration = microphone_reference.derive_calibration(
    target_level=94.0,
    unit="Pa",
)
acceleration_calibration = acceleration_reference.derive_calibration(
    target_rms=1.0,
    unit="m/s^2",
)
measurement = measurement.with_calibration(
    {**microphone_calibration, **acceleration_calibration}
)
```

The scalar is broadcast to every channel in that reference event. References
with different targets, units, or recording times are separate events whose
label mappings can be combined. Derivation always uses the current
`frame.data`/`frame.rms`, including an existing factor, and does not inspect or
change operation history. It requires unique non-empty labels. The reference
and measurement recording chain—including amplifier gain—must represent the
same physical scale.

### RMS, pressure, and level quantities

These APIs return different quantities:

| API | Quantity | Unit/reference |
| --- | --- | --- |
| `frame.rms` | one linear RMS amplitude per channel | calibrated channel unit; Pa only for a pressure channel |
| `frame.rms_trend(dB=False)` | centered-window linear RMS | calibrated channel unit |
| `frame.rms_trend(dB=True)` | `20 log10(window_rms / channel_ref)` | dB relative to each channel reference |
| `frame.sound_level(..., dB=False)` | frequency-weighted, exponentially time-weighted RMS | calibrated channel unit |
| `frame.sound_level(..., dB=True)` | `10 log10(smoothed_power / channel_ref²)` | dB relative to each channel reference |

A Pa channel defaults to the reference pressure `2e-5 Pa`, so its dB results are
dB SPL. An uncalibrated channel uses reference 1 and therefore produces relative
dB re 1 input unit, not dB SPL. Inspect the public
`frame.channels[index].unit` and `frame.channels[index].ref` views when labeling
output. `rms_plot()` plots the reference-relative dB form; it is not the scalar
linear `frame.rms` property.

Frequency weighting and time weighting are separate choices. The implemented
A/C/Z curves and Fast (125 ms)/Slow (1 s) exponential time constants are tested
against their numerical formulas. Wandas does not currently claim the complete
instrument tolerances, detector behavior, calibration, or directional response
required for IEC/JIS sound-level-meter conformance.

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

FFT, STFT, and Welch results use the canonical amplitude, unit, normalization,
and decibel definitions in the
[spectral numerical contracts](../explanation/spectral-numerical-contracts.md).
In particular, `welch()` returns a Welch-averaged amplitude spectrum, not PSD.

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

Its stored values are per-band RMS amplitudes in the input unit; see the
[spectral numerical contracts](../explanation/spectral-numerical-contracts.md)
for the level reference and `G` convention.

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
