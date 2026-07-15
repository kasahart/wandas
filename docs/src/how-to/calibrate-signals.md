# Calibrate Signals from a Known Reference / 既知の基準信号から校正する

Use a recorded calibration signal to convert raw samples, volts, or counts into a
physical quantity such as sound pressure in Pa. Wandas separates the workflow into
two explicit steps:

1. `calibration_signal.derive_calibration(...)` computes an immutable calibration.
2. `measurement.calibrate(calibration)` applies its factors to another frame lazily.

収録した校正信号を使い、raw sample、電圧、countなどをPaの音圧をはじめとする物理量へ
変換できます。Wandasでは処理を次の2段階に分けます。

1. `calibration_signal.derive_calibration(...)`で不変な校正値を計算する。
2. `measurement.calibrate(calibration)`で別のframeへ倍率を遅延適用する。

## Calibrate sound pressure / 音圧を校正する

For an acoustic calibrator whose nominal level is 94 dB SPL, record a steady section
through the same microphone, preamplifier, input range, and digital gain used for the
measurement. Then derive and apply the calibration:

公称94 dB SPLの音響校正器では、測定時と同じマイク、プリアンプ、入力レンジ、デジタルゲインを
通した定常区間を収録し、その信号から校正値を作って適用します。

```python
import wandas as wd

calibration_tone = wd.read("calibrator-94db.wav").remove_dc()
measurement = wd.read("measurement.wav")

calibration = calibration_tone.derive_calibration(
    target_level=94.0,
    unit="Pa",
)
pressure = measurement.calibrate(calibration)

print(calibration.measured_rms)  # RMS recorded from each calibration channel
print(calibration.target_rms)    # physical RMS represented by the calibrator
print(calibration.factors)       # Pa per input sample, one value per channel

level = pressure.sound_level(
    freq_weighting="A",
    time_weighting="Fast",
    dB=True,
)
```

For `unit="Pa"`, the omitted `ref` defaults to 20 µPa. Wandas converts the known
amplitude level to physical RMS using

```text
target_rms = ref * 10 ** (target_level / 20)
factor     = target_rms / measured_rms
```

The exact 94 dB target is approximately 1.0024 Pa RMS; 1 Pa corresponds to about
93.98 dB re 20 µPa. Pass the calibrator's stated level rather than rounding the
physical target yourself.

`unit="Pa"`で`ref`を省略すると20 µPaが使われます。既知の振幅レベルから上式で物理RMSを
求め、校正信号で測ったRMSとの比を倍率にします。厳密な94 dBは約1.0024 Pa RMSで、
1 Paは20 µPa基準で約93.98 dBです。物理値を丸めず、校正器に記載されたレベルを指定します。

## Use a known RMS for other quantities / 他の物理量を既知RMSで校正する

When the reference source states a physical RMS directly, use `target_rms`. For
example, a vibration reference representing 10 m/s² RMS can calibrate raw sensor
samples as follows:

基準源が物理RMSを直接示す場合は`target_rms`を使います。たとえば10 m/s² RMSの振動基準は
次のようにraw sensor sampleへ対応付けられます。

```python
reference_vibration = wd.read("reference-vibration.csv")
raw_vibration = wd.read("measurement.csv")

calibration = reference_vibration.derive_calibration(
    target_rms=10.0,
    unit="m/s^2",
)
acceleration = raw_vibration.calibrate(calibration)
```

For units other than Pa, the default level reference is 1.0. Supply `ref=` explicitly
when the domain uses another reference.

Pa以外の単位ではlevel基準値の既定値は1.0です。別の基準値を使う領域では`ref=`を明示します。

## Align calibration channels / 校正チャンネルを対応させる

- A one-channel calibration broadcasts its factor to every measurement channel.
- A multi-channel calibration must have exactly the same number of channels as the
  measurement; factors are matched by channel position.
- A scalar `target_rms` or `target_level` broadcasts across the calibration channels.
  A sequence must contain one target per calibration channel.

- 1チャンネルの校正値は、測定信号の全チャンネルへ同じ倍率を適用します。
- 複数チャンネルの校正値は測定信号とチャンネル数が一致する必要があり、位置順に対応します。
- scalarの`target_rms`または`target_level`は校正信号の全チャンネルへ展開します。sequenceでは
  校正チャンネルごとに1値を指定します。

Use separate per-channel calibration recordings when sensors have different
sensitivities. Reorder or select channels explicitly before deriving and applying
calibration if the acquisition layouts differ.

センサー感度が異なる場合はチャンネルごとの校正信号を使います。校正収録と測定収録で
チャンネル配置が異なる場合は、校正値の導出・適用前に明示的に並べ替えまたは選択します。

## Understand computation, metadata, and history / 計算・メタデータ・履歴を理解する

`derive_calibration()` is an explicit eager reduction: it computes one RMS scalar per
calibration channel. The returned `Calibration` is frozen and keeps `measured_rms`,
`target_rms`, `unit`, and `ref`; `factors` are derived from those authoritative values.
Derivation does not modify the calibration frame or add frame lineage.

`derive_calibration()`は明示的なeager縮約で、校正チャンネルごとにRMS scalarを計算します。
返される`Calibration`は不変で、`measured_rms`、`target_rms`、`unit`、`ref`を保持し、
`factors`はそれらの正本から導出されます。導出は校正frameを変更せず、frame lineageも追加しません。

`calibrate()` does not compute the measurement samples. It returns a new Dask-backed
`ChannelFrame`, preserves frame metadata, channel extras, sampling rate, channel IDs,
and `source_time_offset`, and atomically replaces each channel's `unit` and `ref`.
The operation history records the calibration's measured and target RMS values, and
the operation can be extracted and replayed through `RecipePlan`.

`calibrate()`は測定sampleを計算せず、新しいDask-backed `ChannelFrame`を返します。frame
metadata、channel extra、sampling rate、channel ID、`source_time_offset`を維持し、各channelの
`unit`と`ref`を一度に更新します。操作履歴には校正の測定RMSと目標RMSが残り、`RecipePlan`で
抽出・再実行できます。

A replayed Recipe applies the captured calibration values; it does not reread or
recompute the original calibration recording. Derive a new calibration whenever the
sensor or acquisition gain changes.

Recipeの再実行では記録済みの校正値を適用し、元の校正収録を再読込・再計算しません。センサーや
収録ゲインが変わった場合は、新しい校正値を導出してください。

## Avoid invalidating physical amplitude / 物理振幅を失わない

Choose a steady calibration interval and remove a known DC offset if necessary before
deriving calibration. Do not call `normalize()` on the calibration signal or on a
measurement used for SPL or other absolute physical metrics: normalization changes
the amplitude that calibration is meant to preserve. Trimming is safe; filters must
be included only when the same transfer function is intentionally part of both the
calibration and measurement paths.

校正値の導出には定常区間を選び、必要なら既知のDC offsetを先に除去します。SPLなど絶対物理量を
求める校正信号・測定信号には`normalize()`を使わないでください。正規化は校正で保持すべき振幅を
変えます。trimは利用できますが、filterは同じ伝達関数を校正経路と測定経路の両方へ意図的に
含める場合だけ使います。
