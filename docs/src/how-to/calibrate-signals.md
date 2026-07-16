# Configure Per-Channel Calibration / チャンネルごとの校正値を設定する

このガイドは、センサの証明書、メーカー仕様、設備台帳、CSVなどに記録された
**既知のraw-to-physical係数**を信号へ設定する方法を説明します。校正信号を全チャンネルで
同時収録することは前提にしません。Wandasが校正信号から係数を推定する手順も、このAPIの
責務ではありません。

`ChannelFrame.with_calibration()`は、現在のチャンネルに係数を対応付けた新しいFrameを返します。
保存されているサンプルはrawのままで、`compute()`、`data`、`rms`、`fft()`、`stft()`、
`sound_level()`などの数値処理だけが`raw * factor`を使います。

## 音と加速度を設定する

マイクと加速度計の2ch収録を例にします。2つのセンサは単位も基準値も異なるため、最初の設定では
`ChannelCalibration`を使って係数、単位、基準値をまとめて指定します。

```python exec="on" session="calibration_howto"
import numpy as np
import wandas as wd

raw = wd.from_numpy(
    np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [0.1, 0.2, 0.3, 0.4],
        ]
    ),
    sampling_rate=8_000,
    ch_labels=["microphone", "accelerometer"],
)

configured = raw.with_calibration(
    [
        wd.ChannelCalibration(factor=0.02, unit="Pa"),
        wd.ChannelCalibration(factor=9.81, unit="m/s^2", ref=1.0),
    ]
)

np.testing.assert_allclose(
    configured.compute(),
    raw.raw_data.compute() * np.array([[0.02], [9.81]]),
)
```

リストは**現在のチャンネル順に完全置換**します。長さがチャンネル数と一致しなければエラーです。
`Pa`の`ref`は省略すると`2e-5`になります。加速度のように別の基準値を使う場合は明示します。

## 係数だけを更新する

単位と基準値がすでに正しければ、数値だけを渡せます。数値は係数だけを置き換え、現在の
`unit`と`ref`を維持します。前の係数とは掛け合わせません。

```python exec="on" session="calibration_howto"
reissued_certificate = configured.with_calibration([0.025, 9.75])

assert reissued_certificate.channels[0].unit == "Pa"
assert reissued_certificate.channels[1].unit == "m/s^2"
assert reissued_certificate.channels[0].calibration.factor == 0.025
assert configured.channels[0].calibration.factor == 0.02
```

`configured`は変更されません。更新後の値をrawに掛けるため、`0.02 * 0.025`のような累積も
起きません。

## ラベルまたは位置で部分更新する

辞書は指定したチャンネルだけを更新します。長期運用やチャンネル数が多い設備では、順序に依存しない
ラベル指定が基本です。

```python exec="on" session="calibration_howto"
by_label = configured.with_calibration({"accelerometer": 9.75})

# 呼出し時点の位置も使用できる（負の位置を含む）
by_index = configured.with_calibration({0: 0.025, -1: 9.75})

assert by_label.channels[0].calibration.factor == 0.02
assert by_label.channels[1].calibration.factor == 9.75
assert [ch.calibration.factor for ch in by_index.channels] == [0.025, 9.75]
```

ラベルと位置は1つの辞書に混在できます。ただし、たとえば`"microphone"`と`0`が同じchを
指す場合は重複としてエラーになります。位置はその呼出し時の順序なので、保存データやRecipeで
長く使う設定にはラベルを選んでください。

## CSVの管理表を使う

CSV専用APIは不要です。CSVの各行を`ラベル -> ChannelCalibration`の辞書へ変換し、同じ
`with_calibration()`へ渡します。この形なら、CSV、データベース、設備管理APIのどれを使っても
Frame側の契約は変わりません。

```python exec="on" session="calibration_howto"
import csv
import io

calibration_csv = io.StringIO(
    "channel,factor,unit,ref\n"
    "microphone,0.021,Pa,0.00002\n"
    "accelerometer,9.79,m/s^2,1.0\n"
)

calibration_by_label = {
    row["channel"]: wd.ChannelCalibration(
        factor=float(row["factor"]),
        unit=row["unit"],
        ref=float(row["ref"]),
    )
    for row in csv.DictReader(calibration_csv)
}

from_csv = raw.with_calibration(calibration_by_label)
assert [ch.calibration.factor for ch in from_csv.channels] == [0.021, 9.79]
```

CSVに存在しないラベル、重複して同じchを指すキー、不正な係数は設定時に失敗します。
空の辞書も「何もしなかった」ことを隠さずエラーになります。

## 100chを設定する

100chでもAPIは同じです。全chを更新するときは生成したリストまたはラベル辞書を1回渡します。
一部の証明書だけ更新された場合は、そのラベルだけの辞書を渡します。chごとのメソッド呼出しは
必要ありません。

```python exec="on" session="calibration_howto"
channel_count = 100
labels = [f"sensor-{index:03d}" for index in range(channel_count)]
hundred = wd.from_numpy(
    np.zeros((channel_count, 16)),
    sampling_rate=8_000,
    ch_labels=labels,
)

all_factors = [1.0 + index / 1_000 for index in range(channel_count)]
configured_hundred = hundred.with_calibration(all_factors)

replaced_every_tenth = configured_hundred.with_calibration(
    {f"sensor-{index:03d}": 2.0 for index in range(0, channel_count, 10)}
)

assert configured_hundred.n_channels == 100
assert replaced_every_tenth.channels[90].calibration.factor == 2.0
assert replaced_every_tenth.channels[91].calibration.factor == all_factors[91]
```

リストも辞書もch数に対して線形に検証され、Dask配列はこの時点では計算されません。100chを
人手で並べるのではなく、証明書や管理表から値を生成し、ラベル集合をレビューできる形にするのが
実用的です。

## raw値、遅延実行、派生Frameを確認する

`raw_data`は保存されている未校正のDask配列です。`compute()`と`data`は校正後の値を返します。
`with_calibration()`は係数をメタデータへ設定してDaskグラフを組むだけで、計算を開始しません。

```python exec="on" session="calibration_howto"
import dask.array as da

assert isinstance(configured.raw_data, da.Array)
np.testing.assert_array_equal(configured.raw_data.compute(), raw.raw_data.compute())

spectrum = configured.fft(n_fft=4, window="boxcar")
assert isinstance(spectrum.raw_data, da.Array)
assert [ch.calibration.factor for ch in spectrum.channels] == [1.0, 1.0]
```

FFTなどの派生Frameは、その時点の`raw * factor`から計算されます。派生データ自体へ同じ係数を
もう一度掛けないよう、派生Frameの係数は`1.0`に戻ります。後から元Frameの校正値を置き換えても、
すでに作った派生Frameの意味は変わりません。新しい値で解析したい場合は、置換後のFrameから
FFTなどを作り直します。

## 履歴、Recipe、WDFで再現する

校正設定は`operation_history`へ1操作として記録されます。Recipeには一時的な位置ではなく、
解決済みの安定ch IDと`factor`、`unit`、`ref`のスナップショットが入るため、直前にchを
並べ替えたワークフローも同じ対応関係で再実行できます。

```python exec="on" session="calibration_howto"
from wandas.pipeline import RecipePlan

workflow = raw.get_channel([1, 0]).with_calibration(
    [
        wd.ChannelCalibration(9.81, "m/s^2", 1.0),
        wd.ChannelCalibration(0.02, "Pa"),
    ]
)
plan = RecipePlan.from_frame(workflow, input_names=("signal",))
replayed = plan.apply({"signal": raw})

assert replayed.labels == ["accelerometer", "microphone"]
np.testing.assert_allclose(replayed.compute(), workflow.compute())
```

WDFへ保存するときは、rawサンプルと現在の校正メタデータを別々に保持します。読込後の
`raw_data`は元のraw値、`compute()`は同じ校正後の値になります。WDFの履歴は表示用の
`operation_history`として復元され、実行可能なRecipeを保存したい場合は`RecipePlan.to_dict()`を
別途保存します。
