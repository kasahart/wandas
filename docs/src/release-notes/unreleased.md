# Unreleased changes

Changes after [Wandas 0.13.1](v0.13.1.md).

0.13.1以降の変更です。

## Added / 追加

- A channel calibration records how its values convert to a level:
  `ChannelCalibration(..., level="amplitude" | "power" | "none")`, read as
  `frame.channels[i].calibration.level`
  ([#651](https://github.com/kasahart/wandas/issues/651)). Amplitude
  quantities (the default, such as Pa or V) keep `20 * log10(|x| / ref)`,
  power quantities (such as Pa^2 or W) use `10 * log10(|x| / ref)`, and
  `"none"` makes the dB properties warn and use the amplitude formula. Spectral, spectrogram and band
  `dB`, `LevelReference.to_level()` and WDF files follow it. RMS-based levels
  (`rms_trend(dB=True)`, `sound_level()`, `leq()`, `percentile_levels()`)
  warn for channels that are not amplitude quantities and compute as before. The results of those
  dB operations are marked `"none"`, since their values are already levels.
  Existing frames, Recipes and WDF files describe amplitude channels and
  their levels do not change.
  チャンネルの校正に、値をレベルへ直す方法を持たせました（`level="amplitude" | "power" | "none"`）。
  振幅量（既定、Pa・Vなど）は今までどおり`20 * log10(|x| / ref)`、パワー量（Pa^2・Wなど）は
  `10 * log10(|x| / ref)`で、`"none"`ではdBの属性が警告を出して振幅の式で計算します。スペクトル・スペクトログラム・
  帯域の`dB`、`LevelReference.to_level()`、WDFがこれに従います。RMSから求めるレベル
  （`rms_trend(dB=True)`・`sound_level()`・`leq()`・`percentile_levels()`）は振幅量以外では警告を出して今までどおり計算し、
  これらのdB操作の結果は既にレベルなので`"none"`になります。既存のFrame・Recipe・WDFは振幅量を表し、
  レベルの値は変わりません。

## Changed / 変更

- Element-wise arithmetic derives the unit, reference and level kind of its
  result instead of keeping the left operand's
  ([#651](https://github.com/kasahart/wandas/issues/651)). `p * p`, `p ** 2`,
  `p.power(2)` and `np.square(p)` of a channel in `Pa` give `Pa^2` with
  reference `4e-10` and a power level (`10 * log10`), so their level in dB
  equals the level of `p`; `p * v` gives `Pa*m/s`, `p / v` gives `Pa*s/m`,
  `p / p` gives `1`, `2 / p` gives `1/Pa`, `np.sqrt(p * p)` gives `Pa` again,
  and `np.log10(p)`, `np.exp(...)` and the trigonometric ufuncs give `1` with
  no level. A dimensionless channel (unit `1`), such as a window, acts as a
  gain and keeps the other operand's unit. Adding or subtracting frames whose
  units differ keeps the left unit as before but now emits a `UserWarning`.
  Units are read as products of factors with exponents, so `Pa*Pa` and `Pa^2`
  are the same unit; text that is not such a product, such as
  `dB SPL re 2e-05 Pa`, is kept as one factor, and an uncalibrated channel
  (unit `""`) stays uncalibrated. Values do not change. New Recipes use
  version 2 of the `wandas.operator.*` operations, `wandas.numpy.ufunc` and
  `wandas.audio.power`; Recipes saved by 0.13.1 (version 1) still replay with
  the left operand's unit.
  要素ごとの演算の結果に、左の被演算子の単位ではなく演算から決まる単位・基準値・レベルの種類を
  付けるようになりました。`Pa`のチャンネルの`p * p`・`p ** 2`・`p.power(2)`・`np.square(p)`は
  基準値`4e-10`の`Pa^2`でパワー量（`10 * log10`）になり、dBのレベルは`p`と同じです。
  `p * v`は`Pa*m/s`、`p / v`は`Pa*s/m`、`p / p`は`1`、`2 / p`は`1/Pa`、`np.sqrt(p * p)`は`Pa`に戻り、
  `np.log10(p)`・`np.exp(...)`・三角関数は`1`（レベルなし）です。単位`1`のチャンネル（窓など）は
  ゲインとして働き、相手の単位を保ちます。単位の違うFrameどうしの加減算は今までどおり左の単位を
  保ちますが、`UserWarning`を出すようになりました。単位は因子と指数の積として読むので`Pa*Pa`と
  `Pa^2`は同じ単位です。`dB SPL re 2e-05 Pa`のように積として読めない文字列は1つの因子として扱い、
  校正していないチャンネル（単位`""`）は校正していないままです。値は変わりません。新しいRecipeは
  `wandas.operator.*`・`wandas.numpy.ufunc`・`wandas.audio.power`のversion 2を使い、0.13.1で保存した
  Recipe（version 1）は今までどおり左の単位で再生します。
- `csd()` and `transfer_function()` write their units in the same canonical
  form ([#651](https://github.com/kasahart/wandas/issues/651)): `Pa*Pa`
  becomes `Pa^2`, `Pa*Pa/Hz` becomes `Pa^2/Hz`, and the transfer function of
  `m/s` to `Pa` becomes `Pa*s/m` instead of `Pa/m/s`, which read as
  Pa/(m·s). References and values do not change. WDF files written by 0.13.1
  load with the new unit text.
  `csd()`と`transfer_function()`の単位も同じ正規の書き方になりました。`Pa*Pa`は`Pa^2`、
  `Pa*Pa/Hz`は`Pa^2/Hz`、`m/s`から`Pa`への伝達関数は`Pa/m/s`（Pa/(m·s)と読めてしまう）ではなく
  `Pa*s/m`です。基準値と値は変わりません。0.13.1で保存したWDFは新しい単位の表記で読み込みます。

- The time-varying psychoacoustic results carry the unit of the metric instead
  of the input channel's unit
  ([#652](https://github.com/kasahart/wandas/issues/652)): `loudness_zwtv()`
  is in `sone`, `roughness_dw()` in `asper`, `roughness_dw_spec()` in
  `asper/Bark` and `sharpness_din()` in `acum`, each with reference 1.0 and
  level kind `"none"`, so their dB properties warn.
  Before, a loudness trace of a channel in `Pa` was still labelled `Pa` with
  reference 2e-05. The values do not change. New Recipes use version 2 of
  these four operations; Recipes saved by 0.13.1 (version 1) still replay with
  the input unit.
  時間変化する心理音響指標の結果は、入力チャンネルの単位ではなく指標の単位を持つようになりました。
  `loudness_zwtv()`は`sone`、`roughness_dw()`は`asper`、`roughness_dw_spec()`は`asper/Bark`、
  `sharpness_din()`は`acum`で、基準値はどれも1.0、レベルの種類は`"none"`（dBは警告）です。これまでは`Pa`のチャンネルから求めた
  ラウドネスも`Pa`・基準値2e-05のままでした。値は変わりません。新しいRecipeはこの4つの操作の
  version 2を使い、0.13.1で保存したRecipe（version 1）は今までどおり入力の単位で再生します。

## Compatibility and migration / 互換性と移行

| Affected surface or artifact | Classification | Deprecation start | Replacement or migration | Removal/change version | Exception reason and decision link |
| --- | --- | --- | --- | --- | --- |
| Unit text of `csd()` and `transfer_function()` results, including replays of Recipes saved by 0.13.1 (`wandas.audio.csd` and `wandas.audio.transfer_function` version 2, `wandas.spectrogram.csd`, `wandas.cross_spectral.transfer_function`, `wandas.segments.transfer_function`) | Stable / Serialized, unit text change without a new operation version | None | Compare units with the canonical form (`Pa^2`, `Pa*s/m`); values, references and level kinds are unchanged, and WDF files written by 0.13.1 load with the new text | Next release | Pairwise frames check their unit against one derivation, so replaying the released text would need two spellings; the released `Pa/m/s` also reads as Pa/(m·s). Maintainer decision on [#656](https://github.com/kasahart/wandas/pull/656) |

| 影響する機能・データ | 分類 | 非推奨の開始 | 代替・移行方法 | 変更バージョン | 例外の理由と判断のリンク |
| --- | --- | --- | --- | --- | --- |
| `csd()`・`transfer_function()`の結果の単位の文字列（0.13.1で保存したRecipe、`wandas.audio.csd`・`wandas.audio.transfer_function`のversion 2、`wandas.spectrogram.csd`、`wandas.cross_spectral.transfer_function`、`wandas.segments.transfer_function`の再生を含む） | Stable / Serialized、操作versionを上げない単位の表記の変更 | なし | 単位は正規の書き方（`Pa^2`、`Pa*s/m`）で比べてください。値・基準値・レベルの種類は変わらず、0.13.1で保存したWDFは新しい表記で読み込みます | 次のリリース | ペアのFrameは単位を1つの導出と照合するため、旧表記で再生するには表記を2通り持つ必要があります。また旧表記`Pa/m/s`はPa/(m·s)と読めます。判断は[#656](https://github.com/kasahart/wandas/pull/656)でメンテナーが行いました |
