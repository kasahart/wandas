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
  `"none"` makes the dB properties raise. Spectral, spectrogram and band
  `dB`, `LevelReference.to_level()` and WDF files follow it. RMS-based levels
  (`rms_trend(dB=True)`, `sound_level()`, `leq()`, `percentile_levels()`)
  raise for channels that are not amplitude quantities. The results of those
  dB operations are marked `"none"`, since their values are already levels.
  Existing frames, Recipes and WDF files describe amplitude channels and
  their levels do not change.
  チャンネルの校正に、値をレベルへ直す方法を持たせました（`level="amplitude" | "power" | "none"`）。
  振幅量（既定、Pa・Vなど）は今までどおり`20 * log10(|x| / ref)`、パワー量（Pa^2・Wなど）は
  `10 * log10(|x| / ref)`で、`"none"`ではdBの属性がエラーになります。スペクトル・スペクトログラム・
  帯域の`dB`、`LevelReference.to_level()`、WDFがこれに従います。RMSから求めるレベル
  （`rms_trend(dB=True)`・`sound_level()`・`leq()`・`percentile_levels()`）は振幅量以外ではエラーになり、
  これらのdB操作の結果は既にレベルなので`"none"`になります。既存のFrame・Recipe・WDFは振幅量を表し、
  レベルの値は変わりません。

## Changed / 変更

- The time-varying psychoacoustic results carry the unit of the metric instead
  of the input channel's unit
  ([#652](https://github.com/kasahart/wandas/issues/652)): `loudness_zwtv()`
  is in `sone`, `roughness_dw()` in `asper`, `roughness_dw_spec()` in
  `asper/Bark` and `sharpness_din()` in `acum`, each with reference 1.0 and
  level kind `"none"`, so their dB properties raise.
  Before, a loudness trace of a channel in `Pa` was still labelled `Pa` with
  reference 2e-05. The values do not change. New Recipes use version 2 of
  these four operations; Recipes saved by 0.13.1 (version 1) still replay with
  the input unit.
  時間変化する心理音響指標の結果は、入力チャンネルの単位ではなく指標の単位を持つようになりました。
  `loudness_zwtv()`は`sone`、`roughness_dw()`は`asper`、`roughness_dw_spec()`は`asper/Bark`、
  `sharpness_din()`は`acum`で、基準値はどれも1.0、レベルの種類は`"none"`（dBはエラー）です。これまでは`Pa`のチャンネルから求めた
  ラウドネスも`Pa`・基準値2e-05のままでした。値は変わりません。新しいRecipeはこの4つの操作の
  version 2を使い、0.13.1で保存したRecipe（version 1）は今までどおり入力の単位で再生します。
