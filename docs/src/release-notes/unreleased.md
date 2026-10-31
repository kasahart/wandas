# Unreleased changes

Changes after [Wandas 0.12.0](v0.12.0.md).

## Added / 追加

- `trim()` selects by time on more frames and in more units
  ([#603](https://github.com/kasahart/wandas/issues/603)). Times are seconds or
  strings with the units `h`, `min`, `s` and `ms`, such as
  `recording.trim("1min30s", "2min")`. `reference="source"` measures them on the
  source timeline instead of from the frame start. `SpectrogramFrame.trim()` and
  `CepstrogramFrame.trim()` keep the frames whose center lies in the half-open
  range, and `SpectrogramFrame.at_time(12.3)` returns the spectrum of the frame
  whose center is nearest, with an optional `tolerance`. All of them are Recipe
  steps that record the requested times, so a plan selects the same seconds on
  files with other sampling rates.
  `trim()`で秒のほか`h`・`min`・`s`・`ms`付きの文字列（`recording.trim("1min30s", "2min")`など）を
  指定できるようになり、`reference="source"`で元の音源の時刻を使えます。`SpectrogramFrame.trim()`と
  `CepstrogramFrame.trim()`は中心が半開区間に入るフレームを残し、`SpectrogramFrame.at_time(12.3)`は
  中心が最も近いフレームのスペクトルを返します（`tolerance`も指定可）。どれもRecipeの段で、
  指定した時刻を記録するため、サンプリング周波数が違うファイルでも同じ秒数を選びます。

## Changed / 変更

- The built-in Recipe operations are now collected from the Frame types that
  `wandas.frames` exports (with their base classes and mixins) instead of a
  hand-written list, so a new public Frame or mixin needs no extra registration
  ([#608](https://github.com/kasahart/wandas/issues/608)). For each Frame type,
  `default_recipe_registry().describe()` lists its operations in the same order
  as before; only the pairwise-frame operations move after the spectrogram ones
  in the full catalog.
  組み込みRecipe operationは手書きの一覧ではなく、`wandas.frames`が公開するFrame型と
  そのbase class・mixinから集めるようになりました。新しい公開Frameやmixinは追加登録が要りません。
  Frame型ごとの`default_recipe_registry().describe()`の並びは変わらず、全体の一覧で
  pairwise系Frameのoperationがspectrogram系の後ろへ移るだけです。
- The eagerly imported `wandas.processing` modules register every Operation
  class they define with its own `name`, instead of a hand-written list at the
  end of each module, and a new test fails when a built-in Operation class
  cannot be resolved by its name
  ([#610](https://github.com/kasahart/wandas/issues/610)). The registered
  operations are unchanged.
  eager importされる`wandas.processing`の各モジュールは、末尾の手書きの一覧ではなく、
  自分の`name`を持つOperation classをすべて登録するようになりました。名前で解決できない
  組み込みOperation classがあると新しいテストが失敗します。登録されるoperationは変わりません。
- `ChannelFrame.trim()` rounds times to the nearest sample (halves up) instead of
  truncating them, so `trim(0.29)` at 100 Hz starts at sample 29, not 28
  (`0.29 * 100` is `28.999...` in floating point). The Recipe step
  `wandas.frame.time_slice` is now version 2; saved version 1 plans replay with
  the old truncation.
  `ChannelFrame.trim()`は時刻をサンプルへ切り捨てずに四捨五入するようになりました。100 Hzで
  `trim(0.29)`は28ではなく29番目のサンプルから始まります（浮動小数点では`0.29 * 100`が`28.999...`）。
  Recipeの段`wandas.frame.time_slice`はversion 2になり、保存済みのversion 1は従来の切り捨てで再実行します。

## Fixed / 修正

- Slicing an `NOctFrame` along the band axis now labels the selected bands
  with their own center frequencies
  ([#600](https://github.com/kasahart/wandas/issues/600)). Before, `freqs`
  restarted from the original lowest band, so `noct[:, 3:6]` of a
  100 Hz–4 kHz spectrum was labelled 100/125/160 Hz instead of
  200/250/315 Hz, and `dBA`, `plot()`, `to_dataframe()` and saved WDF files
  used those wrong labels. The slice now moves `fmin`/`fmax` to the first and
  last selected bands. A stepped band slice (`noct[:, ::2]`) or an empty one
  now raises `ValueError`, because `fmin`/`fmax` can describe only a
  contiguous run of bands. A WDF file saved from a band slice before this fix
  keeps its wrong labels, because it does not store which bands were selected;
  recompute it from the source.
  `NOctFrame`をバンド方向にスライスしたとき、選んだバンドに正しい中心周波数が付くようになりました。
  これまでは`freqs`が元の最低バンドから数え直されていたため、100 Hz〜4 kHzのスペクトルの
  `noct[:, 3:6]`に200/250/315 Hzではなく100/125/160 Hzが付き、`dBA`、`plot()`、
  `to_dataframe()`、保存したWDFファイルもその誤ったラベルを使っていました。スライスは
  `fmin`/`fmax`を選んだ最初と最後のバンドへ移します。`fmin`/`fmax`は連続したバンドしか表せないため、
  ステップ付きのスライス（`noct[:, ::2]`）と空のスライスは`ValueError`になります。
  この修正より前にバンドをスライスして保存したWDFファイルは、どのバンドを選んだかを保存していないため
  誤ったラベルのままです。元のデータから計算し直してください。

## Compatibility / 互換性

| Affected surface or artifact | Classification | Deprecation start | Replacement or migration | Removal/change version | Exception reason and decision link |
| --- | --- | --- | --- | --- | --- |
| Replay of `wandas.frame.index` (v1) Recipes that slice an `NOctFrame` along the band axis | Serialized | `None` | None needed: contiguous slices now replay with correct band labels; rewrite a stepped band slice as separate contiguous slices | Next release | Wrong labels become correct (values unchanged) and stepped slices raise `ValueError`; [decision](https://github.com/kasahart/wandas/issues/600#issuecomment-6514808419) |
