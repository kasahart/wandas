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

- `ChannelFrame.trim()` rounds times to the nearest sample (halves up) instead of
  truncating them, so `trim(0.29)` at 100 Hz starts at sample 29, not 28
  (`0.29 * 100` is `28.999...` in floating point). The Recipe step
  `wandas.frame.time_slice` is now version 2; saved version 1 plans replay with
  the old truncation.
  `ChannelFrame.trim()`は時刻をサンプルへ切り捨てずに四捨五入するようになりました。100 Hzで
  `trim(0.29)`は28ではなく29番目のサンプルから始まります（浮動小数点では`0.29 * 100`が`28.999...`）。
  Recipeの段`wandas.frame.time_slice`はversion 2になり、保存済みのversion 1は従来の切り捨てで再実行します。
