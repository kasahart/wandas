# Unreleased changes

Changes after [Wandas 0.12.0](v0.12.0.md).

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
