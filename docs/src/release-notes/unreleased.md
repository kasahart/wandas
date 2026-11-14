# Unreleased changes

Changes after [Wandas 0.13.1](v0.13.1.md).

0.13.1以降の変更です。

## Changed / 変更

- The time-varying psychoacoustic results carry the unit of the metric instead
  of the input channel's unit
  ([#652](https://github.com/kasahart/wandas/issues/652)): `loudness_zwtv()`
  is in `sone`, `roughness_dw()` in `asper`, `roughness_dw_spec()` in
  `asper/Bark` and `sharpness_din()` in `acum`, each with reference 1.0.
  Before, a loudness trace of a channel in `Pa` was still labelled `Pa` with
  reference 2e-05. The values do not change. New Recipes use version 2 of
  these four operations; Recipes saved by 0.13.1 (version 1) still replay with
  the input unit.
  時間変化する心理音響指標の結果は、入力チャンネルの単位ではなく指標の単位を持つようになりました。
  `loudness_zwtv()`は`sone`、`roughness_dw()`は`asper`、`roughness_dw_spec()`は`asper/Bark`、
  `sharpness_din()`は`acum`で、基準値はどれも1.0です。これまでは`Pa`のチャンネルから求めた
  ラウドネスも`Pa`・基準値2e-05のままでした。値は変わりません。新しいRecipeはこの4つの操作の
  version 2を使い、0.13.1で保存したRecipe（version 1）は今までどおり入力の単位で再生します。
