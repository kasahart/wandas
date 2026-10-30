# Unreleased changes

Changes after [Wandas 0.12.0](v0.12.0.md).

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
- The A, B and C frequency-weighting filters (`a_weighting()`,
  `sound_level(freq_weighting="A" | "C")`, `rms_trend(Aw=True)`, `leq()` and
  `percentile_levels()`) follow the analog IEC 61672-1 curve up to
  0.9 × Nyquist within about 0.2 dB from 8 kHz to 192 kHz
  ([#607](https://github.com/kasahart/wandas/issues/607)). The previous
  design, vendored from waveform-analysis, ran the whole analog filter through
  the bilinear transform and attenuated too much near Nyquist: at 48 kHz it
  was 1.2 dB low at 10 kHz, 6.4 dB at 16 kHz and 15.8 dB at 20 kHz, and
  A-weighted white noise read 0.87 dB low. The low-frequency poles still use
  the bilinear transform; the 12.2 kHz double pole is matched and two zeros
  are fitted to the analog curve by linear least squares. The filter stays
  causal and runs at the input sampling rate. These numbers change, so new
  Recipes use `a_weighting` version 2 and `rms_trend`/`sound_level` version
  3; Recipes saved by 0.12.0 still replay the old filter. The spectral `Aw=True`
  options are unchanged; they already evaluate the analog curve.
  A/B/C特性の周波数重み付けフィルタ（`a_weighting()`、`sound_level(freq_weighting="A" | "C")`、
  `rms_trend(Aw=True)`、`leq()`、`percentile_levels()`）が、8 kHz〜192 kHzでナイキスト周波数の
  0.9倍までIEC 61672-1のアナログ曲線に約0.2 dB以内で一致するようになりました。waveform-analysisから
  取り込んだ従来の設計はアナログフィルタ全体を双一次変換していたため、ナイキスト周波数付近で
  減衰しすぎていました（48 kHzで10 kHzが1.2 dB、16 kHzが6.4 dB、20 kHzが15.8 dB低く、A特性の
  白色雑音は0.87 dB低く出ていました）。低域の極は引き続き双一次変換し、12.2 kHzの二重極は
  matched-z変換で写し、2つの零点を線形最小二乗でアナログ曲線に合わせます。フィルタは因果的で、
  入力のサンプリング周波数のまま動きます。数値が変わるため、新しいRecipeは`a_weighting`の
  version 2、`rms_trend`／`sound_level`のversion 3を使います。0.12.0で保存したRecipeは従来の
  フィルタで再生されます。スペクトルの`Aw=True`はもともとアナログ曲線を評価しているため変わりません。
