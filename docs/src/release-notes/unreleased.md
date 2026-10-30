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
