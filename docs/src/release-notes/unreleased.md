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
