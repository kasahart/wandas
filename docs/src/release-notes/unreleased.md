# Unreleased changes

## Added / 追加

- `RecipeRegistry.describe()` and `RecipeOperation.describe()` return a
  machine-readable catalog of Recipe operations: ID, version, method name,
  input bindings, and method parameters with defaults and `Literal`
  choices. `@recipe_operation` records the decorated method on
  `RecipeOperation.method`; capture and replay are unchanged.
  Recipe操作のカタログ（ID、version、メソッド名、入力、引数・既定値・`Literal`の選択肢）を
  返す`RecipeRegistry.describe()`を追加しました。

See [Wandas 0.9.0](v0.9.0.md) for the previous release.
