# Unreleased changes

## Added / 追加

- `RecipeRegistry.describe()` and `RecipeOperation.describe()` return a
  machine-readable catalog of Recipe operations: ID, version, method name,
  input bindings, and method parameters with defaults and `Literal`
  choices. `@recipe_operation` records the decorated method on
  `RecipeOperation.method`; capture and replay are unchanged.
  Recipe操作のカタログ（ID、version、メソッド名、入力、引数・既定値・`Literal`の選択肢）を
  返す`RecipeRegistry.describe()`を追加しました。

## Changed / 変更

- Parameters that already accept only a fixed set of strings are annotated
  with `Literal[...]`, so type checkers and `describe()` report their choices:
  `average` of `welch()`, `csd()` and `transfer_function()`; `scaling` of
  `csd()`, `transfer_function()` and `SpectrogramFrame.csd()`; `detrend` of
  `stft()` (Frame and dataset); `align` of `mix()`, `add_channel()` and
  `concat_frame()`; `weighting` and `field_type` of `sharpness_din()` and
  `sharpness_din_st()`. Runtime validation and accepted values are unchanged.
  既に固定の文字列だけを受け付ける引数に`Literal[...]`注釈を付けました。実行時の検証と
  受け付ける値は変わりません。

See [Wandas 0.9.0](v0.9.0.md) for the previous release.
