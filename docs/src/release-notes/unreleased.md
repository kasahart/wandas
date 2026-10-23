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
  with `Literal[...]`, so type checkers flag other values and, for Recipe
  operations, `RecipeRegistry.describe()` reports the choices:
  `average` of `welch()`, `csd()` and `transfer_function()`; `scaling` of
  `csd()`, `transfer_function()` and `SpectrogramFrame.csd()`; `detrend` of
  `stft()` (Frame and dataset); `align` of `mix()`, `add_channel()` and
  `concat_frame()`; `weighting` and `field_type` of `sharpness_din()` and
  `sharpness_din_st()`; `field_type` of `loudness_zwtv()` and `loudness_zwst()`; `freq_weighting`
  (`"A"`, `"C"`, `"Z"`, `None`) and `time_weighting` (`"Fast"`, `"Slow"`) of
  `sound_level()`. Runtime validation and accepted values are unchanged;
  `sound_level()` still accepts other spellings such as `"a"` or `"F"` at
  runtime, but type checkers now report them.
  既に固定の文字列だけを受け付ける引数に`Literal[...]`注釈を付けました。実行時の検証と
  受け付ける値は変わりません。
- `fft()` and `cepstrum()` of an input shorter than `n_fft` window the
  available samples before zero-padding, so a short sinusoid keeps its
  amplitude. `SpectralFrame` and `CepstralFrame` gain `win_length` (default
  `n_fft`): Welch and STFT frames record theirs, spectral envelopes inherit
  the cepstrum's, `ifft()` undoes the gain of that window, and frame-frame
  arithmetic rejects operands whose `win_length` differs. The `fft`,
  `cepstrum` and `ifft` Recipe operations move to version 3; released
  version 2 payloads replay their previous numerics, so a saved Recipe gives
  the same result as before and a newly captured one uses the new numerics.
  WDF files of frames whose `win_length` is shorter than `n_fft` store it and
  are rejected by older readers, so upgrade readers before exchanging them;
  other files are unchanged. An empty input with an explicit `n_fft` has no
  analysis window, so `fft()` and `cepstrum()` now raise `ValueError` instead
  of returning an all-zero spectrum or a floor-only cepstrum.
  `n_fft`より短い入力の`fft()`と`cepstrum()`は、窓を掛けてからゼロ詰めするようになり、
  短い正弦波でも振幅が保たれます。`SpectralFrame`と`CepstralFrame`に`win_length`を
  追加し、`fft`・`cepstrum`・`ifft`のRecipe操作をversion 3にしました。保存済みの
  version 2は従来の数値で再生されます。`win_length`が`n_fft`より短いWDFは旧版では
  読み込めません。空の入力に`n_fft`を指定した場合は解析窓がないため、ゼロのスペクトルや
  floorだけのケプストラムを返さず`ValueError`になります。

See [Wandas 0.9.0](v0.9.0.md) for the previous release.
