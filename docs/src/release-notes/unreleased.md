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
- `SpectralFrame.noct_synthesis()` band levels were about 4.8 dB too high
  for a Hann-windowed spectrum, and more for zero-padded input, because the
  peak-amplitude bins were passed to MoSQITo as RMS bins and the window's
  leakage was counted as power. The bins are now converted to RMS and divided
  by the square root of the window's equivalent noise bandwidth, so the levels
  match `noct_spectrum()`. `NOctSynthesis` takes the source `window` and
  `win_length`; `noct_synthesis()` passes the frame's. The Recipe operation
  moves to version 3, and saved version 2 Recipes replay the previous numerics.
  Because the band levels depend on the window, frame-frame arithmetic on
  `SpectralFrame` now also rejects operands whose `window` differs.
  `SpectralFrame.noct_synthesis()`のバンドレベルが、Hann窓で約4.8 dB、ゼロ詰め入力では
  それ以上高く出ていました。ピーク振幅をRMSに換算し、窓の等価雑音帯域幅で補正するように
  したため、`noct_spectrum()`と一致します。Recipe操作はversion 3になり、保存済みの
  version 2は従来の数値で再生されます。窓が異なる`SpectralFrame`同士の演算は
  `ValueError`になります。
- `power()`, `abs()`, `channel_difference()`, `+ - * **` and the NumPy
  ufuncs `add`, `subtract`, `multiply`, `power`, `square`, `absolute` and
  `negative` on a `ChannelFrame` holding integer or boolean samples now
  compute in `float64` whenever NumPy would have produced an integer dtype
  (in plain ufunc calls without `out`, `dtype` or `signature`; methods such as
  `np.multiply.outer` are unchanged). They
  previously wrapped around silently (for example, `int16` `power(2)` of 29500
  gave -2544). Operations with a floating-point operand keep NumPy's dtype
  promotion, so `float32` stays `float32`. `int64` values beyond 2**53 are
  rounded by the `float64` computation.
  整数またはboolサンプルの`ChannelFrame`に対する`power()`・`abs()`・
  `channel_difference()`・`+ - * **`と、NumPyのufunc `add`・`subtract`・`multiply`・
  `power`・`square`・`absolute`・`negative`は、NumPyが整数dtypeで計算する場合に
  `float64`で計算するようになりました（`out`・`dtype`・`signature`を指定しない
  通常のufunc呼び出しが対象で、`np.multiply.outer`などのメソッドは変わりません）。これまでは黙って桁あふれして
  いました（例: `int16`の29500の`power(2)`が-2544）。浮動小数点のオペランドを含む
  演算はNumPyの型昇格のままなので、`float32`は`float32`のままです。2**53を超える
  `int64`の値は`float64`の計算で丸められます。

## Compatibility and migration / 互換性と移行

| Affected surface or artifact | Classification | Deprecation start | Replacement or migration | Removal/change version | Exception reason and decision link |
| --- | --- | --- | --- | --- | --- |
| `wandas.audio.power`, `wandas.audio.abs`, `wandas.audio.channel_difference` and `wandas.operator.*` Recipe nodes on integer input | Serialized, semantic change without a version bump | None | Replays on integer input return `float64`; cast with `astype` only if an integer result is required | Next release | A private replay would only reproduce wrapped-around values; floating-point input is unchanged; [#511](https://github.com/kasahart/wandas/issues/511#issuecomment-6371647712) |

See [Wandas 0.9.0](v0.9.0.md) for the previous release.
