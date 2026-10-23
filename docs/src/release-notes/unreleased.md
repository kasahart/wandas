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
- `ChannelFrame.sum()` and `mean()` reset the reference to the unit default,
  so levels of the result were off when the channels used another reference.
  When the channels' offsets differed, the result's `source_time_offset` was
  reset to 0. The result now keeps the reference the channels share and the
  first channel's offset, the same as binary arithmetic keeps the left
  operand's. The Recipe operations move to version 2, and saved version 1
  Recipes replay the previous metadata.
  `ChannelFrame.sum()`と`mean()`は基準値を単位の既定値に戻していたため、別の基準値を
  使うチャネルではレベルがずれていました。チャネルの時刻オフセットが異なると、結果の
  `source_time_offset`は0になっていました。チャネルに共通の基準値と、二項演算が左辺の
  オフセットを保持するのと同じく先頭チャネルのオフセットを保持するようにしました。
  Recipe操作はversion 2になり、保存済みのversion 1は従来のメタデータで再生されます。
- `roughness_dw_spec()` no longer copies `overlap` into the frame's user
  `metadata`, so the key is no longer written to WDF files. Read it from
  `RoughnessFrame.overlap` or the operation history.
  `roughness_dw_spec()`は`overlap`をフレームのユーザー`metadata`に書き込まなくなり、
  WDFファイルにも保存されません。`RoughnessFrame.overlap`か操作履歴から読んでください。
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
- Chained channel-independent operations (filters, A-weighting, resampling,
  `noct_spectrum`, `normalize`, HPSS, ...) build a lazy graph that grows
  linearly with channels × stages. Each stage copied the previous stage's
  graph once per channel, so an 8-channel frame with six chained filters built
  a graph of about 170,000 tasks before computing. Results are unchanged.
  チャネル独立の演算（フィルタ、A特性、リサンプリング、`noct_spectrum`、`normalize`、
  HPSSなど）を重ねたときの遅延グラフが、チャネル数×段数に比例する大きさになりました。
  これまでは段ごとに前段のグラフをチャネルの数だけ複製していたため、8チャネルに6段の
  フィルタを重ねると計算前に約17万タスクのグラフができていました。計算結果は変わりません。

## Compatibility and migration / 互換性と移行

| Affected surface or artifact | Classification | Deprecation start | Replacement or migration | Removal/change version | Exception reason and decision link |
| --- | --- | --- | --- | --- | --- |
| `wandas.audio.power`, `wandas.audio.abs`, `wandas.audio.channel_difference` and `wandas.operator.*` Recipe nodes on integer input | Serialized, semantic change without a version bump | None | Replays on integer input return `float64`; cast with `astype` only if an integer result is required | Next release | A private replay would only reproduce wrapped-around values; floating-point input is unchanged; [#511](https://github.com/kasahart/wandas/issues/511#issuecomment-6371647712) |

## Fixed / 修正

- Channel selection on a mono `RoughnessFrame` (`frame[0]`, a label,
  `get_channel()`, iteration and `frame[0, :, start:stop]`) raised
  `ValueError` because the mono data had no channel axis and the selection
  indexed the Bark axis. A mono `RoughnessFrame` now stores its data as
  `(1, 47, n_time)`, like one channel of a multi-channel frame, so selection
  works and a mono frame and its selection can be combined arithmetically.
  `.data` and `.shape` of a mono frame are still `(47, n_time)`, and `.data`
  of a single channel selected from a multi-channel frame is now also
  `(47, n_time)` instead of `(1, 47, n_time)`, as for other Frame types. WDF
  files written from a mono frame store the channel axis; older readers read
  them as a one-channel frame.
  モノラルの`RoughnessFrame`はチャネル軸を持たなかったため、チャネル選択（`frame[0]`、
  ラベル、`get_channel()`、反復、`frame[0, :, start:stop]`）がBark軸を添字にしてしまい
  `ValueError`になっていました。モノラルも内部では`(1, 47, n_time)`で保持するようにし、
  選択や、選択結果との四則演算ができるようになりました。モノラルの`.data`と`.shape`は
  `(47, n_time)`のままです。複数チャネルから1チャネル選んだ場合の`.data`も、他のFrameと
  同じく`(47, n_time)`になります（以前は`(1, 47, n_time)`）。
- `WandasOperationTransformer.set_params()` cleared the parameters that were
  not given when `operation` named the operation already configured, so the
  method defaults applied instead (for example `order=8` fell back to 4). It
  now updates only the given parameters; a different `operation` still
  replaces them.
  `WandasOperationTransformer.set_params()`に設定済みと同じ`operation`を渡すと、
  指定しなかった引数が消えてメソッドの既定値に戻っていました（例: `order=8`が4に）。
  指定した引数だけを更新するようにしました。別の`operation`を渡した場合は従来どおり
  引数を置き換えます。

See [Wandas 0.9.0](v0.9.0.md) for the previous release.
