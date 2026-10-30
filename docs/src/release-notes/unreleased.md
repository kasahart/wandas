# Unreleased changes

Changes after [Wandas 0.11.0](v0.11.0.md).

## Added / 追加

- `ChannelFrame.trigger()` cuts out every event where one channel crosses a
  level (rising or falling, with a pre-trigger and a hold-off time against
  re-triggering), for impact hammer tests and other transient measurements
  ([#587](https://github.com/kasahart/wandas/issues/587)), for example
  `hits = wd.read("hammer.wav").high_pass_filter(5).trigger(10.0, duration=1.0, pre=0.01, channel="force")`.
  `ChannelFrame.segment()` cuts at given times instead. Both return the new
  `ChannelFrameSegments`: `hits[i]` is one hit as a lazy `ChannelFrame`,
  `hits.drop([2])` or a mask removes bad hits by a stable event ID, and
  `hits.plot()` overlays all hits per channel. `hits.transfer_function()` and
  `hits.coherence()` average the hits into one H1 frequency response and its
  coherence (each hit is one rectangular-window spectral average), and
  `hits.average()` gives the synchronous time average. `hits.apply(func)`
  processes each hit and keeps the collection (as the generic `FrameSegments`
  when the results are not ChannelFrames), and `hits.to_numpy()` returns a
  `(segment, channel, sample)` array.
  `ChannelFrame.trigger()`は1つのチャンネルがレベルを横切るたびに（立ち上がり／立ち下がり、
  プリトリガーと再トリガーを防ぐホールドオフ時間付き）、全チャンネルを固定長で切り出します。
  インパクトハンマー試験などの過渡的な測定向けです。`ChannelFrame.segment()`は与えた時刻で切り出します。
  どちらも新しい`ChannelFrameSegments`を返します。`hits[i]`は1打撃の遅延評価の`ChannelFrame`、
  `hits.drop([2])`やマスクは変わらないイベントIDで悪い打撃を除き、`hits.plot()`は全打撃を
  チャンネルごとに重ねて描きます。`hits.transfer_function()`と`hits.coherence()`は打撃を平均した
  H1の周波数応答関数とコヒーレンスを返し（1打撃を矩形窓の1区間として平均）、`hits.average()`は
  時間波形の同期加算平均を返します。`hits.apply(func)`は打撃ごとに処理して集合のまま返し
  （結果がChannelFrameでなければ汎用の`FrameSegments`）、`hits.to_numpy()`は
  `(セグメント, チャンネル, サンプル)`の配列を返します。
- The learning path has a new lesson, `09_impact_hammer_frf`, that cuts the
  hits of an impact test with `trigger()`, drops a clipped hit and compares the
  hit-averaged FRF with a known structure
  ([#605](https://github.com/kasahart/wandas/issues/605)).
  学習パスに新しい教材`09_impact_hammer_frf`を追加しました。`trigger()`で打撃試験の打撃を切り出し、
  頭打ちの打撃を除いて、打撃平均のFRFを既知の構造と比べます。
- `ChannelFrame.envelope()` returns the amplitude envelope of each channel,
  `abs(scipy.signal.hilbert(x))` over the whole time axis, as a lazy
  `ChannelFrame` with the same sampling rate, length and calibrated unit.
  float32 input stays float32, and it is a Recipe step
  (`wandas.audio.envelope`). For an envelope spectrum, band-pass first:
  `frame.band_pass_filter(2000, 4000).envelope().remove_dc().fft()`.
  The first and last samples carry SciPy's edge effects.
  `ChannelFrame.envelope()`で各チャンネルの振幅包絡（時間軸全体での
  `abs(scipy.signal.hilbert(x))`）を、サンプリング周波数・長さ・校正済み単位が同じ遅延
  `ChannelFrame`として得られるようになりました。float32入力はfloat32のままで、Recipeの段
  （`wandas.audio.envelope`）にもなります。包絡スペクトルは帯域通過フィルタの後に
  `frame.band_pass_filter(2000, 4000).envelope().remove_dc().fft()`で求められます。
  先頭と末尾のサンプルにはSciPyの端の影響が残ります。
- `ChannelFrame.leq(freq_weighting="Z")` returns the equivalent continuous
  level of each channel, and `ChannelFrame.percentile_levels(n, freq_weighting,
  time_weighting)` the levels exceeded `n` % of the time (L_5, L_50, L_95, ...)
  of the `sound_level(dB=True)` time series. Both are scalar reductions like
  `rms`: they compute immediately and return NumPy arrays, not Frames.
  `leq("A")` of a calibrated Pa channel is L_Aeq in dB SPL and equals
  `channels[i].level_reference.to_level(frame.a_weighting().rms[i])`.
  `ChannelFrame.leq(freq_weighting="Z")`で各チャンネルの等価騒音レベルを、
  `ChannelFrame.percentile_levels(n, freq_weighting, time_weighting)`で
  `sound_level(dB=True)`の時系列から時間率レベル（`n` %の時間超えるレベル。L_5、L_50、
  L_95など）を求められるようになりました。どちらも`rms`と同じくその場で計算し、Frameではなく
  NumPy配列を返します。校正済みのPaチャンネルの`leq("A")`はdB SPLのL_Aeqで、
  `channels[i].level_reference.to_level(frame.a_weighting().rms[i])`と等しくなります。
- `ChannelFrame.transfer_function()` and `CrossSpectralFrame.transfer_function()`
  take `estimator="H1"` (default, unchanged) or `estimator="H2"`. H2 is
  `P_out_out / P_in_out` and is not biased by noise on the input channel, where
  H1 (`P_out_in / P_in_in`) is underestimated; with `average="mean"` (the
  default) `H1 / H2` equals the coherence.
  The result is the same `TransferFunctionFrame` with the same unit and
  reference, plus an `estimator` property. WDF files store an H2 result with
  the new `definition` value `h2_input_cross_denominator`, which older Wandas
  versions cannot load; H1 files are unchanged. Recipes record `estimator`
  only when it is given, so existing Recipes replay as before.
  `ChannelFrame.transfer_function()`と`CrossSpectralFrame.transfer_function()`で
  `estimator="H1"`（既定、従来どおり）または`estimator="H2"`を選べるようになりました。H2は
  `P_out_out / P_in_out`で、入力チャンネル側の雑音で過小評価されるH1（`P_out_in / P_in_in`）と違い、
  入力側の雑音の影響を受けません。`average="mean"`（既定）のとき`H1 / H2`はコヒーレンスに等しくなります。結果は単位と基準が
  同じ`TransferFunctionFrame`で、`estimator`プロパティが増えました。H2の結果をWDFに保存すると
  `definition`は新しい値`h2_input_cross_denominator`になり、古いWandasでは読み込めません。
  H1のファイルは変わりません。Recipeには`estimator`を指定したときだけ記録するため、既存の
  Recipeはこれまでどおり再実行できます。

- `trim()` selects by time on more frames and in more units
  ([#603](https://github.com/kasahart/wandas/issues/603)). Times are seconds or
  strings with the units `h`, `min`, `s` and `ms`, such as
  `recording.trim("1min30s", "2min")`. `reference="source"` measures them on the
  source timeline instead of from the frame start. `SpectrogramFrame.trim()` and
  `CepstrogramFrame.trim()` keep the frames whose center lies in the half-open
  range, and `SpectrogramFrame.at_time(12.3)` returns the spectrum of the frame
  whose center is nearest, with an optional `tolerance`. All of them are Recipe
  steps that record the requested times, so a plan selects the same seconds on
  files with other sampling rates.
  `trim()`で秒のほか`h`・`min`・`s`・`ms`付きの文字列（`recording.trim("1min30s", "2min")`など）を
  指定できるようになり、`reference="source"`で元の音源の時刻を使えます。`SpectrogramFrame.trim()`と
  `CepstrogramFrame.trim()`は中心が半開区間に入るフレームを残し、`SpectrogramFrame.at_time(12.3)`は
  中心が最も近いフレームのスペクトルを返します（`tolerance`も指定可）。どれもRecipeの段で、
  指定した時刻を記録するため、サンプリング周波数が違うファイルでも同じ秒数を選びます。

## Changed / 変更

- `ChannelFrame.trim()` rounds times to the nearest sample (halves up) instead of
  truncating them, so `trim(0.29)` at 100 Hz starts at sample 29, not 28
  (`0.29 * 100` is `28.999...` in floating point). The Recipe step
  `wandas.frame.time_slice` is now version 2; saved version 1 plans replay with
  the old truncation.
  `ChannelFrame.trim()`は時刻をサンプルへ切り捨てずに四捨五入するようになりました。100 Hzで
  `trim(0.29)`は28ではなく29番目のサンプルから始まります（浮動小数点では`0.29 * 100`が`28.999...`）。
  Recipeの段`wandas.frame.time_slice`はversion 2になり、保存済みのversion 1は従来の切り捨てで再実行します。

- `Frame.cache()` no longer hashes every sample to name its Dask array, which
  saves one pass over the data (about 0.1–0.3 s for a 10-minute 48 kHz stereo
  recording) ([#598](https://github.com/kasahart/wandas/issues/598)). Values,
  dtype, chunks, metadata and lineage are unchanged; two separate `cache()`
  calls on equal data no longer share a Dask key.
  `Frame.cache()`はDask配列の名前を付けるために全サンプルをハッシュしなくなり、データを1回読む分だけ
  速くなりました（48 kHzステレオ10分の録音で約0.1〜0.3秒）。値、dtype、チャンク、メタデータ、
  lineageは変わりません。等しいデータに別々に`cache()`を呼んだ結果がDaskのキーを共有しなくなります。
