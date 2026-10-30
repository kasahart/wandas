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
- `ChannelFrame.cross_correlation(max_lag=None)` returns the normalized
  cross-correlation of every output/input channel pair as a new
  `CrossCorrelationFrame` with a lag axis in seconds (positive when the output
  lags the input). Each row is `scipy.signal.correlate(x_out, x_in)` divided by
  `sqrt(E_out * E_in)`, a coefficient in [-1, 1]. `delays` gives each pair's
  lag of the largest `abs(coefficient)` and `peak_coefficients` the signed
  value there; `select_pair()`, contiguous lag slicing, `plot()` (lag on x),
  `to_dataframe()`, WDF save/load and Recipe replay work as for the pairwise
  spectral Frames. `max_lag` keeps only lags within `+-max_lag` seconds.
  `ChannelFrame.cross_correlation(max_lag=None)`で、全ての出力／入力チャンネルの組の正規化相互相関を、
  秒単位のラグ軸（出力が入力より遅れるとき正）を持つ新しい`CrossCorrelationFrame`として得られる
  ようになりました。各行は`scipy.signal.correlate(x_out, x_in)`を`sqrt(E_out * E_in)`で割った
  [-1, 1]の係数です。`delays`は各組で`abs(係数)`が最大になるラグ、`peak_coefficients`はそこでの
  符号付きの値です。`select_pair()`、連続したラグの切り出し、`plot()`（横軸がラグ）、
  `to_dataframe()`、WDFの保存・読み込み、Recipeの再実行は、ペアのスペクトルFrameと同じように
  使えます。`max_lag`を指定すると`±max_lag`秒以内のラグだけを残します。

## Changed / 変更

- `Frame.cache()` no longer hashes every sample to name its Dask array, which
  saves one pass over the data (about 0.1–0.3 s for a 10-minute 48 kHz stereo
  recording) ([#598](https://github.com/kasahart/wandas/issues/598)). Values,
  dtype, chunks, metadata and lineage are unchanged; two separate `cache()`
  calls on equal data no longer share a Dask key.
  `Frame.cache()`はDask配列の名前を付けるために全サンプルをハッシュしなくなり、データを1回読む分だけ
  速くなりました（48 kHzステレオ10分の録音で約0.1〜0.3秒）。値、dtype、チャンク、メタデータ、
  lineageは変わりません。等しいデータに別々に`cache()`を呼んだ結果がDaskのキーを共有しなくなります。
