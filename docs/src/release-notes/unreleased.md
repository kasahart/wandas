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

- `SpectralFrame` (from `fft()` and `welch()`) and `SpectrogramFrame` (from
  `stft()`) can hold a subset of their frequency bins
  ([#602](https://github.com/kasahart/wandas/issues/602)). Before,
  `spectrum[:, 10:20]` and `spec[:, 5:40, :]` raised "Invalid frequency bin
  count". The new read-only `frequency_indices` lists the represented bins of the
  one-sided `n_fft` grid, as on the pairwise frames, and `is_complete_spectrum`
  tells whether every bin is there. `freqs`, `dB`, `dBA`, `plot()`,
  `to_dataframe()`, `get_frame_at()`, `abs()`, arithmetic, WDF files and Recipe
  replay follow the selected bins; arithmetic between two spectra needs the same
  bins. `ifft()`, `istft()`, `noct_synthesis()`,
  `cepstrum()` and `SpectrogramFrame.csd()` need the complete spectrum and raise
  `ValueError` on a subset; slice their results instead. A WDF file of a subset
  stores `frequency_indices` and cannot be read by older Wandas versions; a
  complete spectrum is written as before.
  `SpectralFrame`（`fft()`・`welch()`の結果）と`SpectrogramFrame`（`stft()`の結果）が、
  周波数ビンの一部だけを持てるようになりました。これまでは`spectrum[:, 10:20]`や
  `spec[:, 5:40, :]`が「Invalid frequency bin count」になっていました。新しい読み取り専用の
  `frequency_indices`は、ペアワイズのFrameと同じく、片側`n_fft`グリッドのうち持っているビンを表し、
  `is_complete_spectrum`は全ビンを持つかどうかを返します。`freqs`、`dB`、`dBA`、`plot()`、
  `to_dataframe()`、`get_frame_at()`、`abs()`、四則演算、WDFファイル、Recipeの再実行は選んだビンに
  従います。2つのスペクトル同士の四則演算には同じビンが必要です。`ifft()`、`istft()`、`noct_synthesis()`、`cepstrum()`、`SpectrogramFrame.csd()`は
  全ビンが必要なため、一部だけのときは`ValueError`になります。その場合は結果の方をスライスしてください。
  一部のビンだけのWDFファイルは`frequency_indices`を保存するため、古いWandasでは読めません。
  全ビンのときは従来どおりに書き出します。

## Changed / 変更

- `Frame.cache()` no longer hashes every sample to name its Dask array, which
  saves one pass over the data (about 0.1–0.3 s for a 10-minute 48 kHz stereo
  recording) ([#598](https://github.com/kasahart/wandas/issues/598)). Values,
  dtype, chunks, metadata and lineage are unchanged; two separate `cache()`
  calls on equal data no longer share a Dask key.
  `Frame.cache()`はDask配列の名前を付けるために全サンプルをハッシュしなくなり、データを1回読む分だけ
  速くなりました（48 kHzステレオ10分の録音で約0.1〜0.3秒）。値、dtype、チャンク、メタデータ、
  lineageは変わりません。等しいデータに別々に`cache()`を呼んだ結果がDaskのキーを共有しなくなります。
