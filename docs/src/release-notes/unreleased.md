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

## Changed / 変更

- `Frame.cache()` no longer hashes every sample to name its Dask array, which
  saves one pass over the data (about 0.1–0.3 s for a 10-minute 48 kHz stereo
  recording) ([#598](https://github.com/kasahart/wandas/issues/598)). Values,
  dtype, chunks, metadata and lineage are unchanged; two separate `cache()`
  calls on equal data no longer share a Dask key.
  `Frame.cache()`はDask配列の名前を付けるために全サンプルをハッシュしなくなり、データを1回読む分だけ
  速くなりました（48 kHzステレオ10分の録音で約0.1〜0.3秒）。値、dtype、チャンク、メタデータ、
  lineageは変わりません。等しいデータに別々に`cache()`を呼んだ結果がDaskのキーを共有しなくなります。

## Fixed / 修正

- Slicing an `NOctFrame` along the band axis now labels the selected bands
  with their own center frequencies
  ([#600](https://github.com/kasahart/wandas/issues/600)). Before, `freqs`
  restarted from the original lowest band, so `noct[:, 3:6]` of a
  100 Hz–4 kHz spectrum was labelled 100/125/160 Hz instead of
  200/250/315 Hz, and `dBA`, `plot()`, `to_dataframe()` and saved WDF files
  used those wrong labels. The slice now moves `fmin`/`fmax` to the first and
  last selected bands. A stepped band slice (`noct[:, ::2]`) or an empty one
  now raises `ValueError`, because `fmin`/`fmax` can describe only a
  contiguous run of bands. A WDF file saved from a band slice before this fix
  keeps its wrong labels, because it does not store which bands were selected;
  recompute it from the source.
  `NOctFrame`をバンド方向にスライスしたとき、選んだバンドに正しい中心周波数が付くようになりました。
  これまでは`freqs`が元の最低バンドから数え直されていたため、100 Hz〜4 kHzのスペクトルの
  `noct[:, 3:6]`に200/250/315 Hzではなく100/125/160 Hzが付き、`dBA`、`plot()`、
  `to_dataframe()`、保存したWDFファイルもその誤ったラベルを使っていました。スライスは
  `fmin`/`fmax`を選んだ最初と最後のバンドへ移します。`fmin`/`fmax`は連続したバンドしか表せないため、
  ステップ付きのスライス（`noct[:, ::2]`）と空のスライスは`ValueError`になります。
  この修正より前にバンドをスライスして保存したWDFファイルは、どのバンドを選んだかを保存していないため
  誤ったラベルのままです。元のデータから計算し直してください。

## Compatibility / 互換性

| Affected surface or artifact | Classification | Deprecation start | Replacement or migration | Removal/change version | Exception reason and decision link |
| --- | --- | --- | --- | --- | --- |
| Replay of `wandas.frame.index` (v1) Recipes that slice an `NOctFrame` along the band axis | Serialized | `None` | None needed: contiguous slices now replay with correct band labels; rewrite a stepped band slice as separate contiguous slices | Next release | Wrong labels become correct (values unchanged) and stepped slices raise `ValueError`; [decision](https://github.com/kasahart/wandas/issues/600#issuecomment-6514808419) |
