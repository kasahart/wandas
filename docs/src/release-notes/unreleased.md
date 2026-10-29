# Unreleased changes

Changes after [Wandas 0.11.0](v0.11.0.md).

## Added / 追加

- `ChannelFrame.find_triggers()` finds the times where one channel crosses a
  level (rising or falling, with a hold-off time against re-triggering), and
  `ChannelFrame.segment()` cuts all channels into fixed-length lazy segments
  around those times with a pre-trigger. Each segment is a `ChannelFrame`
  whose `source_time_offset` records its position in the recording, so
  impact hammer tests and other transient measurements can be analysed hit by
  hit with the existing methods ([#587](https://github.com/kasahart/wandas/issues/587)).
  `ChannelFrame.find_triggers()`は1つのチャンネルがレベルを横切る時刻を求めます（立ち上がり／
  立ち下がり、再トリガーを防ぐホールドオフ時間付き）。`ChannelFrame.segment()`はその時刻の前後で
  全チャンネルをプリトリガー付きの固定長セグメントに遅延評価のまま切り出します。各セグメントは
  `ChannelFrame`で、`source_time_offset`に収録上の位置が残るため、インパクトハンマー試験などの
  過渡的な測定を既存のメソッドで打撃ごとに解析できます。
