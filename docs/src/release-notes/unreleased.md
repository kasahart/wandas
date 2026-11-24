# Unreleased changes

Changes after [Wandas 0.13.2](v0.13.2.md).

0.13.2以降の変更です。

## Added / 追加

- `ChannelFrameDataset.trigger()` cuts out level-trigger events across every
  recording of a Dataset and returns one Dataset of all hits. Each hit keeps
  its recording's metadata plus `trigger_index` and `trigger_time`
  ([#589](https://github.com/kasahart/wandas/issues/589)).
  `ChannelFrameDataset.trigger()`で、Datasetの全録音からレベルトリガーのイベントを切り出し、
  全ヒットを1つのDatasetで返します。各ヒットは元の録音のmetadataと`trigger_index`・`trigger_time`を持ちます。
