# Unreleased changes

Changes after [Wandas 0.12.0](v0.12.0.md).

0.12.0以降の変更です。

## New features / 新機能

- `ChannelFrame.peak` and `ChannelFrame.kurtosis` return the peak absolute
  value and the Pearson kurtosis of each channel as NumPy arrays, like `rms`.
  `kurtosis` uses `dask.array.stats.kurtosis`.
  `ChannelFrame.peak`と`ChannelFrame.kurtosis`は、`rms`と同じように各チャンネルの絶対値の最大と
  ピアソンの尖度をNumPy配列で返します。`kurtosis`は`dask.array.stats.kurtosis`で計算します。
