# Unreleased changes

Changes after [Wandas 0.13.0](v0.13.0.md).

0.13.0以降の変更です。

## Improvements / 改善

- `from_numpy()`, `from_file()`, `read_wav()` and `read_csv()` called on a
  `ChannelFrame` subclass now return that subclass, and `add_channel()`,
  `remove_channel()` and `concat_frame()` keep it, so an extension that adds
  methods to `ChannelFrame` can load files directly.
  `ChannelFrame`のサブクラスで`from_numpy()`・`from_file()`・`read_wav()`・`read_csv()`を呼ぶと
  そのサブクラスを返し、`add_channel()`・`remove_channel()`・`concat_frame()`もサブクラスを保つように
  なりました。`ChannelFrame`にメソッドを足す拡張から、ファイルを直接読み込めます。
