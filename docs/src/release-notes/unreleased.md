# Unreleased changes

Changes after [Wandas 0.10.0](v0.10.0.md).

## Changed / 変更

- `ChannelFrame.describe()` with its defaults no longer raises `ImportError`
  when IPython is not installed (for example a plain script on the core
  install). It shows the channel figures with one `matplotlib.pyplot.show()`
  call and then closes them; in Matplotlib interactive mode the figures stay
  open. Audio playback still needs IPython (`pip install "wandas[marimo]"`).
  The notebook path with IPython is unchanged.
  `ChannelFrame.describe()`は、IPythonがない環境（コアのみをインストールした素のスクリプトなど）で
  既定の引数のまま呼んでも`ImportError`にならなくなりました。各チャンネルの図を
  `matplotlib.pyplot.show()`で1回表示してから閉じます。Matplotlibの対話モードでは図を開いたままにします。
  音声再生には引き続きIPython（`pip install "wandas[marimo]"`）が必要です。IPythonがある
  ノートブックでの表示は変わりません。
