# Unreleased changes

Changes after [Wandas 0.10.0](v0.10.0.md).

## Changed / 変更

- `ChannelFrame.describe()` with its defaults now shows its figures outside a
  notebook. Before, a plain script raised `ImportError` without IPython and,
  with IPython installed, only printed `Figure(...)` and `Audio` object text.
  Outside a notebook (a plain script or the terminal IPython shell) it now
  shows the channel figures with one `matplotlib.pyplot.show()` call and then
  closes them; in Matplotlib interactive mode the figures stay open. Audio
  playback still needs a notebook with IPython (`pip install "wandas[marimo]"`).
  Display in Jupyter, VS Code, Colab and marimo notebooks is unchanged.
  `ChannelFrame.describe()`を既定の引数のまま呼んだとき、ノートブックの外でも図が表示されるように
  なりました。これまでは素のスクリプトで、IPythonがなければ`ImportError`になり、IPythonがあっても
  `Figure(...)`と`Audio`オブジェクトの文字列が出るだけでした。ノートブックの外（素のスクリプトや
  ターミナルのIPython）では、各チャンネルの図を`matplotlib.pyplot.show()`で1回表示してから閉じます。
  Matplotlibの対話モードでは図を開いたままにします。音声再生には引き続き、IPythonのある
  ノートブック（`pip install "wandas[marimo]"`）が必要です。Jupyter、VS Code、Colab、marimoの
  ノートブックでの表示は変わりません。
