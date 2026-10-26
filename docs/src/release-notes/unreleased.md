# Unreleased changes

Changes after [Wandas 0.10.0](v0.10.0.md).

## Changed / 変更

- `ChannelFrame.describe()` with its defaults now shows its figures outside a
  notebook. Before, a plain script raised `ImportError` without IPython and,
  with IPython installed, only printed `Figure(...)` and `Audio` object text.
  Outside a notebook (a plain script or the terminal IPython shell) it now
  shows the channel figures with one `matplotlib.pyplot.show()` call and then
  closes them; in Matplotlib interactive mode the figures stay open. Audio
  playback outside PyScript (see below) still needs a notebook with IPython
  (`pip install "wandas[marimo]"`).
  Display in Jupyter, VS Code, Colab and marimo notebooks is unchanged.
  `ChannelFrame.describe()`を既定の引数のまま呼んだとき、ノートブックの外でも図が表示されるように
  なりました。これまでは素のスクリプトで、IPythonがなければ`ImportError`になり、IPythonがあっても
  `Figure(...)`と`Audio`オブジェクトの文字列が出るだけでした。ノートブックの外（素のスクリプトや
  ターミナルのIPython）では、各チャンネルの図を`matplotlib.pyplot.show()`で1回表示してから閉じます。
  Matplotlibの対話モードでは図を開いたままにします。PyScript（下記）以外での音声再生には引き続き、IPythonのある
  ノートブック（`pip install "wandas[marimo]"`）が必要です。Jupyter、VS Code、Colab、marimoの
  ノートブックでの表示は変わりません。
- In a PyScript page, `ChannelFrame.describe()` and the `plot()` methods now
  put their figures on the page with `pyscript.display()` when the running code
  has an output element (`pyscript.current_target()`). `describe()` also adds
  an HTML audio player for each channel. Before, the figures went to
  `matplotlib.pyplot.show()` and never reached the page. A `plot()` call that
  creates its own figure displays and closes it, including `overlay=True`, so
  later changes to the returned Axes do not reach the page. `plot("describe")`
  is the building block of `describe()` and is not displayed on its own.
  `py-editor` cells have no output target, so they keep the previous behaviour.
  PyScriptのページでは、実行中のコードに出力先の要素（`pyscript.current_target()`）があるとき、
  `ChannelFrame.describe()`と各`plot()`が`pyscript.display()`で図をページに出すようになりました。
  `describe()`はチャンネルごとの音声プレーヤーも出します。これまでは図が`matplotlib.pyplot.show()`に
  渡されるだけで、ページには表示されませんでした。図を自分で作った`plot()`（`overlay=True`も含む）は、
  図を表示してから閉じます。そのため、戻り値のAxesをその後に変更してもページには反映されません。
  `plot("describe")`は`describe()`の部品なので、単独では表示しません。
  `py-editor`のセルには出力先がないので、これまでどおりの動作になります。
- `ChannelFrame.welch()` with `average="mean"` (the default) no longer needs
  about ten times the input in memory for a long recording. It now calls
  `scipy.signal.welch` on runs of whole segments and weights each run's mean by
  its segment count; the segments are the same, so the values match the single
  call up to float64 rounding. For 48 kHz, 2 channels, 10 minutes, the peak
  resident memory during `welch(n_fft=2048)` went from about 5.0 GB to 1.5 GB.
  `average="median"` still uses one call over all segments.
  `ChannelFrame.welch()`を`average="mean"`（既定）で使うとき、長い録音で入力の約10倍の
  メモリを使わなくなりました。セグメント単位の区間ごとに`scipy.signal.welch`を呼び、
  セグメント数で重み付けして平均します。セグメントは同じなので、値は1回の呼び出しと
  float64の丸め誤差の範囲で一致します。48 kHz・2 ch・10分の`welch(n_fft=2048)`では、
  最大常駐メモリが約5.0 GBから1.5 GBになりました。`average="median"`はこれまでどおり
  全セグメントで1回だけ呼びます。
