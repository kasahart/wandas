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
- Saving or loading WDF with `h5netcdf` installed but without its `h5py`
  backend now raises `ImportError` with `pip install "wandas[io]"`, like a
  missing `h5netcdf` already did. Before, the save or load failed with
  h5netcdf's own message, which did not name the extra. The `save` and
  `load` docstrings and the Recipe How-to now say that WDF needs the `io` extra.
  `h5netcdf`は入っていても`h5py`が無い環境でWDFを保存・読み込みすると、`h5netcdf`が無い場合と
  同じく`pip install "wandas[io]"`を示す`ImportError`になりました。これまではh5netcdf自身の
  メッセージになり、extraの名前が出ませんでした。`save`／`load`のdocstringとRecipe How-toにも、
  WDFには`io` extraが必要なことを書きました。
