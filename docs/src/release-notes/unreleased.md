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

## Added / 追加

- `RecipeExecutionError` now has `node_id`, `operation` and `cause`
  attributes. When an operation fails during `RecipePlan.apply`, they name the
  failing node, its registered operation id and the original exception (the
  same object as `__cause__`), so a tool can mark the failed step without
  parsing the message. Errors not tied to one node, such as mismatched input
  names, have `None` there. A `RecipeExecutionError` raised inside an
  operation (for example by a nested `apply`) is now wrapped like any other
  failure, so these attributes name the node of the plan being applied; its
  message appears after `Cause:`. When the error is pickled, `cause` is left
  out like `__cause__`. Other message texts are unchanged.
  `RecipeExecutionError`に`node_id`・`operation`・`cause`属性を追加しました。
  `RecipePlan.apply`で処理が失敗すると、失敗したノード、その登録済み操作ID、元の例外
  （`__cause__`と同じ）が入るので、エラー文を読まずに失敗した手順に印を付けられます。
  入力名の不一致のようにノードに関係しないエラーでは`None`です。操作の中で送出された
  `RecipeExecutionError`（入れ子の`apply`など）も他の失敗と同じく包むようになり、属性は実行中の
  Planのノードを指します（元のエラー文は`Cause:`の後に入ります）。pickleでは`cause`を
  `__cause__`と同じく含めません。それ以外のエラー文は変わりません。
