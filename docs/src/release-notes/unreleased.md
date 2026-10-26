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
- `RecipePlan.apply` accepts the Frame (or array) itself when the plan has
  one input: `plan.apply(frame)` is the same as
  `plan.apply({plan.inputs[0].name: frame})`. A plan with several inputs still
  needs the mapping; one value raises `RecipeExecutionError` with the input
  names. Passing a mapping works as before.
  入力が1つの`RecipePlan`は、`plan.apply(frame)`のようにFrame（または配列）をそのまま
  渡せるようになりました。`plan.apply({plan.inputs[0].name: frame})`と同じです。入力が
  複数のPlanに値を1つだけ渡すと、入力名を示した`RecipeExecutionError`になります。
  辞書での渡し方はこれまでどおり使えます。
- `RecipePlan.apply_files(paths, out_dir, *, root=None, jobs=1,
  on_error="continue", overwrite=False)` applies a one-input plan to many
  recordings: it reads each with `wandas.read`, applies the plan, and saves the
  result as WDF under `out_dir` (`a.wav` becomes `a.wav.wdf`; with `root`, the
  folders below it are kept). A failed recording is recorded and the rest go
  on, unless `on_error="raise"`. `jobs` runs recordings in separate processes.
  It returns a pandas table with `path`, `status`, `output`, `error` and
  `seconds` per recording.
  入力が1つの`RecipePlan`を多数の録音に掛ける`RecipePlan.apply_files`を追加しました。録音ごとに
  `wandas.read`で読み、Planを実行し、結果を`out_dir`にWDFで保存します（`a.wav`は`a.wav.wdf`。
  `root`を渡すとその下のフォルダ構成を保ちます）。失敗した録音は記録して先に進みます
  （`on_error="raise"`で最初の失敗で止まります）。`jobs`で別プロセスに分けて並列に実行します。
  戻り値は録音ごとの`path`・`status`・`output`・`error`・`seconds`を持つpandasの表です。
