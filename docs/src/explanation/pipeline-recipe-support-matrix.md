# Pipeline Recipe Support Matrix / Recipe 対応表

このページは「Recipe にできる処理」と「まだ Recipe にしない処理」を先に見るための早見表です。

詳しい理由や内部の話は [Pipeline Recipe Extraction Boundaries](pipeline-recipe-extraction-boundaries.md) にあります。

## 判定の読み方

| 表示 | 意味 |
| --- | --- |
| 対応 | いま Recipe として使える |
| 一部対応 | 使える場合と使えない場合がある |
| 非対応 | いまは Recipe にしない |
| 手動なら対応 | 自動抽出はしないが、自分で Recipe を書けば使える |
| 未決定 | まだ保存形式や使い方を決めていない |

## まず見る表

| やりたいこと | 対応 | 使うもの | 例 | 注意 |
| --- | --- | --- | --- | --- |
| 1つの frame に処理を順番にかける | 対応 | `RecipeSpec.from_frame(...)` | `frame.remove_dc().normalize()` | まずはこれを使う |
| 基本的な信号処理を保存する | 対応 | `RecipeSpec` | `remove_dc`, `highpass_filter`, `trim`, `normalize` | Wandas が名前で呼べる処理が対象 |
| frame method を保存する | 一部対応 | `RecipeSpec` | `fix_length`, `sum`, `mean`, `channel_difference` | 対応済み method だけ |
| channel を選ぶ | 一部対応 | `RecipeSpec` | `frame[0:2]`, `frame[["left", "right"]]`, `get_channel("left")` | 名前、範囲、番号の一覧、True/False の一覧は対応 |
| 複雑な条件で channel を選ぶ | 非対応 | なし | 関数や正規表現で探す選び方 | 選び方の意味をあとで再現しにくい |
| time 方向の範囲を切り出す | 対応 | `RecipeSpec` | `frame[:, 100:400]` | 連続した範囲だけ対応 |
| time 方向で飛び飛びの点を選ぶ | 非対応 | なし | `frame[:, [1, 5, 9]]` | 時刻の基準を再現するルールが未整理 |
| 別の frame 型へ変換する | 一部対応 | `RecipeSpec` | `fft`, `stft`, `welch`, `ifft`, `istft` | 対応済み method だけ |
| 2つの frame を足す、引く、混ぜる | 対応 | `GraphRecipeSpec` または `NodeGraphRecipeSpec` | `signal + noise` | 1回だけ組み合わせるなら `GraphRecipeSpec`、複数回なら `NodeGraphRecipeSpec` |
| 数値 scalar と演算する | 対応 | `RecipeSpec` | `frame * 2.0`, `2.0 - frame` | NaN は非対応 |
| NumPy/Dask array と演算する | 対応 | `NodeGraphRecipeSpec` | `frame + offset_array` | array の中身は Recipe に保存しない |
| channel を追加する | 対応 | `NodeGraphRecipeSpec` | `base.add_channel(raw, label="ref")` | 追加データは外から渡す |
| import できる自作関数を使う | 一部対応 | `RecipeSpec` | `frame.apply(my_func, ...)` | Python module から import できる関数だけ |
| lambda や notebook 内だけの関数を使う | 非対応 | なし | `lambda x: x * 2` | あとで import できない |
| `rms` など frame ではない値を返す | 手動なら対応 | `TerminalStep` | `RecipeSpec([... , TerminalStep("rms")])` | frame からの自動抽出はしない |
| `plot()`, `info()`, `describe()` を保存する | 非対応 | なし | `frame.plot()` | 変換処理ではなく表示や確認のため |
| `compute()` や `persist()` を保存する | 非対応 | なし | `frame.persist()` | 計算の実行タイミングであり、処理内容ではない |
| WDF に Recipe を保存する | 未決定 | 未定 | `save(...).load(...)` | #257 で決める |
| sklearn 形式で Wandas の処理を使う | 一部対応 | `wandas.pipeline.sklearn` | `Pipeline([...])` | 完全な相互変換や export は #258 で決める |
| joblib/skops に export する | 未決定 | 未定 | `joblib.dump(...)` | #258 で決める |

## 迷ったときの選び方

| 状況 | 選ぶもの |
| --- | --- |
| 入力が1つで、結果も frame | `RecipeSpec.from_frame(processed)` |
| 入力が2つで、1回だけ組み合わせる | `GraphRecipeSpec.from_frame(processed, input_names=(...))` |
| 入力が複数、または `add_channel` や array 入力がある | `NodeGraphRecipeSpec.from_frame(processed, input_names=(...))` |
| 結果が NumPy array など frame ではない | `TerminalStep` を使って手で Recipe を書く |
| 表にないもの | 現時点では Recipe にしない。必要なら新しい issue で対応方針を決める |

## 次に整理すること

#259 では、この表の「一部対応」と「非対応」を中心に見直します。

優先する作業:

- 対応済みのものをこの表とテストでそろえる。
- 意味が変わりそうなものは `RecipeExtractionError` で止める。
- WDF 保存は #257、sklearn/export は #258 に残す。
