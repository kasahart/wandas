# Pipeline Recipe Requirements / パイプラインレシピ要件定義

## Purpose / 目的

Wandas の信号処理チェーンを、Notebook やバッチ処理で再利用できる小さな「レシピ」として表現する。scikit-learn の `Pipeline` を正本にせず、Wandas の `Frame.apply_operation(...)` と operation history を正本にすることで、既存のフレーム不変性、メタデータ伝播、Dask lazy execution を保つ。

この試作の対象は、単一入力の `ChannelFrame` に対する直列の既存 Wandas operation である。WDF 保存、sklearn Pipeline との自動相互変換、lambda や任意関数の replayable 判定、Dask-ML 連携は対象外とする。

## Target Users / 対象ユーザー

- 実験 Notebook で作った前処理手順を、後で同じ順序・同じパラメータで再実行したい信号処理ユーザー。
- Wandas の `Frame` 操作を使いながら、sklearn 風の `fit` / `transform` インターフェースで前処理を組みたい ML ユーザー。
- operation history で「どの処理をいつ適用したか」を追跡し、分析ノートやレビューで説明したい開発者。

## Functional Requirements / 機能要件

| ID | Requirement | Acceptance Criteria |
| --- | --- | --- |
| R1 | Replayable な単一 operation を表現できる | `OperationSpec(operation, params)` が operation registry key とパラメータを保持する |
| R2 | 複数 operation を直列適用できる | `RecipeSpec.apply(frame)` が `frame.apply_operation(...)` を順に呼び、最後の frame を返す |
| R3 | 元 frame を破壊しない | レシピ適用後も入力 frame の data、metadata、operation history は変わらない |
| R4 | Dask laziness を維持する | Dask-backed frame に適用しても `.compute()` を呼ばず、lazy graph を返す |
| R5 | sklearn 互換 transformer を提供する | `fit(X, y=None) -> self`、`transform(X)`、`get_params()`、`set_params()`、`to_spec()` が動く |
| R6 | 代表的な signal operation をクラス化する | `HighPassFilter`、`LowPassFilter`、`BandPassFilter`、`Normalize`、`RemoveDC` を提供する |
| R7 | sklearn は optional dependency とする | 通常の `import wandas` では sklearn を import せず、`wandas.pipeline.sklearn` import 時だけ必要にする |
| R8 | 未導入時の案内を明確にする | sklearn 未導入時は `pip install "wandas[sklearn]"` を含む `ImportError` を出す |

## Non-Functional Requirements / 非機能要件

- API surface は小さく保つ。v1 では recipe serialization schema や WDF 永続化を追加しない。
- operation 名は frame method 名ではなく registry key を使う。例: `high_pass_filter(...)` ではなく `"highpass_filter"`。
- `RecipeSpec` と `OperationSpec` は実行時に外部からパラメータを書き換えられない構造にする。
- sklearn adapter は薄い境界に留め、数値処理や frame 作成ロジックを持たない。

## Out of Scope / スコープ外

- `RecipeSpec` と `sklearn.pipeline.Pipeline` の自動相互変換。
- WDF への `recipe_json` 保存。
- `FunctionTransformer`、lambda、任意 Python callable の replayable 判定。
- 複数入力 operation、FFT/STFT など domain transition を含む recipe の一般化。
- joblib/skops export、Dask-ML 連携、モデル学習結果の保存。

## Success Metrics / 成功指標

- Notebook 上で `RecipeSpec` と sklearn `Pipeline` の両方を実行し、同じ operation history を確認できる。
- 既存の Wandas operation history に、レシピ適用順とパラメータが記録される。
- `uv run pytest tests/test_optional_dependencies.py tests/pipeline` が通る。
- `uv run ruff check wandas tests --config=pyproject.toml -v` と `uv run ty check wandas tests` が通る。
