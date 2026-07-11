import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import wandas as wd
    from wandas.pipeline import OperationSpec, RecipeSpec

    operation_spec_type = OperationSpec
    recipe_spec_type = RecipeSpec

    plt.rcParams["figure.figsize"] = (10, 4)

    return json, mo, np, operation_spec_type, plt, recipe_spec_type, wd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Wandas Recipe UX
    ## Frame-first replay / frame起点の再現実験

    Start with normal `Frame` operations. Do not start by building a pipeline object.

    まずは通常の `Frame` 操作から始めます。
    ユーザーが最初に `OperationSpec` や sklearn `Pipeline` を組み立てる必要はありません。

    The beginner workflow is:

    ```python
    explored = frame.remove_dc().high_pass_filter(cutoff=100.0).normalize()
    recipe = RecipeSpec.from_frame(explored)
    reproduced = recipe.apply(other_frame)
    ```

    Direct recipe construction and sklearn adapters are optional tools for advanced or integration use cases.

    明示的なRecipe構築やsklearn adapterは、必要になったときに使うオプションです。
    """)
    return


@app.cell
def _(np, wd):
    def make_demo_frame(freq: float, *, label: str):
        sampling_rate = 16000
        duration = 1.0
        time = np.linspace(0.0, duration, int(sampling_rate * duration), endpoint=False)
        signal = (
            0.35
            + 0.7 * np.sin(2 * np.pi * 50 * time)
            + np.sin(2 * np.pi * freq * time)
            + 0.15 * np.sin(2 * np.pi * 3000 * time)
        )
        frame = wd.from_numpy(signal.reshape(1, -1), sampling_rate=sampling_rate, ch_labels=[label])
        return frame, time

    frame, time = make_demo_frame(1000.0, label="trial-a")
    other_frame, _ = make_demo_frame(1250.0, label="trial-b")

    print("frames ready")
    print("frame sampling_rate:", frame.sampling_rate)
    print("frame shape:", frame.data.shape)
    print("initial operation_history:", frame.operation_history)

    return frame, make_demo_frame, other_frame, time


@app.cell
def _():
    def channel_values(target_frame):
        return target_frame.data.reshape(-1)

    return (channel_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Explore with frame methods / frame methodで探索する

    This is ordinary Wandas analysis code. The result frame records lineage internally.

    これは通常のWandas解析コードです。結果のframeには内部的にlineageが残ります。
    """)
    return


@app.cell
def _(channel_values, frame, plt, time):
    explored = frame.remove_dc().high_pass_filter(cutoff=100.0, order=2).normalize()

    fig, method_ax = plt.subplots()
    method_ax.plot(time[:800], channel_values(frame)[:800], label="raw", alpha=0.65)
    method_ax.plot(time[:800], channel_values(explored)[:800], label="explored", alpha=0.9)
    method_ax.set_title("Frame method chain")
    method_ax.set_xlabel("Time [s]")
    method_ax.set_ylabel("Amplitude")
    method_ax.grid(True, alpha=0.3)
    method_ax.legend()

    print("explored operation_history:")
    for explored_record in explored.operation_history:
        print("  ", explored_record)

    fig
    return explored, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Extract a recipe from the result / 結果frameからRecipeを抽出する

    `RecipeSpec.from_frame(explored)` reads the frame's `operation_graph`.
    It does not read notebook source code, and it does not mutate the original frame.

    `RecipeSpec.from_frame(explored)` はframeの `operation_graph` を読みます。
    Notebookのソースコードを解析するわけではなく、元のframeも破壊しません。
    """)
    return


@app.cell
def _(explored, recipe_spec_type):
    recipe = recipe_spec_type.from_frame(explored)

    print("extracted recipe steps:")
    for step in recipe.steps:
        print("  ", step)

    return (recipe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Replay on another frame / 別frameで再現する

    A recipe preserves the processing steps and parameters, not the original signal values.

    Recipeが保存するのは元信号の値ではなく、処理手順とパラメータです。
    """)
    return


@app.cell
def _(channel_values, explored, np, other_frame, plt, recipe, time):
    reproduced = recipe.apply(other_frame)
    expected = other_frame.remove_dc().high_pass_filter(cutoff=100.0, order=2).normalize()

    np.testing.assert_allclose(reproduced.data, expected.data)

    fig_replay, replay_ax = plt.subplots()
    replay_ax.plot(time[:800], channel_values(explored)[:800], label="trial-a explored", alpha=0.65)
    replay_ax.plot(time[:800], channel_values(reproduced)[:800], label="trial-b replayed", alpha=0.9)
    replay_ax.set_title("Same recipe, different frame")
    replay_ax.set_xlabel("Time [s]")
    replay_ax.set_ylabel("Amplitude")
    replay_ax.grid(True, alpha=0.3)
    replay_ax.legend()

    print("replayed operation_history:")
    for replay_record in reproduced.operation_history:
        print("  ", replay_record)
    print("source frame history remains:", other_frame.operation_history)

    fig_replay
    return expected, fig_replay, reproduced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Optional: inspect or persist the steps / 任意: stepを読む

    Users usually do not need to write `OperationSpec` by hand.
    The extracted steps are still readable when you need auditability.

    通常のユーザーは `OperationSpec` を手書きする必要はありません。
    監査や確認が必要なときだけ、抽出済みstepを読めます。
    """)
    return


@app.cell
def _(json, recipe):
    extracted_summary = [{"operation": step.operation, "params": dict(step.params)} for step in recipe.steps]

    print(json.dumps(extracted_summary, indent=2, ensure_ascii=False))
    return (extracted_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Optional: build a recipe directly / 任意: Recipeを直接構成する

    Direct construction is useful for tests, generated configs, or explicit specifications.
    It is not the beginner entry point.

    直接構成は、テスト、生成された設定、明示仕様には便利です。
    初学者向けの入口ではありません。
    """)
    return


@app.cell
def _(frame, np, operation_spec_type, recipe_spec_type, reproduced):
    explicit_recipe = recipe_spec_type(
        [
            operation_spec_type("remove_dc"),
            operation_spec_type("highpass_filter", {"cutoff": 100.0, "order": 2}),
            operation_spec_type(
                "normalize",
                {"norm": np.inf, "axis": -1, "threshold": None, "fill": None},
            ),
        ]
    )

    explicit_result = explicit_recipe.apply(frame)
    explicit_expected = frame.remove_dc().high_pass_filter(100.0, order=2).normalize()
    np.testing.assert_allclose(explicit_result.data, explicit_expected.data)

    print("explicit recipe works, but it is optional")
    print("replayed result shape:", reproduced.data.shape)
    return explicit_expected, explicit_recipe, explicit_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Optional: sklearn adapter / 任意: sklearn adapter

    Use the sklearn adapter only when another tool already expects `fit` / `transform`.
    Wandas Recipe remains the source of truth.

    sklearn adapterは、外部ツールが `fit` / `transform` を期待する場合の連携手段です。
    正本はWandas Recipeです。
    """)
    return


@app.cell
def _(frame, json, np):
    try:
        from sklearn.pipeline import Pipeline

        from wandas.pipeline.sklearn import HighPassFilter, Normalize, RemoveDC
    except ImportError as exc:
        print('sklearn adapter is optional. Install it with: pip install "wandas[sklearn]"')
        print("import error:", exc)
        sklearn_processed = None
    else:
        sklearn_pipeline = Pipeline(
            [
                ("dc", RemoveDC()),
                ("hp", HighPassFilter(cutoff=100.0, order=2)),
                ("norm", Normalize(norm=np.inf, axis=-1, threshold=None, fill=None)),
            ]
        )
        sklearn_processed = sklearn_pipeline.transform(frame)
        print(json.dumps(sklearn_processed.operation_history, indent=2, ensure_ascii=False))

    return (sklearn_processed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Summary / まとめ

    Recommended order for users:

    1. Explore with normal frame methods.
    2. Extract with `RecipeSpec.from_frame(processed)`.
    3. Replay with `recipe.apply(other_frame)`.
    4. Use explicit recipe construction or sklearn adapter only when the workflow needs it.

    ユーザー向けの推奨順序:

    1. 通常のframe methodで探索する。
    2. `RecipeSpec.from_frame(processed)` で抽出する。
    3. `recipe.apply(other_frame)` で別frameに再現する。
    4. 明示Recipe構築やsklearn adapterは、必要なワークフローだけで使う。
    """)
    return


if __name__ == "__main__":
    app.run()
