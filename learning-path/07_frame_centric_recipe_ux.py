import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 07 Frame-Centric Recipe UX
    ## Pipelineを意識しない探索解析と再現実験

    This marimo app explains the target UX for Wandas Recipe:
    users should explore with normal `Frame` operations, turn the result into a replayable recipe,
    and run the same analysis on another compatible frame.

    このmarimoアプリでは、Wandas RecipeのあるべきUXを確認します。
    ユーザーは通常の `Frame` 操作で探索し、その結果をRecipe化し、別の互換Frameで同じ解析を再現できるべきです。

    The goal is frame-centric:

    ```python
    processed = frame.remove_dc().high_pass_filter(cutoff=100).normalize()
    recipe = recipe_from_frame(processed)
    replayed = recipe.replay(other_frame)
    ```

    The user should not need to start by choosing `OperationSpec`, `GraphRecipeSpec`, or `TerminalStep`.
    Those classes can remain implementation details or advanced escape hatches.
    """)
    return


@app.cell
def _():
    from collections.abc import Mapping
    from dataclasses import dataclass
    from typing import Any

    import numpy as np

    from wandas.frames.channel import ChannelFrame
    from wandas.pipeline import (
        GraphRecipeSpec,
        NodeGraphRecipeSpec,
        RecipeExtractionError,
        RecipeSpec,
        TerminalStep,
    )

    any_value = Any
    channel_frame_type = ChannelFrame
    graph_recipe_spec_type = GraphRecipeSpec
    mapping_type = Mapping
    node_graph_recipe_spec_type = NodeGraphRecipeSpec
    recipe_extraction_error_type = RecipeExtractionError
    recipe_spec_type = RecipeSpec
    terminal_step_type = TerminalStep

    return (
        any_value,
        channel_frame_type,
        dataclass,
        graph_recipe_spec_type,
        mapping_type,
        node_graph_recipe_spec_type,
        np,
        recipe_extraction_error_type,
        recipe_spec_type,
        terminal_step_type,
    )


@app.cell
def _(channel_frame_type, np):
    def make_frame(freq: float, *, label: str) -> channel_frame_type:
        sampling_rate = 8000
        time = np.linspace(0.0, 0.1, int(sampling_rate * 0.1), endpoint=False)
        data = (0.2 + np.sin(2 * np.pi * freq * time)).reshape(1, -1)
        return channel_frame_type.from_numpy(data, sampling_rate=sampling_rate, label=label)

    frame = make_frame(440.0, label="trial-a")
    other_frame = make_frame(660.0, label="trial-b")
    noise = make_frame(1200.0, label="noise-a")
    other_noise = make_frame(1500.0, label="noise-b")

    print("frames ready")
    print("frame shape:", frame.shape)
    print("other_frame shape:", other_frame.shape)
    return frame, make_frame, noise, other_frame, other_noise


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Current supported happy path / 現在サポート済みの基本導線

    The current implementation already supports the most important beginner workflow:

    1. Work with a normal frame method chain.
    2. Extract a recipe from the processed frame.
    3. Replay it on another frame.

    現在の実装でも、初学者にとって最も重要な流れは成立しています。

    1. 通常のframe method chainで探索する。
    2. 処理済みframeからRecipeを抽出する。
    3. 別のframeへreplayする。
    """)
    return


@app.cell
def _(frame, np, other_frame, recipe_spec_type):
    explored = frame.remove_dc().high_pass_filter(cutoff=100.0, order=2).normalize(norm=2.0)

    current_recipe = recipe_spec_type.from_frame(explored)
    replayed = current_recipe.apply(other_frame)
    expected = other_frame.remove_dc().high_pass_filter(cutoff=100.0, order=2).normalize(norm=2.0)

    np.testing.assert_allclose(replayed.data, expected.data)

    print("current API: RecipeSpec.from_frame(processed)")
    print("extracted steps:")
    for step in current_recipe.steps:
        print("  ", step)
    print("replayed history:")
    for replay_record in replayed.operation_history:
        print("  ", replay_record)

    return current_recipe, expected, explored, replayed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Target facade / あるべき単一入口

    The UX gap is not the replay engine. The gap is the entry point.
    Users should not decide whether a result needs `RecipeSpec`, `GraphRecipeSpec`, or `NodeGraphRecipeSpec`.

    replay engineそのものではなく、入口が課題です。
    ユーザーが `RecipeSpec`、`GraphRecipeSpec`、`NodeGraphRecipeSpec` のどれを使うべきかを
    先に判断しなくてよい形が望ましいです。

    The next cell implements a small local facade to show the intended API shape.
    This is notebook code, not yet a Wandas public API.

    次のセルでは、あるべきAPI形状を小さなローカルfacadeとして実装します。
    これはNotebook内の説明用コードであり、まだWandasのpublic APIではありません。
    """)
    return


@app.cell
def _(
    any_value,
    dataclass,
    graph_recipe_spec_type,
    mapping_type,
    node_graph_recipe_spec_type,
    recipe_extraction_error_type,
    recipe_spec_type,
    terminal_step_type,
):
    @dataclass(frozen=True)
    class FrameCentricRecipe:
        recipe: any_value
        mode: str
        source_names: tuple[str, ...] = ()

        def replay(self, frame: any_value | None = None, **sources: any_value) -> any_value:
            if self.mode == "single":
                if frame is None:
                    if len(sources) != 1:
                        raise TypeError("single-input replay needs one frame")
                    frame = next(iter(sources.values()))
                return self.recipe.apply(frame)
            if frame is not None:
                raise TypeError("graph replay uses named sources, not a positional frame")
            return self.recipe.apply(sources)

        def then(self, terminal: str) -> "FrameCentricRecipe":
            if self.mode != "single":
                raise NotImplementedError("terminal facade for graph recipes needs a separate UX decision")
            return FrameCentricRecipe(
                recipe_spec_type([*self.recipe.steps, terminal_step_type(terminal)]),
                mode="single",
                source_names=self.source_names,
            )

        def summary(self) -> str:
            if self.mode == "single":
                return f"single-input recipe, {len(self.recipe.steps)} steps"
            return f"graph recipe, sources={list(self.source_names)}"

    def recipe_from_frame(
        processed: any_value,
        *,
        sources: mapping_type[str, any_value] | None = None,
    ) -> FrameCentricRecipe:
        if sources is None:
            try:
                return FrameCentricRecipe(recipe_spec_type.from_frame(processed), mode="single")
            except recipe_extraction_error_type as exc:
                raise recipe_extraction_error_type(
                    "This result is not a single-input linear frame recipe. "
                    "Pass sources={...} for graph or external-input replay."
                ) from exc

        source_names = tuple(sources)
        try:
            graph_recipe = graph_recipe_spec_type.from_frame(processed, input_names=source_names)
        except recipe_extraction_error_type:
            graph_recipe = node_graph_recipe_spec_type.from_frame(processed, input_names=source_names)
        return FrameCentricRecipe(graph_recipe, mode="graph", source_names=source_names)

    return FrameCentricRecipe, recipe_from_frame


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Beginner workflow with the facade / facadeを使った初学者向けワークフロー

    The user still writes Wandas frame code.
    The only new concept is: "remember this processed frame as a recipe."

    ユーザーが書くのは引き続きWandasのframe codeです。
    新しく覚える概念は「この処理済みframeをRecipeとして覚える」だけです。
    """)
    return


@app.cell
def _(expected, explored, np, other_frame, recipe_from_frame):
    remembered = recipe_from_frame(explored)
    reproduced = remembered.replay(other_frame)

    np.testing.assert_allclose(reproduced.data, expected.data)

    print(remembered.summary())
    print("reproduced history:")
    for reproduced_record in reproduced.operation_history:
        print("  ", reproduced_record)

    return remembered, reproduced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reproducible experiments / 再現実験

    A recipe becomes useful when the second frame is not the same signal.
    The recipe should preserve the processing contract, not the original signal values.

    2つ目のframeが同じ信号でないときにRecipeの価値が出ます。
    Recipeは元の信号値ではなく、処理手順とパラメータを保存します。
    """)
    return


@app.cell
def _(np, other_frame, remembered):
    experiment_result = remembered.replay(other_frame)

    before_rms = other_frame.rms
    after_rms = experiment_result.rms

    print("before rms:", np.round(before_rms, 6))
    print("after rms:", np.round(after_rms, 6))
    print("same frame object:", experiment_result is other_frame)
    print("source history unchanged:", other_frame.operation_history)

    return after_rms, before_rms, experiment_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph workflows should still feel frame-centric / Graphでもframe中心でありたい

    Current Wandas can replay supported graph recipes, but today users must choose graph-specific classes.
    The facade hides that choice and lets the user provide source frames by domain names.

    現在のWandasは対応済みgraph recipeをreplayできますが、現状ではユーザーがgraph専用classを選ぶ必要があります。
    facadeではその判断を隠し、ユーザーはドメイン名でsource frameを渡すだけにします。
    """)
    return


@app.cell
def _(frame, noise, np, other_frame, other_noise, recipe_from_frame):
    mixed = frame.remove_dc() + noise.normalize(norm=2.0)

    mix_recipe = recipe_from_frame(
        mixed,
        sources={"signal": frame, "noise": noise},
    )
    remixed = mix_recipe.replay(signal=other_frame, noise=other_noise)
    expected_mix = other_frame.remove_dc() + other_noise.normalize(norm=2.0)

    np.testing.assert_allclose(remixed.data, expected_mix.data)

    print(mix_recipe.summary())
    print("remixed history:")
    for remixed_record in remixed.operation_history:
        print("  ", remixed_record)

    return expected_mix, mix_recipe, mixed, remixed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## External data and add_channel / 外部データとadd_channel

    `add_channel(raw_array)` is a normal frame operation, so the ideal UX should not force users
    to learn graph internals.
    The source data still has to be supplied at replay time, but the user can think in terms of named data sources.

    `add_channel(raw_array)` は通常のframe操作です。
    したがって、理想的なUXではユーザーがgraph internalsを学ばなくてもよいはずです。
    replay時には外部データを渡す必要がありますが、ユーザーは名前付きデータソースとして考えられます。
    """)
    return


@app.cell
def _(frame, np, other_frame, recipe_from_frame):
    raw = np.zeros(frame.n_samples)
    processed_with_raw = frame.add_channel(raw, label="raw", source_time_offset=0.0).normalize()

    raw_recipe = recipe_from_frame(
        processed_with_raw,
        sources={"base": frame, "raw": raw},
    )

    new_raw = np.ones(other_frame.n_samples) * 0.01
    raw_replayed = raw_recipe.replay(base=other_frame, raw=new_raw)
    expected_raw = other_frame.add_channel(new_raw, label="raw", source_time_offset=0.0).normalize()

    np.testing.assert_allclose(raw_replayed.data, expected_raw.data)

    print(raw_recipe.summary())
    print("labels:", raw_replayed.labels)

    return new_raw, processed_with_raw, raw, raw_recipe, raw_replayed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Terminal values / terminal値

    Current frame terminal values such as `frame.rms` return arrays, not frames.
    Arrays do not carry `operation_graph`, so current public API needs `TerminalStep`.

    ただし、初学者向けUXとしては `TerminalStep` を直接見せたくありません。
    あるべき姿は、処理済みframeを覚えた後に `.then("rms")` のようにterminalを追加する形です。
    """)
    return


@app.cell
def _(explored, np, other_frame, recipe_from_frame):
    rms_recipe = recipe_from_frame(explored).then("rms")
    reproduced_rms = rms_recipe.replay(other_frame)
    expected_rms = other_frame.remove_dc().high_pass_filter(cutoff=100.0, order=2).normalize(norm=2.0).rms

    np.testing.assert_allclose(reproduced_rms, expected_rms)

    print(rms_recipe.summary())
    print("rms:", np.round(reproduced_rms, 6))

    return expected_rms, reproduced_rms, rms_recipe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boundaries should explain the next frame-centric action / 境界は次の行動を説明する

    `RecipeExtractionError` should not feel like a pipeline failure.
    It should tell the user what frame-centric information is missing.

    `RecipeExtractionError` はpipelineの失敗ではなく、
    Recipe化に必要なframe中心の情報が足りないことを説明するべきです。
    """)
    return


@app.cell
def _(frame, noise, recipe_extraction_error_type, recipe_from_frame):
    graph_result = frame + noise

    try:
        recipe_from_frame(graph_result)
    except recipe_extraction_error_type as exc:
        print("without sources:")
        print(str(exc))

    with_sources = recipe_from_frame(
        graph_result,
        sources={"signal": frame, "noise": noise},
    )
    print("\nwith sources:")
    print(with_sources.summary())

    return graph_result, with_sources


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this means for the product / プロダクトとしてのあるべき姿

    The frame-centric API should make these paths feel like one idea:

    - `recipe_from_frame(processed)`
      - single-input frame result
    - `recipe_from_frame(processed, sources={...})`
      - multi-input frame graph, external arrays, `add_channel`
    - `recipe_from_frame(processed).then("rms")`
      - terminal result after a replayable frame chain

    A future public Wandas API could expose this as:

    ```python
    recipe = processed.to_recipe()
    recipe = processed.to_recipe(sources={"signal": signal, "noise": noise})
    rms_recipe = processed.to_recipe().then("rms")
    ```

    The implementation can still use `RecipeSpec`, `GraphRecipeSpec`, `NodeGraphRecipeSpec`,
    and step classes internally.
    初学者ユーザーには、まずframe操作と再現実験だけが見える状態を目指します。
    """)
    return


if __name__ == "__main__":
    app.run()
