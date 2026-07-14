import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    # この教材で使う公開APIと検証用ライブラリを読み込む
    import json

    import marimo as mo
    import numpy as np

    import wandas as wd
    from wandas import pipeline as pipeline_api

    return json, mo, np, pipeline_api, wd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 同じ信号処理をRecipeとして再利用する

    複数の録音へ同じ前処理を適用するとき、メソッドチェーンを各所へコピーすると、
    パラメータ変更や処理順の確認が難しくなります。`RecipePlan` は、通常のFrame操作から
    「どの公開操作を、どの順番とパラメータで適用するか」を取り出します。

    Recipeは波形データそのものを保存しません。別のFrameを名前付き入力として渡すと、
    同じ処理をlazyなFrameグラフとして組み立て直します。

    この教材では次を確認します。

    1. 通常のFrame操作からRecipeを作る
    2. JSONとして保存できるschemaへ変換し、読み戻す
    3. 別のFrameへ適用し、直接呼び出した結果と一致することを確かめる
    4. 複数Frameを使う処理でも入力名が明示されることを確かめる
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 普通のFrame操作を記録する

    Recipe専用のbuilderは使いません。代表入力に対して、再利用したい公開Frame操作を
    そのまま呼び出します。ここではDC成分を除き、振幅を正規化します。
    """)
    return


@app.cell
def _(np, wd):
    # 再利用したい前処理を、代表となる小さな信号へ適用する
    template_signal = wd.from_numpy(
        np.array([[1.0, 2.0, 4.0, 7.0]]),
        sampling_rate=8_000,
        ch_labels=["sensor"],
    )
    template_result = template_signal.remove_dc().normalize()

    print("代表入力の履歴:", [record["operation"] for record in template_result.operation_history])
    return template_result, template_signal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `RecipePlan.from_frame()` は結果Frameのsemantic lineageを読みます。公開操作1回が
    Recipe node 1件になり、元のサンプル値はplanへ入りません。`input_names` は、あとで
    runtime dataを渡すための名前です。
    """)
    return


@app.cell
def _(pipeline_api, template_result):
    # semantic lineageを名前付き入力を持つRecipeへ変換する
    recipe_plan = pipeline_api.RecipePlan.from_frame(template_result, input_names=("signal",))
    recipe_payload = recipe_plan.to_dict()
    _operation_ids = [node["operation"] for node in recipe_payload["nodes"]]

    print("入力名:", recipe_payload["inputs"][0]["name"])
    print("Recipe operations:", _operation_ids)
    return recipe_payload


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. JSONで保存できる形へ変換する

    `to_dict()` の結果はschema version付きのJSON-compatible valueです。
    payloadはliveなPython operation objectを保存せず、stable operation IDを保持します。
    `RecipePlan.from_dict()` はIDをbuilt-in registryで解決し、未知のfield、operation、version、
    不正なgraphを拒否します。extensionではextract/load/applyに同じregistryを渡します。
    """)
    return


@app.cell
def _(json, pipeline_api, recipe_payload):
    # schemaをJSON文字列にし、runtime objectを共有せずにplanを読み戻す
    recipe_json = json.dumps(recipe_payload)
    loaded_recipe = pipeline_api.RecipePlan.from_dict(json.loads(recipe_json))

    print("Schema:", recipe_payload["schema"], recipe_payload["version"])
    print("JSON size:", len(recipe_json.encode("utf-8")), "bytes")
    return (loaded_recipe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 別のFrameへ適用する

    planに記録されるのは操作intentです。サンプル、metadata、label、sampling rateは
    `apply()` へ渡すruntime Frameが所有します。`apply()` 自体はlazy graphを作るだけで、
    このセルでは検証のため最後に `compute()` します。
    """)
    return


@app.cell
def _(loaded_recipe, np, wd):
    # Recipe replayと同じFrameメソッドの直接呼び出しを比較する
    runtime_signal = wd.from_numpy(
        np.array([[2.0, 5.0, 8.0, 14.0]]),
        sampling_rate=8_000,
        metadata={"recording": "next"},
        ch_labels=["sensor"],
    )
    replayed_signal = loaded_recipe.apply({"signal": runtime_signal})
    direct_signal = runtime_signal.remove_dc().normalize()

    _replayed_values = replayed_signal.compute()
    _direct_values = direct_signal.compute()
    np.testing.assert_allclose(_replayed_values, _direct_values)
    assert replayed_signal.metadata == {"recording": "next"}
    assert runtime_signal.operation_history == []

    print("直接呼び出しと一致: yes")
    print("runtime metadata:", replayed_signal.metadata)
    print("元のruntime Frameの履歴:", runtime_signal.operation_history)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 複数入力も名前で区別する

    Frame同士の演算、`mix()`、NumPy/Dask operandは、追加のruntime inputになります。
    値やNumPy/Daskというcontainer種別はschemaへ埋め込まれません。

    次の例では `base` と `other` を明示し、別の2つのFrameへmix Recipeを適用します。
    `mix()` はsource timeではなく現在のarray indexで信号を重ねます。
    """)
    return


@app.cell
def _(np, pipeline_api, wd):
    # 2つのFrame入力を持つmix Recipeを作り、別の入力ペアへ適用する
    base_template = wd.from_numpy(np.array([[1.0, 1.0, 1.0, 1.0]]), sampling_rate=8_000)
    other_template = wd.from_numpy(np.array([[2.0, 2.0, 2.0, 2.0]]), sampling_rate=8_000)
    mix_template_result = base_template.mix(other_template)
    mix_recipe = pipeline_api.RecipePlan.from_frame(mix_template_result, input_names=("base", "other"))

    next_base = wd.from_numpy(np.array([[3.0, 3.0, 3.0, 3.0]]), sampling_rate=8_000)
    next_other = wd.from_numpy(np.array([[4.0, 4.0, 4.0, 4.0]]), sampling_rate=8_000)
    mixed_replay = mix_recipe.apply({"base": next_base, "other": next_other})
    _mix_values = mixed_replay.compute()
    np.testing.assert_allclose(_mix_values, 7.0)

    print("mix入力:", [item.name for item in mix_recipe.inputs])
    print("replay結果:", _mix_values.tolist())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## まとめ

    - いつもどおりFrameメソッドを呼び、結果から`RecipePlan`を作る
    - Recipeは操作intentとgraphだけを持ち、波形やDask graphを保存しない
    - 実行時のoperation behaviorはstable IDに対応するregistry handlerが供給する
    - `to_dict()` / `from_dict()` でstrictなschemaを往復する
    - `apply()` では抽出時に決めた名前でruntime inputを渡す
    - replay後もruntime Frameのmetadataを保持し、入力Frameは変更しない
    - 任意の`Frame.apply(callable)`はruntime-onlyで、portable Recipeにはならない

    詳細な入力形状と制約は
    [RecipePlan how-to](../how-to/pipeline-recipes/) を、API signatureは
    [Pipeline API reference](../api/pipeline/) を参照してください。
    """)
    return


if __name__ == "__main__":
    app.run()
