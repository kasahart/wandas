import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from scripts.learning_path_i18n import (
        docs_reference_links,
        language_switch_markdown,
        load_catalog,
        locale_from_argv,
        navigation_markdown,
    )

    locale = locale_from_argv()
    catalog = load_catalog("06_reusable_pipeline_recipes", locale)

    def t(key, **values):
        return catalog.text(key, **values)

    return (
        catalog,
        docs_reference_links,
        language_switch_markdown,
        locale,
        mo,
        navigation_markdown,
        t,
    )


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(f"# {t('title')}\n\n{t('intro')}")
    return


@app.cell(hide_code=True)
def _(language_switch_markdown, locale, mo):
    mo.md(language_switch_markdown("06_reusable_pipeline_recipes", locale))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("record_frame"))
    return


@app.cell
def _():
    import json

    import numpy as np

    import wandas as wd
    from wandas import pipeline as pipeline_api

    return json, np, pipeline_api, wd


@app.cell
def _(np, wd):
    template_signal = wd.from_numpy(
        np.array([[1.0, 2.0, 4.0, 7.0]]),
        sampling_rate=8_000,
        ch_labels=["sensor"],
    )
    template_result = template_signal.remove_dc().normalize()
    return template_result, template_signal


@app.cell(hide_code=True)
def _(mo, t, template_result):
    mo.md(t("template_history", operations=[record["operation"] for record in template_result.operation_history]))
    return


@app.cell(hide_code=True)
def _(mo, t, template_result):
    mo.md(t("lineage_explanation"))
    return


@app.cell
def _(pipeline_api, template_result):
    recipe_plan = pipeline_api.RecipePlan.from_frame(template_result, input_names=("signal",))
    recipe_payload = recipe_plan.to_dict()
    return (recipe_payload,)


@app.cell(hide_code=True)
def _(mo, recipe_payload, t):
    operation_ids = [node["operation"] for node in recipe_payload["nodes"]]
    mo.md(
        t(
            "recipe_table",
            input_name=recipe_payload["inputs"][0]["name"],
            node_count=len(operation_ids),
            operations=", ".join(operation_ids),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("recipe_schema"))
    return


@app.cell
def _(json, pipeline_api, recipe_payload):
    recipe_json = json.dumps(recipe_payload)
    loaded_recipe = pipeline_api.RecipePlan.from_dict(json.loads(recipe_json))
    return loaded_recipe, recipe_json


@app.cell(hide_code=True)
def _(mo, recipe_json, recipe_payload, t):
    mo.md(
        t(
            "schema_info",
            schema=recipe_payload["schema"],
            version=recipe_payload["version"],
            size=len(recipe_json.encode("utf-8")),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("json_round_trip"))
    return


@app.cell
def _(loaded_recipe, np, wd):
    runtime_signal = wd.from_numpy(
        np.array([[2.0, 5.0, 8.0, 14.0]]),
        sampling_rate=8_000,
        metadata={"recording": "next"},
        ch_labels=["sensor"],
    )
    replayed_signal = loaded_recipe.apply({"signal": runtime_signal})
    direct_signal = runtime_signal.remove_dc().normalize()
    return direct_signal, replayed_signal, runtime_signal


@app.cell(hide_code=True)
def _(direct_signal, mo, np, replayed_signal, runtime_signal, t):
    np.testing.assert_allclose(replayed_signal.data, direct_signal.data)
    assert replayed_signal.metadata == {"recording": "next"}
    assert runtime_signal.operation_history == []
    mo.md(
        t(
            "direct_result",
            metadata=replayed_signal.metadata,
            history=runtime_signal.operation_history,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("multiple_inputs"))
    return


@app.cell
def _(np, pipeline_api, wd):
    base_template = wd.from_numpy(np.array([[1.0, 1.0, 1.0, 1.0]]), sampling_rate=8_000)
    other_template = wd.from_numpy(np.array([[2.0, 2.0, 2.0, 2.0]]), sampling_rate=8_000)
    mix_template_result = base_template.mix(other_template)
    mix_recipe = pipeline_api.RecipePlan.from_frame(mix_template_result, input_names=("base", "other"))

    next_base = wd.from_numpy(np.array([[3.0, 3.0, 3.0, 3.0]]), sampling_rate=8_000)
    next_other = wd.from_numpy(np.array([[4.0, 4.0, 4.0, 4.0]]), sampling_rate=8_000)
    mixed_replay = mix_recipe.apply({"base": next_base, "other": next_other})
    return mix_recipe, mixed_replay


@app.cell(hide_code=True)
def _(mix_recipe, mixed_replay, mo, np, t):
    mix_values = mixed_replay.data
    np.testing.assert_allclose(mix_values, 7.0)
    mo.md(t("mix_result", inputs=[item.name for item in mix_recipe.inputs], values=mix_values.tolist()))
    return


@app.cell(hide_code=True)
def _(catalog, docs_reference_links, locale, mo, t):
    mo.md(t("summary", **docs_reference_links(locale, catalog)))
    return


@app.cell(hide_code=True)
def _(locale, mo, navigation_markdown):
    mo.md(navigation_markdown("06_reusable_pipeline_recipes", locale))
    return


if __name__ == "__main__":
    app.run()
