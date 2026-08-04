import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    # Import the public APIs and validation libraries used in this lesson.
    import json

    import marimo as mo
    import numpy as np

    import wandas as wd
    from scripts.learning_path_i18n import (
        docs_relative_href,
        load_catalog,
        locale_from_argv,
        navigation_markdown,
    )
    from wandas import pipeline as pipeline_api

    locale = locale_from_argv()
    catalog = load_catalog("06_reusable_pipeline_recipes", locale)

    def t(key, **values):
        return catalog.text(key, **values)

    return docs_relative_href, json, locale, mo, navigation_markdown, np, pipeline_api, t, wd


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(f"# {t('title')}\n\n{t('intro')}")
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("record_frame"))
    return


@app.cell
def _(np, t, wd):
    # Apply the reusable preprocessing to a small representative signal.
    template_signal = wd.from_numpy(
        np.array([[1.0, 2.0, 4.0, 7.0]]),
        sampling_rate=8_000,
        ch_labels=["sensor"],
    )
    template_result = template_signal.remove_dc().normalize()

    print(t("template_history", operations=[record["operation"] for record in template_result.operation_history]))
    return template_result, template_signal


@app.cell(hide_code=True)
def _(mo, t, template_result):
    mo.md(t("lineage_explanation"))
    return


@app.cell
def _(pipeline_api, t, template_result):
    # Convert semantic lineage into a Recipe with named inputs.
    recipe_plan = pipeline_api.RecipePlan.from_frame(template_result, input_names=("signal",))
    recipe_payload = recipe_plan.to_dict()
    _operation_ids = [node["operation"] for node in recipe_payload["nodes"]]

    print(t("recipe_inputs", input_name=recipe_payload["inputs"][0]["name"]))
    print(t("recipe_operations", operations=_operation_ids))
    return recipe_payload


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("recipe_schema"))
    return


@app.cell
def _(json, pipeline_api, recipe_payload, t):
    # Serialize the schema and load the plan without sharing runtime objects.
    recipe_json = json.dumps(recipe_payload)
    loaded_recipe = pipeline_api.RecipePlan.from_dict(json.loads(recipe_json))

    print(
        t(
            "schema_info",
            schema=recipe_payload["schema"],
            version=recipe_payload["version"],
            size=len(recipe_json.encode("utf-8")),
        )
    )
    return (loaded_recipe,)


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("json_round_trip"))
    return


@app.cell
def _(loaded_recipe, np, t, wd):
    # Compare Recipe replay with the same Frame methods called directly.
    runtime_signal = wd.from_numpy(
        np.array([[2.0, 5.0, 8.0, 14.0]]),
        sampling_rate=8_000,
        metadata={"recording": "next"},
        ch_labels=["sensor"],
    )
    replayed_signal = loaded_recipe.apply({"signal": runtime_signal})
    direct_signal = runtime_signal.remove_dc().normalize()

    _replayed_values = replayed_signal.data
    _direct_values = direct_signal.data
    np.testing.assert_allclose(_replayed_values, _direct_values)
    assert replayed_signal.metadata == {"recording": "next"}
    assert runtime_signal.operation_history == []

    print(
        t(
            "direct_result",
            metadata=replayed_signal.metadata,
            history=runtime_signal.operation_history,
        )
    )
    return


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
    mo.md(t("multiple_inputs"))
    return


@app.cell
def _(np, pipeline_api, t, wd):
    # Build a two-input mix Recipe and apply it to another input pair.
    base_template = wd.from_numpy(np.array([[1.0, 1.0, 1.0, 1.0]]), sampling_rate=8_000)
    other_template = wd.from_numpy(np.array([[2.0, 2.0, 2.0, 2.0]]), sampling_rate=8_000)
    mix_template_result = base_template.mix(other_template)
    mix_recipe = pipeline_api.RecipePlan.from_frame(mix_template_result, input_names=("base", "other"))

    next_base = wd.from_numpy(np.array([[3.0, 3.0, 3.0, 3.0]]), sampling_rate=8_000)
    next_other = wd.from_numpy(np.array([[4.0, 4.0, 4.0, 4.0]]), sampling_rate=8_000)
    mixed_replay = mix_recipe.apply({"base": next_base, "other": next_other})
    _mix_values = mixed_replay.data
    np.testing.assert_allclose(_mix_values, 7.0)

    print(t("mix_result", inputs=[item.name for item in mix_recipe.inputs], values=_mix_values.tolist()))
    return


@app.cell(hide_code=True)
def _(docs_relative_href, locale, mo, t):
    mo.md(
        t(
            "summary",
            how_to_href=docs_relative_href(locale, "how-to/pipeline-recipes/"),
            api_href=docs_relative_href(locale, "api/pipeline/"),
        )
    )
    return


@app.cell(hide_code=True)
def _(locale, mo, navigation_markdown):
    mo.md(navigation_markdown("06_reusable_pipeline_recipes", locale))
    return


if __name__ == "__main__":
    app.run()
