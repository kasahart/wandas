import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from scripts.learning_path_i18n import (
        docs_relative_href,
        language_switch_markdown,
        load_catalog,
        locale_from_argv,
        navigation_markdown,
    )

    locale = locale_from_argv()
    catalog = load_catalog("08_metadata_driven_dataset_search", locale)

    def t(key, **values):
        return catalog.text(key, **values)

    return docs_relative_href, language_switch_markdown, locale, mo, navigation_markdown, t


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(f"# {t('title')}\n\n{t('intro')}")
    return


@app.cell(hide_code=True)
def _(language_switch_markdown, locale, mo):
    mo.md(language_switch_markdown("08_metadata_driven_dataset_search", locale))
    return


@app.cell
def _():
    import pathlib

    import pandas as pd

    import wandas as wd

    return pathlib, pd, wd


@app.cell
def _(pathlib):
    root = pathlib.Path(__file__).parent / "data" / "metadata_search"
    _relative_paths = sorted(path.relative_to(root) for path in root.rglob("*.wav"))
    file_count = len(_relative_paths)
    assert file_count == 3
    return file_count, root


@app.cell(hide_code=True)
def _(file_count, mo, root, t):
    mo.md(t("discovery_result", root=root, count=file_count))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("path_metadata_section"))
    return


@app.cell
def _(root, wd):
    dataset = wd.from_folder(
        str(root),
        recursive=True,
        file_extensions=[".wav"],
        path_metadata=True,
    )
    assert dataset.get_metadata()["lazy_loading"] is True
    return (dataset,)


@app.cell(hide_code=True)
def _(dataset, mo, t):
    dataset_metadata = dataset.get_metadata()
    mo.md(
        t(
            "dataset_result",
            count=len(dataset),
            lazy_loading=dataset_metadata["lazy_loading"],
            loaded_count=dataset_metadata["loaded_count"],
        )
    )
    return


@app.cell
def _(dataset):
    selected = dataset.select(partition_0="group_a", partition_1="batch_01")
    assert len(selected) == 1
    return (selected,)


@app.cell(hide_code=True)
def _(mo, selected, t):
    mo.md(t("selection_result", count=len(selected)))
    return


@app.cell
def _(dataset):
    try:
        dataset.select(missing_key="value")
    except KeyError as error:
        unknown_key_error = type(error).__name__
    else:
        raise AssertionError("Unknown metadata keys must raise KeyError")

    empty_selection = dataset.select(partition_0="missing_group")
    assert len(empty_selection) == 0
    return empty_selection, unknown_key_error


@app.cell(hide_code=True)
def _(empty_selection, mo, t, unknown_key_error):
    mo.md(
        t(
            "selection_contract",
            error=unknown_key_error,
            empty_count=len(empty_selection),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("lazy_loading_section"))
    return


@app.cell
def _(selected):
    before_count = selected.get_metadata()["loaded_count"]
    selected_frame = selected[0]
    assert selected_frame is not None
    after_item_count = selected.get_metadata()["loaded_count"]
    sample_values = selected_frame.data
    after_data_count = selected.get_metadata()["loaded_count"]
    assert before_count == 0
    assert after_item_count == 1
    assert after_data_count == after_item_count
    sample_preview = sample_values[:5].tolist()
    return after_data_count, after_item_count, before_count, sample_preview


@app.cell(hide_code=True)
def _(after_data_count, after_item_count, before_count, mo, sample_preview, t):
    mo.md(
        t(
            "lazy_boundary_result",
            before=before_count,
            after_item=after_item_count,
            after_data=after_data_count,
            samples=sample_preview,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("dataset_chaining_section"))
    return


@app.cell
def _(dataset):
    processed_dataset = dataset.normalize().stft(n_fft=128)
    processed_selected = processed_dataset.select(partition_0="group_a", partition_1="batch_01")
    assert len(processed_selected) == 1
    processed_frame = processed_selected[0]
    assert processed_frame is not None
    processed_values = processed_frame.data
    assert processed_values.size > 0
    assert processed_frame.metadata["partition_0"] == "group_a"
    assert processed_frame.metadata["partition_1"] == "batch_01"
    return processed_frame, processed_selected


@app.cell(hide_code=True)
def _(mo, processed_frame, processed_selected, t):
    mo.md(
        t(
            "transform_result",
            count=len(processed_selected),
            metadata=processed_frame.metadata["partition_0"],
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("csv_section"))
    return


@app.cell
def _(pd, root):
    recordings = pd.read_csv(root / "recordings.csv")
    return (recordings,)


@app.cell(hide_code=True)
def _(mo, recordings, t):
    mo.vstack([mo.md(t("csv_table")), recordings])
    return


@app.cell
def _(recordings):
    lookup = recordings.set_index("path")[["condition", "priority"]].to_dict(orient="index")
    return (lookup,)


@app.cell
def _(lookup, root, wd):
    csv_dataset = wd.from_folder(
        str(root),
        recursive=True,
        file_extensions=[".wav"],
        metadata_resolver=lambda path: lookup[path.as_posix()],
    )
    reference_files = csv_dataset.select(condition="reference", priority=1)
    assert len(reference_files) == 1
    return (reference_files,)


@app.cell(hide_code=True)
def _(mo, reference_files, t):
    mo.md(t("csv_result", count=len(reference_files)))
    return


@app.cell(hide_code=True)
def _(docs_relative_href, locale, mo, t):
    api_link = f"[Frame Dataset utility reference]({docs_relative_href(locale, 'api/utils/')})"
    mo.md(t("summary", api_link=api_link))
    return


@app.cell(hide_code=True)
def _(locale, mo, navigation_markdown):
    mo.md(navigation_markdown("08_metadata_driven_dataset_search", locale))
    return


if __name__ == "__main__":
    app.run()
