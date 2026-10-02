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

    return catalog, docs_relative_href, language_switch_markdown, locale, mo, navigation_markdown, t


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
    relative_paths = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.wav"))
    file_count = len(relative_paths)
    assert file_count == 3
    return file_count, relative_paths, root


@app.cell(hide_code=True)
def _(file_count, mo, relative_paths, root, t):
    paths = "\n".join(f"- `{path}`" for path in relative_paths)
    mo.md(t("discovery_result", root=root, count=file_count, paths=paths))
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
    metadata = {key: processed_frame.metadata[key] for key in ("partition_0", "partition_1")}
    mo.md(
        t(
            "transform_result",
            count=len(processed_selected),
            metadata=metadata,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("file_list_section"))
    return


@app.cell
def _(relative_paths, root, wd):
    chosen_paths = [relative_paths[2], relative_paths[0], relative_paths[2]]
    listed_dataset = wd.from_files(chosen_paths, base_dir=root)
    listed_loaded = listed_dataset.get_metadata()["loaded_count"]
    assert len(listed_dataset) == len(chosen_paths) == 3
    assert listed_loaded == 0
    return chosen_paths, listed_dataset, listed_loaded


@app.cell(hide_code=True)
def _(chosen_paths, listed_dataset, listed_loaded, mo, t):
    mo.md(t("file_list_result", paths=chosen_paths, count=len(listed_dataset), loaded=listed_loaded))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("catalog_section"))
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
def _(root, wd):
    catalog_dataset = wd.from_table(root / "recordings.csv", path_column="path")
    catalog_reference = catalog_dataset.select(condition="reference", priority="1")
    catalog_wrong_type = catalog_dataset.select(priority=1)
    assert len(catalog_reference) == 1
    assert len(catalog_wrong_type) == 0
    assert catalog_dataset.get_metadata()["loaded_count"] == 0
    return catalog_dataset, catalog_reference, catalog_wrong_type


@app.cell
def _(recordings, root, wd):
    table_dataset = wd.from_table(recordings, path_column="path", base_dir=root)
    table_reference = table_dataset.select(condition="reference", priority=1)
    assert len(table_reference) == 1
    assert table_dataset.get_metadata()["loaded_count"] == 0
    return table_dataset, table_reference


@app.cell(hide_code=True)
def _(catalog_reference, catalog_wrong_type, mo, t, table_reference):
    mo.md(
        t(
            "catalog_result",
            csv_count=len(catalog_reference),
            wrong_count=len(catalog_wrong_type),
            dataframe_count=len(table_reference),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("observations_section"))
    return


@app.cell
def _(pd, relative_paths):
    observations = pd.DataFrame(
        {
            "path": [relative_paths[0], relative_paths[0], relative_paths[1]],
            "observation": ["followup", "reference", "missing_note"],
            "priority": [2, 1, 1],
            "note": [pd.NA, "reviewed", pd.NA],
        }
    )
    return (observations,)


@app.cell(hide_code=True)
def _(mo, observations, t):
    mo.vstack([mo.md(t("observations_table")), observations])
    return


@app.cell
def _(observations, root, wd):
    observation_dataset = wd.from_table(observations, path_column="path", base_dir=root)
    missing_note = observation_dataset.select(priority=1, note=None)
    assert len(observation_dataset) == 3
    assert len(missing_note) == 1
    assert observation_dataset.get_metadata()["loaded_count"] == 0
    observation_before = missing_note.get_metadata()["loaded_count"]
    return missing_note, observation_before, observation_dataset


@app.cell
def _(missing_note):
    observation_frame = missing_note[0]
    assert observation_frame is not None
    observation_after_item = missing_note.get_metadata()["loaded_count"]
    assert observation_frame.metadata["note"] is None
    assert observation_frame.metadata["observation"] == "missing_note"
    observation_samples = observation_frame.data
    assert observation_samples.size > 0
    observation_after_data = missing_note.get_metadata()["loaded_count"]
    assert observation_after_item == observation_after_data == 1
    return observation_after_data, observation_after_item, observation_samples


@app.cell(hide_code=True)
def _(mo, observation_after_data, observation_after_item, observation_before, observation_samples, t):
    mo.md(
        t(
            "observations_result",
            before=observation_before,
            after_item=observation_after_item,
            after_data=observation_after_data,
            shape=observation_samples.shape,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("csv_section"))
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
def _(catalog, docs_relative_href, locale, mo, t):
    suffix = f" ({catalog.text('navigation.japanese_only')})" if locale == "en" else ""
    api_link = f"[Frame Dataset utility reference{suffix}]({docs_relative_href(locale, 'api/utils/')})"
    mo.md(t("summary", api_link=api_link))
    return


@app.cell(hide_code=True)
def _(locale, mo, navigation_markdown):
    mo.md(navigation_markdown("08_metadata_driven_dataset_search", locale))
    return


if __name__ == "__main__":
    app.run()
