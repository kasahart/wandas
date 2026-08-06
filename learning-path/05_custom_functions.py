import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from scripts.learning_path_i18n import (
        language_switch_markdown,
        load_catalog,
        locale_from_argv,
        navigation_markdown,
    )

    locale = locale_from_argv()
    catalog = load_catalog("05_custom_functions", locale)

    def t(key, **values):
        return catalog.text(key, **values)

    return language_switch_markdown, locale, mo, navigation_markdown, t


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(f"# {t('title')}\n\n{t('intro')}")
    return


@app.cell(hide_code=True)
def _(language_switch_markdown, locale, mo):
    mo.md(language_switch_markdown("05_custom_functions", locale))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("choice_guide"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("input_section"))
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal as scipy_signal

    import wandas as wd

    plt.rcParams["figure.figsize"] = (12, 6)
    return np, plt, scipy_signal, wd


@app.cell
def _(np, wd):
    rng = np.random.default_rng(7)
    sampling_rate = 1000.0
    sample_count = 200
    time_axis = np.arange(sample_count, dtype=np.float64) / sampling_rate
    signal_data = np.vstack(
        [
            np.sin(2 * np.pi * 25 * time_axis) + 0.02 * rng.normal(size=sample_count),
            0.5 * np.cos(2 * np.pi * 40 * time_axis) + 0.02 * rng.normal(size=sample_count),
        ]
    ).astype(np.float32)
    input_frame = wd.from_numpy(
        data=signal_data,
        sampling_rate=sampling_rate,
        metadata={"source": "lesson_05", "unit": "V"},
        ch_labels=["left", "right"],
        ch_units=["V", "V"],
    ).with_source_time_offset(0.25)
    return input_frame, sampling_rate, signal_data, time_axis


@app.cell(hide_code=True)
def _(input_frame, mo, signal_data, t):
    mo.md(
        t(
            "input_result",
            shape=input_frame.shape,
            dtype=signal_data.dtype,
            channels=input_frame.n_channels,
            samples=input_frame.n_samples,
            sampling_rate=input_frame.sampling_rate,
            duration=f"{input_frame.duration:.3f}",
            offset=input_frame.source_time_offset.tolist(),
            labels=input_frame.labels,
            units=[channel.unit for channel in input_frame.channels],
            metadata=input_frame.metadata,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("apply_section"))
    return


@app.cell
def _(input_frame):
    def scale_channels(data, factor):
        return data * factor

    def same_shape(input_shape):
        return input_shape

    scaled_frame = input_frame.apply(scale_channels, output_shape_func=same_shape, factor=1.5)
    return same_shape, scale_channels, scaled_frame


@app.cell
def _(input_frame):
    normalized_frame = input_frame.normalize()
    return (normalized_frame,)


@app.cell(hide_code=True)
def _(input_frame, mo, np, normalized_frame, same_shape, scaled_frame, signal_data, t):
    execution_counter = {"calls": 0}

    def probe(data):
        execution_counter["calls"] += 1
        return data

    lazy_probe_frame = input_frame.apply(probe, output_shape_func=same_shape)
    probe_calls_after_apply = execution_counter["calls"]
    _probe_values = lazy_probe_frame.data
    probe_calls_after_data = execution_counter["calls"]
    assert probe_calls_after_apply == 0
    assert probe_calls_after_data > probe_calls_after_apply

    input_data_before = input_frame.data.copy()
    scaled_values = scaled_frame.data
    normalized_values = normalized_frame.data
    assert scaled_frame.shape == input_frame.shape
    assert scaled_frame.sampling_rate == input_frame.sampling_rate
    assert scaled_frame.previous is input_frame
    assert normalized_frame.previous is input_frame
    assert scaled_frame.lineage is not input_frame.lineage
    assert scaled_frame.operation_history[-1]["operation"] == "wandas.custom.apply"
    assert normalized_frame.operation_history[-1]["operation"] == "wandas.audio.normalize"
    assert scaled_values.dtype == signal_data.dtype
    assert normalized_values.dtype == signal_data.dtype
    assert scaled_frame.metadata == input_frame.metadata
    assert [channel.unit for channel in scaled_frame.channels] == ["V", "V"]
    assert scaled_frame.labels == ["scale_channels(left)", "scale_channels(right)"]
    np.testing.assert_array_equal(input_data_before, signal_data)
    assert input_frame.labels == ["left", "right"]
    assert input_frame.metadata == {"source": "lesson_05", "unit": "V"}
    mo.md(
        t(
            "apply_result",
            materialized_type=type(scaled_values).__name__,
            input_shape=input_frame.shape,
            output_shape=scaled_frame.shape,
            input_dtype=signal_data.dtype,
            output_dtype=scaled_values.dtype,
            lazy_calls_after_apply=probe_calls_after_apply,
            lazy_calls_after_data=probe_calls_after_data,
            sampling_rate=scaled_frame.sampling_rate,
            metadata=scaled_frame.metadata,
            input_labels=input_frame.labels,
            output_labels=scaled_frame.labels,
            channel_units=[channel.unit for channel in scaled_frame.channels],
            custom_operation=scaled_frame.operation_history[-1]["operation"],
            standard_operation=normalized_frame.operation_history[-1]["operation"],
            lineage_status=scaled_frame.lineage is not None,
            previous_status=scaled_frame.previous is input_frame,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("runtime_only_section"))
    return


@app.cell
def _(normalized_frame):
    from wandas.pipeline import RecipePlan

    standard_recipe = RecipePlan.from_frame(normalized_frame, input_names=("signal",))
    standard_recipe_payload = standard_recipe.to_dict()
    return (standard_recipe_payload,)


@app.cell(hide_code=True)
def _(mo, scaled_frame, standard_recipe_payload, t):
    from wandas.pipeline import RecipeExtractionError as _RecipeExtractionError
    from wandas.pipeline import RecipePlan as _RecipePlan

    runtime_recipe_error = ""
    try:
        _RecipePlan.from_frame(scaled_frame)
    except _RecipeExtractionError as error:
        runtime_recipe_error = str(error)
    else:
        raise AssertionError("Frame.apply(callable) must remain runtime-only")

    portable_operation_ids = [node["operation"] for node in standard_recipe_payload["nodes"]]
    assert "Frame.apply(callable) is runtime-only" in runtime_recipe_error
    assert scaled_frame.operation_history[-1]["operation"] == "wandas.custom.apply"
    assert "wandas.audio.normalize" in portable_operation_ids
    mo.md(
        t(
            "recipe_result",
            rejected_operation=scaled_frame.operation_history[-1]["operation"],
            error=runtime_recipe_error,
            portable_operations=portable_operation_ids,
            schema=standard_recipe_payload["schema"],
            version=standard_recipe_payload["version"],
            node_count=len(standard_recipe_payload["nodes"]),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("shape_section"))
    return


@app.cell
def _(input_frame):
    def take_first_half(data):
        return data[:, : data.shape[1] // 2]

    def half_shape(input_shape):
        return (input_shape[0], input_shape[1] // 2)

    first_half_frame = input_frame.apply(take_first_half, output_shape_func=half_shape)
    semantic_trimmed_frame = input_frame.trim(start=0.05, end=0.15)
    return first_half_frame, half_shape, semantic_trimmed_frame, take_first_half


@app.cell(hide_code=True)
def _(first_half_frame, input_frame, mo, np, semantic_trimmed_frame, t):
    assert first_half_frame.shape == (2, 100)
    assert semantic_trimmed_frame.shape == (2, 100)
    assert first_half_frame.sampling_rate == input_frame.sampling_rate
    assert semantic_trimmed_frame.sampling_rate == input_frame.sampling_rate
    assert first_half_frame.duration == 0.1
    assert semantic_trimmed_frame.duration == 0.1
    np.testing.assert_allclose(first_half_frame.source_time_offset, [0.25, 0.25])
    np.testing.assert_allclose(semantic_trimmed_frame.source_time_offset, [0.30, 0.30])
    np.testing.assert_allclose(first_half_frame.source_time[:, 0], [0.25, 0.25])
    np.testing.assert_allclose(semantic_trimmed_frame.source_time[:, 0], [0.30, 0.30])
    assert first_half_frame.metadata == input_frame.metadata
    assert semantic_trimmed_frame.metadata == input_frame.metadata
    assert first_half_frame.previous is input_frame
    assert semantic_trimmed_frame.previous is input_frame
    assert first_half_frame.operation_history[-1]["operation"] == "wandas.custom.apply"
    assert semantic_trimmed_frame.operation_history[-1]["operation"] == "wandas.frame.time_slice"
    mo.md(
        t(
            "shape_result",
            apply_shape=first_half_frame.shape,
            apply_samples=first_half_frame.n_samples,
            apply_sampling_rate=first_half_frame.sampling_rate,
            apply_duration=f"{first_half_frame.duration:.3f}",
            apply_offset=first_half_frame.source_time_offset.tolist(),
            apply_source_time=first_half_frame.source_time[:, 0].tolist(),
            apply_metadata=first_half_frame.metadata,
            trim_shape=semantic_trimmed_frame.shape,
            trim_samples=semantic_trimmed_frame.n_samples,
            trim_sampling_rate=semantic_trimmed_frame.sampling_rate,
            trim_duration=f"{semantic_trimmed_frame.duration:.3f}",
            trim_offset=semantic_trimmed_frame.source_time_offset.tolist(),
            trim_source_time=semantic_trimmed_frame.source_time[:, 0].tolist(),
            trim_metadata=semantic_trimmed_frame.metadata,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("scipy_section"))
    return


@app.cell
def _(input_frame, plt, scipy_signal):
    def median_filter_channel_first(data, kernel_size):
        return scipy_signal.medfilt(data, kernel_size=(1, kernel_size))

    median_frame = input_frame.apply(
        median_filter_channel_first,
        output_shape_func=lambda input_shape: input_shape,
        kernel_size=5,
    )
    (_fig, _axes) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    input_frame.plot(ax=_axes[0], title="Input signal", overlay=True)
    median_frame.plot(ax=_axes[1], title="Median filtered signal", overlay=True)
    _axes[1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()
    return median_filter_channel_first, median_frame


@app.cell(hide_code=True)
def _(input_frame, median_filter_channel_first, median_frame, mo, np, scipy_signal, t):
    input_values = input_frame.data
    filtered_values = median_frame.data
    expected_values = scipy_signal.medfilt(input_values, kernel_size=(1, 5))
    channelwise_expected = np.vstack(
        [median_filter_channel_first(input_values[index : index + 1], 5)[0] for index in range(input_values.shape[0])]
    )
    boundary_probe = np.array([[1, 2, 3, 4, 5]], dtype=input_values.dtype)
    boundary_result = median_filter_channel_first(boundary_probe, 5)
    np.testing.assert_array_equal(filtered_values, expected_values)
    np.testing.assert_array_equal(filtered_values, channelwise_expected)
    assert filtered_values.shape == input_values.shape
    assert filtered_values.dtype == input_values.dtype
    np.testing.assert_array_equal(boundary_result, scipy_signal.medfilt(boundary_probe, kernel_size=(1, 5)))
    mo.md(
        t(
            "scipy_result",
            axis="axis=1",
            kernel_shape=(1, 5),
            input_shape=input_values.shape,
            output_shape=filtered_values.shape,
            input_dtype=input_values.dtype,
            output_dtype=filtered_values.dtype,
            channel_independent=True,
            boundary_result=boundary_result.tolist(),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("extension_section"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("summary"))
    return


@app.cell(hide_code=True)
def _(locale, mo, navigation_markdown):
    mo.md(navigation_markdown("05_custom_functions", locale))
    return


if __name__ == "__main__":
    app.run()
