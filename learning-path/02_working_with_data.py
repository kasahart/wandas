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
    catalog = load_catalog("02_working_with_data", locale)

    def t(key, **values):
        return catalog.text(key, **values)

    return language_switch_markdown, locale, mo, navigation_markdown, t


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(f"# {t('title')}\n\n{t('intro')}")
    return


@app.cell(hide_code=True)
def _(language_switch_markdown, locale, mo):
    mo.md(language_switch_markdown("02_working_with_data", locale))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("why_loading"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("supported_inputs"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("wav_section"))
    return


@app.cell
def _():
    import re
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    import wandas as wd

    pathlib_path = Path
    plt.rcParams["figure.figsize"] = (12, 6)
    return np, pathlib_path, plt, re, sf, wd


@app.cell
def _(pathlib_path):
    wav_path = pathlib_path(__file__).resolve().parent / "sample_audio.wav"
    csv_path = pathlib_path(__file__).resolve().parent / "sensor_data.csv"
    if not wav_path.is_file():
        raise FileNotFoundError(f"Checked-in WAV fixture not found: {wav_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Checked-in CSV fixture not found: {csv_path}")
    return csv_path, wav_path


@app.cell(hide_code=True)
def _(locale, pathlib_path):
    output_dir = pathlib_path("output") / locale
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir,)


@app.cell
def _(wav_path, wd):
    audio = wd.read(wav_path)
    return (audio,)


@app.cell(hide_code=True)
def _(audio, mo, t, wav_path):
    mo.md(
        t(
            "wav_info",
            file=wav_path.name,
            channels=audio.n_channels,
            sampling_rate=audio.sampling_rate,
            duration=f"{audio.duration:.2f}",
            samples=audio.n_samples,
            dtype=audio.data.dtype,
            shape=audio.shape,
            labels=audio.labels,
            metadata_keys=list(audio.metadata),
        )
    )
    return


@app.cell
def _(audio):
    audio.describe()
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("csv_section"))
    return


@app.cell
def _(csv_path, wd):
    sensor_data = wd.read(csv_path, time_column="time", delimiter=",")
    return (sensor_data,)


@app.cell(hide_code=True)
def _(csv_path, mo, sensor_data, t):
    mo.md(
        t(
            "csv_info",
            file=csv_path.name,
            channels=sensor_data.n_channels,
            sampling_rate=f"{sensor_data.sampling_rate:.1f}",
            duration=f"{sensor_data.duration:.1f}",
            labels=sensor_data.labels,
            shape=sensor_data.shape,
        )
    )
    return


@app.cell
def _(sensor_data):
    sensor_data.describe()
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("numpy_section"))
    return


@app.cell
def _(np, wd):
    np.random.seed(123)
    sampling_rate = 1000
    duration = 2.0
    n_samples = int(duration * sampling_rate)
    time = np.arange(n_samples) / sampling_rate
    left_channel = np.sin(2 * np.pi * 440 * time) + 0.1 * np.random.randn(n_samples)
    right_channel = np.sin(2 * np.pi * 440 * time + np.pi / 4) + 0.1 * np.random.randn(n_samples)
    stereo_data = np.vstack([left_channel, right_channel])
    stereo_audio = wd.from_numpy(data=stereo_data, sampling_rate=sampling_rate, ch_labels=["Left", "Right"])
    return sampling_rate, stereo_audio, stereo_data


@app.cell(hide_code=True)
def _(mo, stereo_audio, stereo_data, t):
    mo.md(
        t(
            "numpy_info",
            shape=stereo_data.shape,
            sampling_rate=stereo_audio.sampling_rate,
            labels=stereo_audio.labels,
            dtype=stereo_audio.data.dtype,
        )
    )
    return


@app.cell
def _(stereo_audio):
    stereo_audio.plot(title="Stereo Audio from NumPy Array", overlay=False)
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("channel_frame"))
    return


@app.cell(hide_code=True)
def _(mo, sensor_data, t):
    mo.md(
        t(
            "frame_info",
            frame_type=type(sensor_data).__name__,
            shape=sensor_data.shape,
            dtype=sensor_data.data.dtype,
            sampling_rate=sensor_data.sampling_rate,
            channels=sensor_data.n_channels,
            samples=sensor_data.n_samples,
            duration=f"{sensor_data.duration:.2f}",
            labels=sensor_data.labels,
            metadata_keys=list(sensor_data.metadata),
            history_count=len(sensor_data.operation_history),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("channel_access"))
    return


@app.cell
def _(sensor_data):
    first_channel = sensor_data[0]
    accel_x = sensor_data["accel_x"]
    accel_channels = sensor_data[["accel_x", "accel_y", "accel_z"]]
    first_two = sensor_data[0:2]
    return accel_channels, accel_x, first_channel, first_two


@app.cell(hide_code=True)
def _(accel_channels, accel_x, first_channel, first_two, mo, t):
    mo.md(
        t(
            "channel_access_result",
            first_shape=first_channel.shape,
            first_labels=first_channel.labels,
            named_shape=accel_x.shape,
            named_labels=accel_x.labels,
            group_shape=accel_channels.shape,
            group_labels=accel_channels.labels,
            slice_shape=first_two.shape,
            slice_labels=first_two.labels,
        )
    )
    return


@app.cell
def _(np, sensor_data):
    rms_values = sensor_data.rms
    active_channels = sensor_data[rms_values > 0.5]
    time_slice = sensor_data[:, 100:200]
    magnitude = np.sqrt(sensor_data["accel_x"] ** 2 + sensor_data["accel_y"] ** 2 + sensor_data["accel_z"] ** 2)
    return active_channels, magnitude, rms_values, time_slice


@app.cell(hide_code=True)
def _(active_channels, magnitude, mo, rms_values, sensor_data, t, time_slice):
    mo.md(
        t(
            "channel_operation_result",
            rms_values=dict(zip(sensor_data.labels, rms_values)),
            active_labels=active_channels.labels,
            time_shape=time_slice.shape,
            magnitude_shape=magnitude.shape,
        )
    )
    return


@app.cell
def _(active_channels, plt, sensor_data, time_slice):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sensor_data.plot(ax=axes[0, 0], title="All Channels", overlay=True)
    sensor_data[["accel_x", "accel_y", "accel_z"]].plot(ax=axes[0, 1], title="Acceleration Only", overlay=False)
    active_channels.plot(ax=axes[1, 0], title="Active Channels Only", overlay=True)
    time_slice.plot(ax=axes[1, 1], title="Time Slice (samples 100:200)", overlay=True)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("query_section"))
    return


@app.cell
def _(re, sensor_data):
    selected_exact = sensor_data.get_channel(query="accel_x")
    selected_partial = sensor_data.get_channel(query=re.compile(r".*ccel.*"))

    def high_energy_predicate(channel_metadata):
        channel_index = sensor_data.labels.index(channel_metadata.label)
        return sensor_data.rms[channel_index] > 0.5

    selected_predicate = sensor_data.get_channel(query=high_energy_predicate)
    selected_dict = sensor_data.get_channel(query={"label": "temperature"})
    selected_dict_regex = sensor_data.get_channel(query={"label": re.compile(r"accel_.*")})
    return selected_dict, selected_dict_regex, selected_exact, selected_partial, selected_predicate


@app.cell(hide_code=True)
def _(mo, selected_dict, selected_dict_regex, selected_exact, selected_partial, selected_predicate, t):
    mo.md(
        t(
            "query_result",
            exact=selected_exact.labels,
            partial=selected_partial.labels,
            predicate=selected_predicate.labels,
            dictionary=selected_dict.labels,
            dictionary_regex=selected_dict_regex.labels,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("query_notes"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("selection_notes"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("saving"))
    return


@app.cell
def _(audio, np, output_dir, sensor_data, sf, stereo_audio, wd):
    wav_output = output_dir / "processed_audio.wav"
    audio.to_wav(wav_output)
    round_trip_audio = wd.read(wav_output)
    wav_subtype = sf.info(wav_output).subtype

    trimmed_sensor = sensor_data.trim(start=0, end=1)
    wdf_output = output_dir / "sensor_data.wdf"
    trimmed_sensor.save(wdf_output, overwrite=True)

    np_output = output_dir / "stereo_audio.npy"
    np.save(np_output, stereo_audio.data)

    csv_output = output_dir / "processed_sensors.csv"
    sensor_data.to_dataframe().to_csv(csv_output)
    return csv_output, np_output, round_trip_audio, trimmed_sensor, wav_output, wav_subtype, wdf_output


@app.cell(hide_code=True)
def _(audio, mo, np, round_trip_audio, t, wav_output, wav_subtype):
    np.testing.assert_array_equal(round_trip_audio.data, audio.data)
    mo.md(
        t(
            "wav_save_result",
            path=wav_output,
            subtype=wav_subtype,
            dtype=round_trip_audio.data.dtype,
            shape=round_trip_audio.shape,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t, trimmed_sensor, wdf_output):
    mo.md(t("wdf_save_result", path=wdf_output, shape=trimmed_sensor.shape))
    return


@app.cell(hide_code=True)
def _(mo, np_output, stereo_audio, t):
    mo.md(t("numpy_save_result", path=np_output, shape=stereo_audio.shape))
    return


@app.cell(hide_code=True)
def _(csv_output, mo, sensor_data, t):
    mo.md(t("csv_save_result", path=csv_output, labels=sensor_data.labels))
    return


@app.cell
def _(csv_output, np, np_output, stereo_audio, wd, wdf_output):
    loaded_wdf = wd.load(wdf_output)
    loaded_np = wd.from_numpy(
        data=np.load(np_output),
        sampling_rate=stereo_audio.sampling_rate,
        ch_labels=stereo_audio.labels,
    )
    loaded_csv = wd.read(csv_output, time_column="time")
    return loaded_csv, loaded_np, loaded_wdf


@app.cell(hide_code=True)
def _(loaded_csv, loaded_np, loaded_wdf, mo, t):
    mo.md(
        t(
            "reload_result",
            wdf_shape=loaded_wdf.shape,
            wdf_metadata_keys=list(loaded_wdf.metadata),
            numpy_shape=loaded_np.shape,
            numpy_sampling_rate=loaded_np.sampling_rate,
            csv_shape=loaded_csv.shape,
            csv_labels=loaded_csv.labels,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("workflow"))
    return


@app.cell
def _(csv_path, output_dir, wd):
    data = wd.read(csv_path, time_column="time")
    processed = data.high_pass_filter(cutoff=0.5).low_pass_filter(cutoff=10).normalize()
    features = {
        "rms": processed.rms,
        "peak": processed.abs().data.max(),
        "crest_factor": processed.abs().data.max(-1) / processed.rms,
    }
    processed.describe()
    final_output = output_dir / "analyzed_sensor_data.wdf"
    processed.save(final_output, overwrite=True)
    return data, features, final_output, processed


@app.cell(hide_code=True)
def _(data, features, final_output, mo, processed, t):
    mo.md(
        t(
            "workflow_result",
            input_shape=data.shape,
            operation_count=len(processed.operation_history),
            rms=dict(zip(processed.labels, features["rms"])),
            peak=features["peak"],
            output=final_output,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("troubleshooting"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("summary"))
    return


@app.cell(hide_code=True)
def _(locale, mo, navigation_markdown):
    mo.md(navigation_markdown("02_working_with_data", locale))
    return


if __name__ == "__main__":
    app.run()
