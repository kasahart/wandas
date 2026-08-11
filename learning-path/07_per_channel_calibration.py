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
    catalog = load_catalog("07_per_channel_calibration", locale)

    def t(key, **values):
        return catalog.text(key, **values)

    return catalog, docs_relative_href, language_switch_markdown, locale, mo, navigation_markdown, t


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(f"# {t('title')}\n\n{t('intro')}")
    return


@app.cell(hide_code=True)
def _(language_switch_markdown, locale, mo):
    mo.md(language_switch_markdown("07_per_channel_calibration", locale))
    return


@app.cell
def _():
    import io
    import pathlib
    import tempfile

    import numpy as np
    import pandas as pd
    import soundfile as sf

    import wandas as wd
    from wandas import pipeline as pipeline_api

    return io, np, pathlib, pd, pipeline_api, sf, tempfile, wd


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("reference_events"))
    return


@app.cell
def _(np, pathlib, sf, tempfile, wd):
    calibration_directory = tempfile.TemporaryDirectory()
    calibration_root = pathlib.Path(calibration_directory.name)
    microphone_path = calibration_root / "microphone-reference.wav"
    acceleration_path = calibration_root / "acceleration-reference.wav"
    measurement_path = calibration_root / "measurement.wav"
    sf.write(microphone_path, np.array([0.5, -0.5]), 8_000, subtype="DOUBLE")
    sf.write(acceleration_path, np.array([0.25, -0.25]), 8_000, subtype="DOUBLE")
    sf.write(measurement_path, np.array([[1.0, 2.0], [-1.0, -2.0]]), 8_000, subtype="DOUBLE")

    microphone_reference = wd.read(microphone_path, ch_labels=["microphone"])
    acceleration_reference = wd.read(acceleration_path, ch_labels=["accelerometer"])
    multichannel_measurement = wd.read(
        measurement_path,
        ch_labels=["microphone", "accelerometer"],
    )
    derived_by_label = {
        **microphone_reference.derive_calibration(target_level=94.0, unit="Pa"),
        **acceleration_reference.derive_calibration(target_rms=1.0, unit="m/s^2"),
    }
    derived_measurement = multichannel_measurement.with_calibration(derived_by_label)
    assert derived_measurement.channels[0].ref == 2e-5
    return calibration_directory, derived_measurement


@app.cell(hide_code=True)
def _(derived_measurement, mo, pd, t):
    derived_summary = pd.DataFrame(
        {
            t("table.channel"): derived_measurement.labels,
            t("table.factor"): [channel.calibration.factor for channel in derived_measurement.channels],
            t("table.unit"): [channel.unit for channel in derived_measurement.channels],
            t("table.physical_first_sample"): derived_measurement.data[:, 0],
        }
    )
    mo.vstack([mo.md(t("derived_result")), derived_summary])
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("physical_relationship"))
    return


@app.cell
def _(np, wd):
    recorded_signal = wd.from_numpy(
        np.array(
            [
                [10.0, 20.0, 30.0, 40.0],
                [0.1, 0.2, 0.3, 0.4],
            ]
        ),
        sampling_rate=8_000,
        ch_labels=["microphone", "accelerometer"],
    )
    return (recorded_signal,)


@app.cell(hide_code=True)
def _(mo, pd, recorded_signal, t):
    recorded_preview = pd.DataFrame(
        recorded_signal.data.T,
        columns=recorded_signal.labels,
    )
    mo.vstack([mo.md(t("recorded_result")), recorded_preview])
    return


@app.cell
def _(np, recorded_signal, wd):
    configured_signal = recorded_signal.with_calibration(
        [
            wd.ChannelCalibration(factor=0.02, unit="Pa"),
            wd.ChannelCalibration(factor=9.81, unit="m/s^2", ref=1.0),
        ]
    )

    _recorded_values = recorded_signal.data
    physical_values = configured_signal.data
    np.testing.assert_allclose(
        physical_values,
        _recorded_values * np.array([[0.02], [9.81]]),
    )
    assert configured_signal.channels[0].ref == 2e-5

    alternate_reference_signal = recorded_signal.with_calibration(
        [
            wd.ChannelCalibration(factor=0.02, unit="Pa", ref=1.0),
            wd.ChannelCalibration(factor=9.81, unit="m/s^2", ref=1.0),
        ]
    )
    np.testing.assert_allclose(alternate_reference_signal.data, physical_values)
    assert alternate_reference_signal.channels[0].ref == 1.0
    return (configured_signal,)


@app.cell(hide_code=True)
def _(configured_signal, mo, pd, recorded_signal, t):
    calibration_summary = pd.DataFrame(
        {
            t("table.channel"): configured_signal.labels,
            t("table.recorded_first_sample"): recorded_signal.data[:, 0],
            t("table.factor"): [channel.calibration.factor for channel in configured_signal.channels],
            t("table.physical_first_sample"): configured_signal.data[:, 0],
            t("table.unit"): [channel.unit for channel in configured_signal.channels],
            t("table.ref"): [channel.ref for channel in configured_signal.channels],
        }
    )
    mo.vstack([mo.md(t("configured_result")), calibration_summary])
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("calibration_input_contract"))
    return


@app.cell
def _(configured_signal):
    reissued_signal = configured_signal.with_calibration([0.025, 9.75])
    assert configured_signal.channels[0].calibration.factor == 0.02
    assert reissued_signal.channels[0].unit == "Pa"
    assert reissued_signal.channels[1].ref == 1.0
    return (reissued_signal,)


@app.cell(hide_code=True)
def _(configured_signal, mo, pd, reissued_signal, t):
    replacement_summary = pd.DataFrame(
        {
            t("table.channel"): reissued_signal.labels,
            t("table.before_factor"): [channel.calibration.factor for channel in configured_signal.channels],
            t("table.after_factor"): [channel.calibration.factor for channel in reissued_signal.channels],
            t("table.unit_preserved"): [channel.unit for channel in reissued_signal.channels],
            t("table.ref_preserved"): [channel.ref for channel in reissued_signal.channels],
        }
    )
    mo.vstack([mo.md(t("replacement_result")), replacement_summary])
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("csv_section"))
    return


@app.cell
def _(io, pd):
    calibration_table = pd.read_csv(
        io.StringIO("channel,factor,unit,ref\naccelerometer,9.79,m/s^2,1.0\nmicrophone,0.021,Pa,0.00002\n")
    )
    return (calibration_table,)


@app.cell(hide_code=True)
def _(calibration_table, mo, t):
    display_table = calibration_table.rename(
        columns={
            "channel": t("table.channel"),
            "factor": t("table.factor"),
            "unit": t("table.unit"),
            "ref": t("table.ref"),
        }
    )
    mo.vstack([mo.md(t("csv_table")), display_table])
    return


@app.cell
def _(calibration_table, recorded_signal, wd):
    calibration_by_label = {
        str(row.channel): wd.ChannelCalibration(
            factor=float(row.factor),
            unit=str(row.unit),
            ref=float(row.ref),
        )
        for row in calibration_table.itertuples(index=False)
    }
    csv_configured_signal = recorded_signal.with_calibration(calibration_by_label)
    csv_result = {
        "labels": csv_configured_signal.labels,
        "factors": [channel.calibration.factor for channel in csv_configured_signal.channels],
    }
    assert csv_result["factors"] == [0.021, 9.79]
    return (csv_configured_signal,)


@app.cell(hide_code=True)
def _(csv_configured_signal, mo, pd, t):
    csv_summary = pd.DataFrame(
        {
            t("table.frame_order"): csv_configured_signal.labels,
            t("table.factor"): [channel.calibration.factor for channel in csv_configured_signal.channels],
            t("table.unit"): [channel.unit for channel in csv_configured_signal.channels],
        }
    )
    mo.vstack([mo.md(t("csv_result")), csv_summary])
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("array_section"))
    return


@app.cell
def _(calibration_table, configured_signal):
    ordered_factors = calibration_table.set_index("channel").loc[configured_signal.labels, "factor"].to_numpy()
    array_configured_signal = configured_signal.with_calibration(ordered_factors)
    assert [channel.calibration.factor for channel in array_configured_signal.channels] == [0.021, 9.79]
    return (array_configured_signal,)


@app.cell(hide_code=True)
def _(array_configured_signal, mo, pd, t):
    array_summary = pd.DataFrame(
        {
            t("table.channel"): array_configured_signal.labels,
            t("table.factor_from_array"): [channel.calibration.factor for channel in array_configured_signal.channels],
            t("table.unit_preserved"): [channel.unit for channel in array_configured_signal.channels],
        }
    )
    mo.vstack([mo.md(t("array_result")), array_summary])
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("hundred_channel_section"))
    return


@app.cell
def _(np, wd):
    channel_count = 100
    labels = [f"sensor-{index:03d}" for index in range(channel_count)]
    hundred_recorded = wd.from_numpy(
        np.ones((channel_count, 16)),
        sampling_rate=8_000,
        ch_labels=labels,
    )
    all_factors = 1.0 + np.arange(channel_count) / 1_000
    all_calibrations = [wd.ChannelCalibration(factor=float(factor), unit="m/s^2", ref=1.0) for factor in all_factors]
    configured_hundred = hundred_recorded.with_calibration(all_calibrations)
    partially_updated = configured_hundred.with_calibration(
        {f"sensor-{index:03d}": 2.0 for index in range(0, channel_count, 10)}
    )

    hundred_values = partially_updated.data
    assert partially_updated.n_channels == 100
    assert hundred_values.shape == (100, 16)
    assert partially_updated.channels[0].calibration.factor == 2.0
    assert partially_updated.channels[1].calibration.factor == 1.001
    assert partially_updated.channels[10].calibration.factor == 2.0
    return (partially_updated,)


@app.cell(hide_code=True)
def _(mo, np, partially_updated, pd, t):
    inspect_indices = [0, 1, 10, 90, 91]
    hundred_summary = pd.DataFrame(
        {
            t("table.index"): inspect_indices,
            t("table.channel"): [partially_updated.labels[index] for index in inspect_indices],
            t("table.factor"): [partially_updated.channels[index].calibration.factor for index in inspect_indices],
            t("table.physical_first_sample"): partially_updated.data[inspect_indices, 0],
            t("table.unit"): [partially_updated.channels[index].unit for index in inspect_indices],
        }
    )
    mo.vstack([mo.md(t("hundred_channel_result")), hundred_summary])
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("data_section"))
    return


@app.cell
def _(np, recorded_signal, wd):
    raw_values = recorded_signal.data
    calibrated_values = recorded_signal.with_calibration([0.02, 9.81]).data
    np.testing.assert_array_equal(raw_values[0], [10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(calibrated_values[0], [0.2, 0.4, 0.6, 0.8])

    calibrated_signal = recorded_signal.with_calibration(
        [wd.ChannelCalibration(factor=0.02, unit="Pa"), wd.ChannelCalibration(factor=9.81, unit="m/s^2", ref=1.0)]
    )
    spectrum = calibrated_signal.fft(n_fft=4, window="boxcar")
    expected_spectrum = wd.from_numpy(
        calibrated_values,
        sampling_rate=calibrated_signal.sampling_rate,
        ch_labels=calibrated_signal.labels,
        ch_units=[channel.unit for channel in calibrated_signal.channels],
    ).fft(n_fft=4, window="boxcar")
    np.testing.assert_allclose(spectrum.data, expected_spectrum.data)
    return calibrated_signal, raw_values, calibrated_values, spectrum


@app.cell(hide_code=True)
def _(calibrated_values, mo, np, pd, raw_values, t):
    boundary_summary = pd.DataFrame(
        {
            t("table.check"): [
                t("table.raw_first_sample"),
                t("table.calibrated_first_sample"),
                t("table.fft_uses_calibrated_values"),
            ],
            t("table.result"): [raw_values[0, 0], calibrated_values[0, 0], t("result.yes")],
        }
    )
    mo.vstack([mo.md(t("data_result")), boundary_summary])
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("level_section"))
    return


@app.cell
def _(calibrated_signal, np, spectrum):
    linear_rms = calibrated_signal.rms
    rms_levels = calibrated_signal.rms_level
    level_references = [channel.level_reference for channel in calibrated_signal.channels]

    np.testing.assert_allclose(
        spectrum.dB[0],
        level_references[0].to_level(spectrum.magnitude[0]),
    )
    return level_references, linear_rms, rms_levels


@app.cell(hide_code=True)
def _(calibrated_signal, level_references, linear_rms, mo, pd, rms_levels, t):
    level_summary = pd.DataFrame(
        {
            t("table.channel"): calibrated_signal.labels,
            t("table.linear_rms"): linear_rms,
            t("table.rms_level"): rms_levels,
            t("table.level_reference"): [reference.label for reference in level_references],
        }
    )
    mo.vstack([mo.md(t("level_result")), level_summary])
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("recipe_section"))
    return


@app.cell
def _(np, pipeline_api, recorded_signal, wd):
    workflow = recorded_signal.get_channel([1, 0]).with_calibration(
        [
            wd.ChannelCalibration(9.81, "m/s^2", 1.0),
            wd.ChannelCalibration(0.02, "Pa"),
        ]
    )
    plan = pipeline_api.RecipePlan.from_frame(workflow, input_names=("signal",))
    replayed = plan.apply({"signal": recorded_signal})
    np.testing.assert_allclose(replayed.data, workflow.data)
    return (replayed,)


@app.cell(hide_code=True)
def _(mo, pd, replayed, t):
    recipe_summary = pd.DataFrame(
        {
            t("table.channel"): replayed.labels,
            t("table.factor"): [channel.calibration.factor for channel in replayed.channels],
            t("table.unit"): [channel.unit for channel in replayed.channels],
        }
    )
    mo.vstack([mo.md(t("recipe_result")), recipe_summary])
    return


@app.cell
def _(configured_signal, np, pathlib, tempfile, wd):
    with tempfile.TemporaryDirectory() as temporary_directory:
        wdf_path = pathlib.Path(temporary_directory) / "calibrated.wdf"
        configured_signal.save(wdf_path)
        loaded_signal = wd.load(wdf_path)
        loaded_physical = np.array(loaded_signal.data, copy=True)
        loaded_factors = [channel.calibration.factor for channel in loaded_signal.channels]
        del loaded_signal

    np.testing.assert_allclose(loaded_physical, configured_signal.data)
    return loaded_factors, loaded_physical


@app.cell(hide_code=True)
def _(configured_signal, loaded_factors, loaded_physical, mo, pd, t):
    wdf_summary = pd.DataFrame(
        {
            t("table.preserved"): [t("table.frame_data"), t("table.calibration_factors")],
            t("table.result"): [t("result.yes"), str(loaded_factors)],
        }
    )
    assert loaded_physical.shape == configured_signal.data.shape
    mo.vstack([mo.md(t("wdf_result")), wdf_summary])
    return


@app.cell(hide_code=True)
def _(catalog, docs_relative_href, locale, mo, t):
    suffix = f" ({catalog.text('navigation.japanese_only')})" if locale == "en" else ""
    frames_link = f"[Frames API reference{suffix}]({docs_relative_href(locale, 'api/frames/')})"
    wdf_link = f"[WDF File I/O{suffix}]({docs_relative_href(locale, 'api/wdf_io/')})"
    mo.md(t("summary", frames_link=frames_link, wdf_link=wdf_link))
    return


@app.cell(hide_code=True)
def _(locale, mo, navigation_markdown):
    mo.md(navigation_markdown("07_per_channel_calibration", locale))
    return


if __name__ == "__main__":
    app.run()
