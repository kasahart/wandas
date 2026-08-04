import marimo

__generated_with = "0.23.1"
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
    catalog = load_catalog("01_getting_started", locale)

    def t(key, **values):
        return catalog.text(key, **values)

    return language_switch_markdown, locale, mo, navigation_markdown, t


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(f"# {t('title')}\n\n{t('intro')}")
    return


@app.cell(hide_code=True)
def _(language_switch_markdown, locale, mo):
    mo.md(language_switch_markdown("01_getting_started", locale))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("why_environment"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("installation_pypi"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("installation_dev"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("installation_visualization"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("imports"))
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    import wandas as wd

    return np, plt, wd


@app.cell(hide_code=True)
def _(mo, np, plt, t, wd):
    mo.md(
        t(
            "library_versions",
            wandas=wd.__version__,
            numpy=np.__version__,
            matplotlib=plt.matplotlib.__version__,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("plot_config"))
    return


@app.cell
def _(plt):
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("marimo_check"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    from importlib.metadata import version as _metadata_version

    mo.md(t("marimo_available", version=_metadata_version("marimo")))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("signal_generation"))
    return


@app.cell
def _(wd):
    simple_tone = wd.generate_sin()
    return (simple_tone,)


@app.cell(hide_code=True)
def _(mo, simple_tone, t):
    mo.md(
        t(
            "signal_info",
            channels=simple_tone.n_channels,
            sampling_rate=simple_tone.sampling_rate,
            duration=f"{simple_tone.duration:.1f}",
            samples=simple_tone.n_samples,
            labels=simple_tone.labels,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("visualization"))
    return


@app.cell
def _(simple_tone):
    simple_tone.describe(is_close=False)
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("describe_result"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("method_chain"))
    return


@app.cell
def _(wd):
    complex_signal = wd.generate_sin(freqs=[440, 880, 1320], duration=2.0, sampling_rate=8000).sum()
    processed = complex_signal.fade(fade_ms=10).low_pass_filter(cutoff=1000)
    processed.print_operation_history()

    config = {
        "fmin": 100,
        "fmax": 3000,
        "cmap": "jet",
        "vmin": -80,
        "vmax": -20,
        "waveform": {"ylim": (-3, 3)},
        "spectral": {"xlim": (-60, 0)},
    }
    combined_signal = complex_signal.concat_frame(processed, label_prefix="processed")
    combined_signal.describe(**config)
    return (combined_signal,)


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("spectrum_context"))
    return


@app.cell
def _(combined_signal):
    combined_signal.fft().plot(overlay=True, title="Original and filtered signal spectrum")
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("interactive_experiment"))
    return


@app.cell
def _(wd):
    def experiment_with_signal(frequency=440, duration=1.0, filter_cutoff=500):
        signal = wd.generate_sin(
            freqs=[frequency, frequency * 2],
            duration=duration,
            sampling_rate=4000,
        ).sum()
        filtered = signal.low_pass_filter(cutoff=filter_cutoff)
        return signal.concat_frame(filtered, label_prefix="filtered")

    experiment_with_signal().fft().plot(overlay=True, title="Original and filtered signal spectrum")
    return (experiment_with_signal,)


@app.cell
def _(experiment_with_signal):
    experiment_with_signal(frequency=880, filter_cutoff=1500).fft().plot(
        overlay=True, title="Original and filtered signal spectrum"
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("troubleshooting"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    from importlib.metadata import version as _metadata_version

    import matplotlib

    mo.md(
        t(
            "troubleshooting_output",
            backend=matplotlib.get_backend(),
            version=_metadata_version("marimo"),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("s3"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("summary"))
    return


@app.cell(hide_code=True)
def _(locale, mo, navigation_markdown):
    mo.md(navigation_markdown("01_getting_started", locale))
    return


if __name__ == "__main__":
    app.run()
