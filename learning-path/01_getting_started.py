import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    from scripts.learning_path_i18n import load_catalog, locale_from_argv, navigation_markdown

    locale = locale_from_argv()
    catalog = load_catalog("01_getting_started", locale)

    def t(key, **values):
        return catalog.text(key, **values)

    return locale, mo, navigation_markdown, t


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(f"# {t('title')}\n\n{t('intro')}")
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("why_environment"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("installation_pypi"))
    return


@app.cell
def _(t):
    # Show the recommended installation command without executing it.
    # !pip install "wandas[marimo,io,psychoacoustic]"

    # Optional development version:
    # !pip install git+https://github.com/kasahart/wandas.git

    print(t("install_recommended"))
    print('!pip install "wandas[marimo,io,psychoacoustic]"')
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("installation_dev"))
    return


@app.cell
def _(t):
    # Development installation for this repository.
    # !pip install -e .

    print(t("development_install"))
    print("!pip install -e .")
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("installation_visualization"))
    return


@app.cell
def _(t):
    # Dependencies used by the marimo Learning Path apps.
    # !pip install "wandas[marimo,io,psychoacoustic]"

    print(t("visualization_install"))
    print('!pip install "wandas[marimo,io,psychoacoustic]"')
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("imports"))
    return


@app.cell
def _(t):
    # Import the basic libraries.
    import matplotlib.pyplot as plt  # Matplotlib visualization library.
    import numpy as np  # NumPy numerical foundation.

    import wandas as wd  # Wandas signal-processing library.

    print(
        t(
            "library_versions",
            wandas=wd.__version__,
            numpy=np.__version__,
            matplotlib=plt.matplotlib.__version__,
        )
    )

    print(t("libraries_ready"))
    return plt, wd


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("plot_config"))
    return


@app.cell
def _(plt, t):
    # Configure plot display.

    plt.rcParams["figure.figsize"] = (10, 6)  # Readable figure size.
    plt.rcParams["figure.dpi"] = 100  # Dots per inch.

    print(t("plot_configuration_complete"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("marimo_check"))
    return


@app.cell
def _(t):
    # Verify that marimo is available.
    from importlib.metadata import version as _metadata_version

    marimo_version = _metadata_version("marimo")
    print(t("marimo_available", version=marimo_version))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("signal_generation"))
    return


@app.cell
def _(t, wd):
    # Generate a one-channel sine wave with the default settings.
    simple_tone = wd.generate_sin()

    print(
        t(
            "signal_info",
            channels=simple_tone.n_channels,
            sampling_rate=simple_tone.sampling_rate,
            duration=f"{simple_tone.duration:.1f}",
            samples=simple_tone.n_samples,
            labels=simple_tone.labels,
        )
    )
    return (simple_tone,)


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("visualization"))
    return


@app.cell
def _(simple_tone):
    # Display a complete analysis with the public describe() API.
    # Keep the figure open so marimo can render it as cell output.
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
def _(t, wd):
    # Generate a more complex signal.
    complex_signal = wd.generate_sin(
        freqs=[440, 880, 1320],  # Fundamental plus the second and third harmonics.
        duration=2.0,  # Two seconds.
        sampling_rate=8000,  # 8 kHz sampling rate.
    ).sum()

    # Process it with a readable method chain.
    processed = complex_signal.fade(fade_ms=10).low_pass_filter(cutoff=1000)

    print(t("method_chain_complete"))

    # Inspect which operations were applied.
    processed.print_operation_history()

    # Compare the original and processed signals.
    combined_signal = complex_signal.concat_frame(processed, label_prefix="processed")

    # Pass detailed settings to the public describe() API.
    config = {
        "fmin": 100,
        "fmax": 3000,
        "cmap": "jet",
        "vmin": -80,
        "vmax": -20,
        "waveform": {"ylim": (-3, 3)},
        "spectral": {"xlim": (-60, 0)},
    }

    # Display the detailed analysis with these settings.
    combined_signal.describe(**config)
    return (combined_signal,)


@app.cell
def _(combined_signal, t):
    combined_signal.fft().plot(overlay=True, title=t("spectrum_plot_title"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("interactive_experiment"))
    return


@app.cell
def _(t, wd):
    # Define a function for trying different processing parameters.
    def experiment_with_signal(frequency=440, duration=1.0, filter_cutoff=500):
        """
        Experiment with frequency, duration, and filter cutoff.

        Generate a sine-wave signal with the requested parameters, apply a low-pass
        filter, and return the original and filtered signals for comparison.

        Args:
            frequency: Fundamental frequency in Hz. The default is 440 Hz (A4).
            duration: Signal duration in seconds. The default is 1.0 second.
            filter_cutoff: Low-pass cutoff frequency in Hz. Components above this
                frequency are attenuated.

        Returns:
            ChannelFrame: A frame containing the original signal and filtered signal.
                The channel labels are "signal" and "filtered_signal".

        Examples:
            >>> # Run with the default parameters.
            >>> result = experiment_with_signal()
            >>> result.fft().plot(overlay=True)

            >>> # Try different parameters.
            >>> result = experiment_with_signal(frequency=880, filter_cutoff=1500)
            >>> result.fft().plot(overlay=True)
        """

        # Generate the fundamental and its second harmonic.
        signal = wd.generate_sin(
            freqs=[frequency, frequency * 2],
            duration=duration,
            sampling_rate=4000,
        ).sum()

        # Apply the low-pass filter.
        filtered = signal.low_pass_filter(cutoff=filter_cutoff)

        # Add the processed signal for comparison.
        combined = signal.concat_frame(filtered, label_prefix="filtered")
        return combined

    # Run the basic experiment with default parameters.
    experiment_with_signal().fft().plot(overlay=True, title=t("spectrum_plot_title"))
    return (experiment_with_signal,)


@app.cell
def _(experiment_with_signal, t):
    # Try a higher frequency and a different cutoff.
    experiment_with_signal(frequency=880, filter_cutoff=1500).fft().plot(overlay=True, title=t("spectrum_plot_title"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("troubleshooting"))
    return


@app.cell
def _(t):
    # Print a short troubleshooting checklist.
    # !pip install "wandas[marimo,io,psychoacoustic]" --upgrade
    from importlib.metadata import version as _metadata_version

    import matplotlib

    print(
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
    mo.md(t("next_steps"))
    return


@app.cell(hide_code=True)
def _(locale, mo, navigation_markdown):
    mo.md(navigation_markdown("01_getting_started", locale))
    return


if __name__ == "__main__":
    app.run()
