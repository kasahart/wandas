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
    catalog = load_catalog("00_why_wandas", locale)

    def t(key, **values):
        return catalog.text(key, **values)

    return language_switch_markdown, locale, mo, navigation_markdown, t


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(f"# {t('title')}\n\n{t('intro')}")
    return


@app.cell(hide_code=True)
def _(language_switch_markdown, locale, mo):
    mo.md(language_switch_markdown("00_why_wandas", locale))
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.signal

    import wandas as wd

    return np, plt, scipy, wd


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("challenge_toolchain"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("challenge_data_management"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("challenge_reproducibility"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("traditional_example"))
    return


@app.cell
def _(np, plt, scipy, wd):
    _sampling_rate = 16000
    _duration = 1.0
    data = wd.generate_sin(freqs=[440], duration=1.0, sampling_rate=16000).data
    time_axis = np.arange(int(_sampling_rate * _duration)) / _sampling_rate
    data = data + np.random.randn(len(data))
    (b, a) = scipy.signal.butter(4, 1000 / (_sampling_rate / 2))
    filtered = scipy.signal.filtfilt(b, a, data)
    window = scipy.signal.windows.hann(len(filtered))
    windowed = filtered * window
    fft_result = np.fft.fft(windowed, norm="forward")
    _freqs = np.fft.fftfreq(len(data), 1 / _sampling_rate)
    (_fig, (_ax1, _ax2)) = plt.subplots(2, 1, figsize=(10, 8))
    _fig.suptitle("Traditional Approach")
    time_axis = np.arange(len(filtered)) / _sampling_rate
    _ax1.plot(time_axis, filtered)
    _ax1.set(title="Time Domain Signal")
    _ax1.grid(True, alpha=0.3)
    magnitude_db = 20 * np.log10(np.abs(fft_result) + 1e-10)
    _ax2.plot(_freqs[: len(_freqs) // 2], magnitude_db[: len(_freqs) // 2])
    _ax2.set(title="Filtered Spectrum", ylim=(-60, 0))
    _ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("wandas_approach"))
    return


@app.cell
def _(np, plt, wd):
    signal = wd.generate_sin(freqs=[440], duration=1.0, sampling_rate=16000)
    signal = signal + np.random.randn(signal.n_samples)
    result = signal.low_pass_filter(cutoff=1000).fft()
    (_fig, (_ax1, _ax2)) = plt.subplots(2, 1, figsize=(10, 8))
    _fig.suptitle("Wandas Approach")
    result.previous.plot(ax=_ax1, title="Time Domain Signal")
    result.plot(ax=_ax2, title="Filtered Spectrum", ylim=(-50, 10))
    plt.tight_layout()
    result.info()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("integrated_frame"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("visualization"))
    return


@app.cell
def _(wd):
    signal_1 = wd.generate_sin(freqs=[440, 880], duration=2.0, sampling_rate=16000)
    signal_1.describe()
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("environmental_analysis"))
    return


@app.cell
def _(np, wd):
    fs = 51200
    _duration = 10
    _time = np.arange(fs * _duration) / fs
    np.random.seed(42)
    ch1_noise = np.random.randn(fs * _duration) * 0.1 + 0.05 * np.sin(2 * np.pi * 0.2 * _time)
    ch2_noise = np.random.randn(fs * _duration) * 0.15
    ch3_noise = np.random.randn(fs * _duration) * 0.08 + 0.03 * np.sin(2 * np.pi * 0.4 * _time)
    recording = wd.from_numpy(
        data=np.array([ch1_noise, ch2_noise, ch3_noise]),
        sampling_rate=fs,
        ch_labels=["Location A", "Location B", "Location C"],
        ch_units="Pa",
    )
    a_weighted_rms_pa = recording.a_weighting().rms
    reference_pressure_pa = recording.channels[0].ref
    a_fast_level_db_spl = recording.sound_level(
        freq_weighting="A",
        time_weighting="Fast",
        dB=True,
    )
    a_fast_level_db_spl.plot(
        title=f"A/F-weighted level (reference {reference_pressure_pa:g} Pa)",
        ylabel=f"Level [dB SPL re {reference_pressure_pa:g} Pa]",
    )
    recording.noct_spectrum().plot(
        title="One-Third-Octave Band Spectrum of Ambient Recording",
        Aw=True,
        overlay=True,
        ylim=(20, 80),
    )
    return a_weighted_rms_pa, recording, reference_pressure_pa


@app.cell(hide_code=True)
def _(a_weighted_rms_pa, mo, recording, reference_pressure_pa, t):
    mo.md(
        t(
            "environment_result",
            channel_count=recording.n_channels,
            labels=", ".join(map(str, recording.labels)),
            rms=str(a_weighted_rms_pa),
            reference_pressure=f"{reference_pressure_pa:g}",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("ml_overview"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("ml_dataset_setup"))
    return


@app.cell
def _(np, wd):
    import os
    import tempfile

    temp_dir = tempfile.mkdtemp()
    _sampling_rate = 16000
    _duration = 10.0
    n_files = 10
    for i in range(n_files):
        _split = "train" if i < 8 else "validation"
        _split_dir = os.path.join(temp_dir, f"split={_split}")
        os.makedirs(_split_dir, exist_ok=True)
        _freqs = [440 + i * 100, 880 + i * 50]
        audio = wd.generate_sin(freqs=_freqs, duration=_duration, sampling_rate=_sampling_rate)
        audio = audio + np.random.randn(audio.n_samples) * 0.1
        filename = os.path.join(_split_dir, f"audio_sample_{i + 1:03d}.wav")
        audio.to_wav(filename)
    return n_files, temp_dir


@app.cell(hide_code=True)
def _(mo, n_files, t):
    mo.md(t("dataset_created", file_count=n_files))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("ml_preprocessing"))
    return


@app.cell
def _(temp_dir, wd):
    dataset = wd.from_folder(
        folder_path=temp_dir,
        lazy_loading=True,
        recursive=True,
        path_metadata=True,
    )
    discovered_count = len(dataset)
    selected_dataset = dataset.select(split="train")

    source_frame = selected_dataset[0]
    if source_frame is None:
        raise RuntimeError("The first selected recording could not be loaded.")

    dataset = selected_dataset.trim(start=0, end=5).resample(target_sr=8000).normalize()
    processed_frame = dataset[0]
    if processed_frame is None:
        raise RuntimeError("The first recording could not be preprocessed.")

    spectrogram_dataset = dataset.stft(n_fft=512, hop_length=256)

    spectrogram_frame = spectrogram_dataset[0]
    if spectrogram_frame is None:
        raise RuntimeError("The first recording could not be converted to a spectrogram.")

    spectrogram_frame[0].plot(title="Spectrogram Sample for ML Input")
    return dataset, discovered_count, processed_frame, source_frame, spectrogram_dataset, spectrogram_frame


@app.cell(hide_code=True)
def _(dataset, discovered_count, mo, processed_frame, source_frame, spectrogram_dataset, spectrogram_frame, t):
    mo.md(
        t(
            "preprocessing_result",
            discovered_count=discovered_count,
            selected_count=len(dataset),
            source_sampling_rate=source_frame.sampling_rate,
            source_duration=f"{source_frame.duration:.1f}",
            processed_sampling_rate=processed_frame.sampling_rate,
            processed_duration=f"{processed_frame.duration:.1f}",
            spectrogram_count=len(spectrogram_dataset),
            frequency_bins=spectrogram_frame.n_freq_bins,
            time_frames=spectrogram_frame.n_frames,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("ml_spectrogram_processing"))
    return


@app.cell
def _(np, spectrogram_dataset, wd):
    def threshold_spectrogram(data, threshold):
        return np.where(np.abs(data) < threshold, 0, data)

    def process_ml(frame):
        _ml_values = threshold_spectrogram(frame.data, threshold=0.05)
        return wd.SpectrogramFrame.from_numpy(
            data=_ml_values,
            sampling_rate=frame.sampling_rate,
            n_fft=frame.n_fft,
            hop_length=frame.hop_length,
            win_length=frame.win_length,
            window=frame.window,
            metadata=frame.metadata,
            channel_metadata=frame.channels,
            previous=frame,
        )

    ml_results = spectrogram_dataset.apply(process_ml)
    _ml_result = ml_results[0]
    if _ml_result is None or _ml_result.previous is None:
        raise RuntimeError("The first spectrogram could not be processed.")
    _ml_result.previous.plot(vmin=-60, vmax=0, title="Original Spectrogram")
    _ml_result.plot(vmin=-60, vmax=0, title="ML Spectrogram")
    return (ml_results,)


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("ml_validation"))
    return


@app.cell
def _(ml_results):
    _ml_result = ml_results[0]
    if _ml_result is None or _ml_result.previous is None:
        raise RuntimeError("The first spectrogram could not be processed.")
    _ml_result.previous.istft().describe()
    _ml_result.istft().describe()
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("cleanup"))
    return


@app.cell
def _(temp_dir):
    import shutil

    shutil.rmtree(temp_dir)
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("quality_control"))
    return


@app.cell
def _(np, wd):
    np.random.seed(42)
    normal_vibration = wd.from_numpy(
        data=np.random.randn(1, 16000) * 0.1,
        sampling_rate=16000,
        ch_labels=["Normal Vibration"],
    )

    _vibration_time = np.arange(16000) / 16000
    abnormal_vibration = wd.from_numpy(
        data=np.random.randn(1, 16000) * 0.3 + np.sin(2 * np.pi * 2 * _vibration_time),
        sampling_rate=16000,
        ch_labels=["Abnormal Vibration"],
    )

    normal_vibration.info()
    abnormal_vibration.info()

    def extract_features(vibration_data: wd.ChannelFrame) -> tuple[float, wd.ChannelFrame]:
        preprocessed = vibration_data.band_pass_filter(20, 1000)
        rms = preprocessed.rms
        return (rms[0], preprocessed)

    normal_features, normal_preprocessed = extract_features(normal_vibration)
    abnormal_features, abnormal_preprocessed = extract_features(abnormal_vibration)

    threshold = (normal_features + abnormal_features) / 2
    detected = abnormal_features > threshold

    ax = normal_preprocessed.rms_plot()
    abnormal_preprocessed.rms_plot(ax=ax, title="RMS Comparison")
    return abnormal_features, abnormal_preprocessed, detected, normal_features, normal_preprocessed, threshold


@app.cell(hide_code=True)
def _(abnormal_features, detected, mo, normal_features, t, threshold):
    values = {
        "abnormal": f"{abnormal_features:.3f}",
        "normal": f"{normal_features:.3f}",
        "threshold": f"{threshold:.3f}",
    }
    if detected:
        _ = mo.md(t("quality_detected", **values))
    else:
        _ = mo.md(t("quality_normal", **values))
    return


@app.cell
def _(abnormal_preprocessed, normal_preprocessed):
    normal_preprocessed.describe()
    abnormal_preprocessed.describe()
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("first_code"))
    return


@app.cell
def _(plt, wd):
    plt.rcParams["figure.figsize"] = (10, 6)
    return


@app.cell
def _(wd):
    signal_2 = wd.generate_sin(freqs=[1000, 4000], duration=2.0, sampling_rate=16000)
    signal_2.info()
    return (signal_2,)


@app.cell
def _(signal_2):
    signal_2.describe()
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("describe_result"))
    return


@app.cell
def _(signal_2):
    processed = signal_2.low_pass_filter(cutoff=2000).normalize()
    return (processed,)


@app.cell(hide_code=True)
def _(mo, processed, t):
    mo.md(t("chain_result", operations=", ".join(op["operation"] for op in processed.operation_history)))
    return


@app.cell
def _(plt, processed, signal_2):
    (_fig, (_ax1, _ax2)) = plt.subplots(1, 2, figsize=(12, 4))
    signal_2.fft().plot(ax=_ax1, title="Original Signal Spectrum")
    processed.fft().plot(ax=_ax2, title="Filtered Signal Spectrum")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("benefits"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("audience"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("next_steps"))
    return


@app.cell(hide_code=True)
def _(locale, mo, navigation_markdown):
    mo.md(navigation_markdown("00_why_wandas", locale))
    return


if __name__ == "__main__":
    app.run()
