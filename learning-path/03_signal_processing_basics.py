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
    catalog = load_catalog("03_signal_processing_basics", locale)

    def t(key, **values):
        return catalog.text(key, **values)

    return language_switch_markdown, locale, mo, navigation_markdown, t


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(f"# {t('title')}\n\n{t('intro')}")
    return


@app.cell(hide_code=True)
def _(language_switch_markdown, locale, mo):
    mo.md(language_switch_markdown("03_signal_processing_basics", locale))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("why_signal_processing"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("time_frequency_domains"))
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    import wandas as wd

    plt.rcParams["figure.figsize"] = (12, 6)
    return np, plt, wd


@app.cell
def _(np, wd):
    np.random.seed(42)
    sampling_rate = 1000
    duration = 2.0
    time = np.arange(int(duration * sampling_rate)) / sampling_rate
    _signal = (
        1.0 * np.sin(2 * np.pi * 50 * time)
        + 0.7 * np.sin(2 * np.pi * 120 * time)
        + 0.5 * np.sin(2 * np.pi * 200 * time)
        + 0.1 * np.random.randn(len(time))
    )
    demo_signal = wd.from_numpy(data=_signal.reshape(1, -1), sampling_rate=sampling_rate, ch_labels=["Demo Signal"])
    return demo_signal, duration, sampling_rate, time


@app.cell(hide_code=True)
def _(demo_signal, duration, mo, sampling_rate, t, time):
    mo.md(
        t(
            "demo_signal_info",
            channels=demo_signal.n_channels,
            sampling_rate=sampling_rate,
            duration=f"{duration:.1f}",
            samples=len(time),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("fft_intro"))
    return


@app.cell
def _(demo_signal):
    demo_signal.plot(title="Time Domain Signal")
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("frequency_domain_intro"))
    return


@app.cell
def _(demo_signal):
    spectrum = demo_signal.fft()
    spectrum.plot(title="Frequency Domain (Amplitude Level Spectrum)")
    return (spectrum,)


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("spectrum_observation"))
    return


@app.cell
def _(plt, spectrum):
    (_fig, (_ax1, _ax2)) = plt.subplots(2, 1, figsize=(12, 8))
    spectrum.plot(ax=_ax1, title="Amplitude Level Spectrum")
    _ax2.plot(spectrum.freqs, spectrum.unwrapped_phase, "b-", linewidth=1)
    _ax2.set_title("Phase Spectrum")
    _ax2.set_xlabel("Frequency [Hz]")
    _ax2.set_ylabel("Phase [rad]")
    _ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("welch_explanation"))
    return


@app.cell
def _(demo_signal, spectrum):
    welch_result = demo_signal.welch(win_length=1024, hop_length=512, n_fft=2048)
    _ax = spectrum[0].plot(overlay=True)
    welch_result[0].plot(ax=_ax, title="FFT and Welch Spectrum", xlim=(0, 300), overlay=True)
    _ax.legend(("FFT", "Welch"))
    _ax.axvline(50, color="green", linestyle="--", alpha=0.7, linewidth=1, label="50Hz")
    _ax.axvline(120, color="orange", linestyle="--", alpha=0.7, linewidth=1, label="120Hz")
    _ax.axvline(200, color="purple", linestyle="--", alpha=0.7, linewidth=1, label="200Hz")
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("welch_details"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("filtering_overview"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("lowpass_intro"))
    return


@app.cell
def _(demo_signal, plt):
    lowpass_filtered = demo_signal.low_pass_filter(cutoff=150)
    (_fig, (_ax1, _ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    demo_signal.fft().plot(ax=_ax1, title="Original Spectrum")
    lowpass_filtered.fft().plot(ax=_ax2, title="Low-pass Filtered (150Hz)")
    plt.tight_layout()
    plt.show()
    return (lowpass_filtered,)


@app.cell(hide_code=True)
def _(demo_signal, lowpass_filtered, mo, t):
    mo.md(
        t(
            "lowpass_result",
            cutoff=150,
            original_rms=f"{demo_signal.rms[0]:.4f}",
            filtered_rms=f"{lowpass_filtered.rms[0]:.4f}",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("lowpass_effect"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("highpass_intro"))
    return


@app.cell
def _(demo_signal, plt):
    signal_with_offset = demo_signal + 2.0
    highpass_filtered = signal_with_offset.high_pass_filter(cutoff=30)
    (_fig, (_ax1, _ax2, ax3)) = plt.subplots(3, 1, figsize=(12, 9))
    demo_signal.plot(ax=_ax1, title="Original Signal")
    signal_with_offset.plot(ax=_ax2, title="With DC Offset")
    highpass_filtered.plot(ax=ax3, title="High-pass Filtered (30Hz)")
    plt.tight_layout()
    plt.show()
    return highpass_filtered, signal_with_offset


@app.cell(hide_code=True)
def _(highpass_filtered, mo, signal_with_offset, t):
    mo.md(
        t(
            "highpass_result",
            cutoff=30,
            offset_rms=f"{signal_with_offset.rms[0]:.4f}",
            filtered_rms=f"{highpass_filtered.rms[0]:.4f}",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("highpass_effect"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("bandpass_intro"))
    return


@app.cell
def _(demo_signal, plt):
    bandpass_filtered = demo_signal.band_pass_filter(low_cutoff=80, high_cutoff=160)
    (_fig, (_ax1, _ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    demo_signal.fft().plot(ax=_ax1, title="Original Spectrum")
    bandpass_filtered.fft().plot(ax=_ax2, title="Band-pass Filtered (80-160Hz)")
    _ax2.axvspan(80, 160, alpha=0.2, color="green", label="Pass Band")
    _ax2.legend()
    plt.tight_layout()
    plt.show()
    return (bandpass_filtered,)


@app.cell(hide_code=True)
def _(bandpass_filtered, demo_signal, mo, t):
    mo.md(
        t(
            "bandpass_result",
            low_cutoff=80,
            high_cutoff=160,
            center_frequency=120,
            original_rms=f"{demo_signal.rms[0]:.4f}",
            filtered_rms=f"{bandpass_filtered.rms[0]:.4f}",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("bandpass_effect"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("filter_order_intro"))
    return


@app.cell
def _(demo_signal, plt):
    orders = [2, 4, 8]
    filtered_signals = {}
    for order in orders:
        filtered = demo_signal.low_pass_filter(cutoff=100, order=order)
        filtered_signals[f"Order {order}"] = filtered
    (_fig, _axes) = plt.subplots(1, len(orders) + 1, figsize=(18, 4))
    demo_signal.fft().plot(ax=_axes[0], title="Original")
    for i, (name, _signal) in enumerate(filtered_signals.items(), 1):
        _signal.fft().plot(ax=_axes[i], title=name)
        _axes[i].axvline(100, color="red", linestyle="--", alpha=0.7, label="Cutoff")
        _axes[i].legend()
    plt.tight_layout()
    plt.show()
    return (filtered_signals,)


@app.cell(hide_code=True)
def _(filtered_signals, mo, t):
    mo.md(
        t(
            "filter_order_result",
            order_2_rms=f"{filtered_signals['Order 2'].rms[0]:.4f}",
            order_4_rms=f"{filtered_signals['Order 4'].rms[0]:.4f}",
            order_8_rms=f"{filtered_signals['Order 8'].rms[0]:.4f}",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("filter_order_effect"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("spectrogram_intro"))
    return


@app.cell
def _(np, sampling_rate, time, wd):
    time_varying_signal = np.zeros_like(time)
    mask1 = time < 1.0
    time_varying_signal[mask1] = 1.0 * np.sin(2 * np.pi * 50 * time[mask1]) + 0.8 * np.sin(
        2 * np.pi * 100 * time[mask1]
    )
    mask2 = time >= 1.0
    time_varying_signal[mask2] = 1.0 * np.sin(2 * np.pi * 150 * time[mask2]) + 0.8 * np.sin(
        2 * np.pi * 200 * time[mask2]
    )
    tv_signal = wd.from_numpy(
        data=time_varying_signal.reshape(1, -1), sampling_rate=sampling_rate, ch_labels=["Time-Varying Signal"]
    )
    return (tv_signal,)


@app.cell
def _(plt, tv_signal):
    spectrogram = tv_signal.stft(n_fft=256, hop_length=128, window="hann")
    spectrogram.plot(title="Spectrogram of Time-Varying Signal")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("spectrogram_reading"))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("workflow_intro"))
    return


@app.cell
def _(np, plt, wd):
    np.random.seed(123)
    sensor_sr = 2000
    sensor_duration = 5.0
    sensor_time = np.arange(int(sensor_duration * sensor_sr)) / sensor_sr
    drift_component = 1.2 * np.sin(2 * np.pi * 0.15 * sensor_time) + 0.0025 * sensor_time
    vibration_signal = (
        drift_component
        + 1.0 * np.sin(2 * np.pi * 55 * sensor_time) * np.exp(-sensor_time / 2)
        + 0.9 * np.sin(2 * np.pi * 120 * sensor_time)
        + 0.5 * np.sin(2 * np.pi * 280 * sensor_time)
        + 0.25 * np.random.randn(len(sensor_time))
    )
    vibration_signal = vibration_signal + 5.0
    vibration_data = wd.from_numpy(
        data=vibration_signal.reshape(1, -1),
        sampling_rate=sensor_sr,
        ch_labels=["Vibration Sensor"],
    )
    (_fig, _ax) = plt.subplots(figsize=(12, 4))
    _ax.plot(sensor_time, vibration_signal, color="#1f77b4", linewidth=0.8)
    _ax.set_title("Raw Vibration Signal with Drift and DC Offset")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return (vibration_data,)


@app.cell(hide_code=True)
def _(mo, t, vibration_data):
    mo.md(
        t(
            "sensor_info",
            channels=vibration_data.n_channels,
            samples=vibration_data.n_samples,
            sampling_rate=vibration_data.sampling_rate,
            duration=f"{vibration_data.duration:.1f}",
        )
    )
    return


@app.cell
def _(np, plt, vibration_data):
    target_band = (80.0, 180.0)
    cleaned = vibration_data.remove_dc().high_pass_filter(cutoff=20)
    denoised = cleaned.low_pass_filter(cutoff=250)
    focused = denoised.band_pass_filter(low_cutoff=target_band[0], high_cutoff=target_band[1])
    spectral_raw = vibration_data.welch()
    spectral_denoised = denoised.welch()
    spectral_focused = focused.welch()
    trend = focused.rms_trend(frame_length=1024, hop_length=256)
    spectrogram_1 = focused.stft(n_fft=512, hop_length=256)
    time_np = np.asarray(vibration_data.time)
    (_fig, _axes) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    vibration_data.plot(ax=_axes[0], label="Raw", color="#636efa", linewidth=0.8)
    cleaned.plot(ax=_axes[0], label="remove_dc + high_pass", color="#ef553b", linewidth=0.8)
    _axes[0].set_title("Step 1 -> Step 2: Drift removal")
    _axes[0].legend(loc="upper right")
    _axes[0].grid(True, alpha=0.3)
    _axes[0].set_xlim(time_np[0], 1.0)
    cleaned.plot(ax=_axes[1], label="After high-pass", color="#ef553b", linewidth=0.8)
    denoised.plot(ax=_axes[1], label="After low-pass", color="#00cc96", linewidth=0.8)
    _axes[1].set_title("Step 2 -> Step 3: Broadband noise suppression")
    _axes[1].legend(loc="upper right")
    _axes[1].grid(True, alpha=0.3)
    _axes[1].set_xlim(time_np[0], 1.0)
    denoised.plot(ax=_axes[2], label="Before band-pass", color="#00cc96", linewidth=0.8)
    focused.plot(ax=_axes[2], label="Band-focused", color="#ab63fa", linewidth=0.8)
    _axes[2].set_title("Step 3 -> Step 4: Target band extraction")
    _axes[2].set_xlabel("Time [s]")
    _axes[2].legend(loc="upper right")
    _axes[2].grid(True, alpha=0.3)
    _axes[2].set_xlim(time_np[0], 1.0)
    plt.tight_layout()
    plt.show()
    return (
        denoised,
        focused,
        spectral_denoised,
        spectral_focused,
        spectral_raw,
        spectrogram_1,
        target_band,
        trend,
    )


@app.cell(hide_code=True)
def _(mo, np, spectral_denoised, spectral_focused, spectral_raw, t, target_band, trend, vibration_data):
    def peak_summary(spectrum):
        freqs = np.asarray(spectrum.freqs)
        magnitudes = np.abs(np.asarray(spectrum.data[0]))
        top_indices = np.argsort(magnitudes)[-3:][::-1]
        return ", ".join(f"{freqs[i]:.1f} Hz" for i in top_indices)

    trend_values = np.asarray(trend.data)[0]
    mo.md(
        t(
            "workflow_results",
            duration=f"{vibration_data.duration:.1f}",
            rms=f"{vibration_data.rms[0]:.4f}",
            mean=f"{float(vibration_data.data.mean()):.4f}",
            target_band=f"{target_band[0]:.0f}–{target_band[1]:.0f} Hz",
            raw_peaks=peak_summary(spectral_raw),
            denoised_peaks=peak_summary(spectral_denoised),
            focused_peaks=peak_summary(spectral_focused),
            trend_mean=f"{trend_values.mean():.4f}",
            trend_max=f"{trend_values.max():.4f}",
            trend_std=f"{trend_values.std():.4f}",
        )
    )
    return


@app.cell
def _(
    denoised,
    focused,
    plt,
    spectral_denoised,
    spectral_focused,
    spectral_raw,
    spectrogram_1,
    target_band,
    trend,
    vibration_data,
):
    (_fig, _axes) = plt.subplots(2, 3, figsize=(18, 8))
    vibration_data.plot(ax=_axes[0, 0], title="Raw (2 kHz)", xlim=(0, 1.0))
    denoised.plot(ax=_axes[0, 1], title="Denoised", xlim=(0, 1.0))
    focused.plot(ax=_axes[0, 2], title="Band-focused", xlim=(0, 1.0))
    spectral_raw.plot(ax=_axes[1, 0], title="Raw Spectrum", xlim=(0, 500), ylim=(-120, 0))
    spectral_denoised.plot(ax=_axes[1, 1], title="Denoised Spectrum", xlim=(0, 500), ylim=(-120, 0))
    spectral_focused.plot(ax=_axes[1, 2], title="Band-focused Spectrum", xlim=(0, 500), ylim=(-120, 0))
    _axes[1, 1].axvspan(target_band[0], target_band[1], alpha=0.15, color="tab:green", label="Target band")
    _axes[1, 2].axvspan(target_band[0], target_band[1], alpha=0.15, color="tab:green")
    _axes[1, 1].legend(loc="upper right")
    for _ax in _axes.flat:
        _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    spectrogram_raw = vibration_data.stft(n_fft=512, hop_length=256)
    (_fig, (ax_trend, ax_spec_raw, ax_spec_focused)) = plt.subplots(1, 3, figsize=(18, 4))
    trend.plot(ax=ax_trend, title="RMS Trend (band-focused)", xlabel="Frame", ylabel="RMS")
    spectrogram_raw.plot(ax=ax_spec_raw, title="Spectrogram (Raw)")
    spectrogram_1.plot(ax=ax_spec_focused, title="Spectrogram (Band-focused)")
    for _ax in (ax_trend, ax_spec_raw, ax_spec_focused):
        _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("workflow_interpretation"))
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
    mo.md(navigation_markdown("03_signal_processing_basics", locale))
    return


if __name__ == "__main__":
    app.run()
