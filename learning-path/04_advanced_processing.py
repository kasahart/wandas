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
    catalog = load_catalog("04_advanced_processing", locale)

    def t(key, **values):
        return catalog.text(key, **values)

    return language_switch_markdown, locale, mo, navigation_markdown, t


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(f"# {t('title')}\n\n{t('intro')}")
    return


@app.cell(hide_code=True)
def _(language_switch_markdown, locale, mo):
    mo.md(language_switch_markdown("04_advanced_processing", locale))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("stft_section"))
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.io import wavfile

    import wandas as wd

    plt.rcParams["figure.figsize"] = (12, 6)
    return np, plt, wavfile, wd


@app.cell
def _(np, wd):
    sampling_rate = 8_000
    time = np.arange(3 * sampling_rate) / sampling_rate
    rng_time = np.random.default_rng(404)
    time_varying_values = np.zeros_like(time)
    time_varying_values[time < 1.0] = np.sin(2 * np.pi * 400 * time[time < 1.0])
    time_varying_values[(time >= 1.0) & (time < 2.0)] = np.sin(2 * np.pi * 800 * time[(time >= 1.0) & (time < 2.0)])
    time_varying_values[time >= 2.0] = 0.8 * np.sin(2 * np.pi * 1_200 * time[time >= 2.0]) + 0.5 * np.sin(
        2 * np.pi * 1_240 * time[time >= 2.0]
    )
    time_varying_values += 0.03 * rng_time.standard_normal(time.size)
    time_varying_signal = wd.from_numpy(
        data=time_varying_values.reshape(1, -1),
        sampling_rate=sampling_rate,
        ch_labels=["time_varying"],
    )
    return sampling_rate, time, time_varying_signal, time_varying_values


@app.cell
def _(plt, time_varying_signal):
    time_varying_signal.plot(title="Time-varying signal", xlabel="Time [s]", ylabel="Amplitude")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, stft_long_window, stft_short_window, stft_zero_padded, t):
    mo.md(
        t(
            "stft_parameters",
            short_first_time=f"{stft_short_window.times[0]:.3f}",
            short_last_time=f"{stft_short_window.times[-1]:.3f}",
            padded_first_time=f"{stft_zero_padded.times[0]:.3f}",
            padded_last_time=f"{stft_zero_padded.times[-1]:.3f}",
            long_first_time=f"{stft_long_window.times[0]:.3f}",
            long_last_time=f"{stft_long_window.times[-1]:.3f}",
        )
    )
    return


@app.cell
def _(time_varying_signal):
    stft_short_window = time_varying_signal.stft(
        n_fft=256,
        win_length=256,
        hop_length=64,
        window="hann",
    )
    stft_zero_padded = time_varying_signal.stft(
        n_fft=1_024,
        win_length=256,
        hop_length=64,
        window="hann",
    )
    stft_long_window = time_varying_signal.stft(
        n_fft=1_024,
        win_length=1_024,
        hop_length=256,
        window="hann",
    )
    return stft_long_window, stft_short_window, stft_zero_padded


@app.cell
def _(plt, stft_long_window, stft_short_window, stft_zero_padded):
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    stft_short_window.plot(
        ax=_axes[0],
        title="Short window",
        ylim=(0, 2_000),
        vmin=-70,
        vmax=5,
    )
    stft_zero_padded.plot(
        ax=_axes[1],
        title="Same window, zero-padded FFT",
        ylim=(0, 2_000),
        vmin=-70,
        vmax=5,
    )
    stft_long_window.plot(
        ax=_axes[2],
        title="Long window",
        ylim=(0, 2_000),
        vmin=-70,
        vmax=5,
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, sampling_rate, stft_long_window, stft_short_window, stft_zero_padded, t):
    mo.md(
        t(
            "stft_evidence",
            short_shape=stft_short_window.shape,
            padded_shape=stft_zero_padded.shape,
            long_shape=stft_long_window.shape,
            short_bin_spacing=f"{sampling_rate / stft_short_window.n_fft:.2f}",
            padded_bin_spacing=f"{sampling_rate / stft_zero_padded.n_fft:.2f}",
            short_window_scale=f"{sampling_rate / stft_short_window.win_length:.2f}",
            long_window_scale=f"{sampling_rate / stft_long_window.win_length:.2f}",
            short_time_step=f"{stft_short_window.hop_length / sampling_rate * 1_000:.1f}",
            long_time_step=f"{stft_long_window.hop_length / sampling_rate * 1_000:.1f}",
            short_first_time=f"{stft_short_window.times[0]:.3f}",
            short_last_time=f"{stft_short_window.times[-1]:.3f}",
            padded_first_time=f"{stft_zero_padded.times[0]:.3f}",
            padded_last_time=f"{stft_zero_padded.times[-1]:.3f}",
            long_first_time=f"{stft_long_window.times[0]:.3f}",
            long_last_time=f"{stft_long_window.times[-1]:.3f}",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("window_section"))
    return


@app.cell
def _(np, sampling_rate, wd):
    steady_time = np.arange(int(1.5 * sampling_rate)) / sampling_rate
    steady_values = 0.8 * np.sin(2 * np.pi * 437 * steady_time) + 0.5 * np.sin(2 * np.pi * 451 * steady_time)
    steady_signal = wd.from_numpy(
        data=steady_values.reshape(1, -1),
        sampling_rate=sampling_rate,
        ch_labels=["steady"],
    )
    return steady_signal


@app.cell
def _(steady_signal):
    window_boxcar_spectrogram = steady_signal.stft(
        n_fft=1_024,
        win_length=1_024,
        hop_length=256,
        window="boxcar",
    )
    window_hann_spectrogram = steady_signal.stft(
        n_fft=1_024,
        win_length=1_024,
        hop_length=256,
        window="hann",
    )
    window_boxcar_frame = window_boxcar_spectrogram.get_frame_at(window_boxcar_spectrogram.n_frames // 2)
    window_hann_frame = window_hann_spectrogram.get_frame_at(window_hann_spectrogram.n_frames // 2)
    return window_boxcar_frame, window_hann_frame


@app.cell
def _(plt, window_boxcar_frame, window_hann_frame):
    _fig, (_ax_boxcar, _ax_hann) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    window_boxcar_frame.plot(
        ax=_ax_boxcar,
        title="Boxcar window",
        xlim=(350, 550),
        ylim=(-80, 5),
    )
    window_hann_frame.plot(
        ax=_ax_hann,
        title="Hann window",
        xlim=(350, 550),
        ylim=(-80, 5),
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, np, t, window_boxcar_frame, window_hann_frame):
    def single_channel_spectrum(frame):
        frequencies = np.asarray(frame.freqs).reshape(-1)
        values = np.asarray(frame.dB)
        if values.ndim == 2:
            values = values[0]
        values = values.reshape(-1)
        if values.shape != frequencies.shape:
            raise ValueError("spectrum values and frequencies must have matching shapes")
        return frequencies, values

    boxcar_freqs, boxcar_db = single_channel_spectrum(window_boxcar_frame)
    hann_freqs, hann_db = single_channel_spectrum(window_hann_frame)
    tone_frequencies = (437.0, 451.0)
    exclusion_half_width = 5.0
    boxcar_off_peak_mask = np.logical_and.reduce(
        [np.abs(boxcar_freqs - frequency) > exclusion_half_width for frequency in tone_frequencies]
    )
    hann_off_peak_mask = np.logical_and.reduce(
        [np.abs(hann_freqs - frequency) > exclusion_half_width for frequency in tone_frequencies]
    )
    boxcar_off_peak = float(np.max(boxcar_db[boxcar_off_peak_mask]))
    hann_off_peak = float(np.max(hann_db[hann_off_peak_mask]))
    mo.md(
        t(
            "window_evidence",
            boxcar_peak=f"{float(np.max(boxcar_db)):.1f}",
            hann_peak=f"{float(np.max(hann_db)):.1f}",
            boxcar_off_peak=f"{boxcar_off_peak:.1f}",
            hann_off_peak=f"{hann_off_peak:.1f}",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("welch_section"))
    return


@app.cell
def _(np, sampling_rate, wd):
    welch_time = np.arange(4 * sampling_rate) / sampling_rate
    rng_welch = np.random.default_rng(405)
    welch_values = (
        0.8 * np.sin(2 * np.pi * 600 * welch_time)
        + 0.5 * np.sin(2 * np.pi * 660 * welch_time)
        + 0.2 * rng_welch.standard_normal(welch_time.size)
    )
    welch_signal = wd.from_numpy(
        data=welch_values.reshape(1, -1),
        sampling_rate=sampling_rate,
        ch_labels=["noisy_tone"],
    )
    return welch_signal


@app.cell
def _(welch_signal):
    welch_reference = welch_signal.welch(
        n_fft=512,
        win_length=512,
        hop_length=256,
        window="hann",
        average="mean",
    )
    welch_more_overlap = welch_signal.welch(
        n_fft=512,
        win_length=512,
        hop_length=128,
        window="hann",
        average="mean",
    )
    welch_zero_padded = welch_signal.welch(
        n_fft=1_024,
        win_length=512,
        hop_length=256,
        window="hann",
        average="mean",
    )
    return welch_more_overlap, welch_reference, welch_zero_padded


@app.cell
def _(plt, welch_more_overlap, welch_reference, welch_zero_padded):
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    welch_reference.plot(
        ax=_axes[0],
        title="50% overlap",
        xlim=(450, 800),
        ylim=(-70, 5),
    )
    welch_more_overlap.plot(
        ax=_axes[1],
        title="75% overlap",
        xlim=(450, 800),
        ylim=(-70, 5),
    )
    welch_zero_padded.plot(
        ax=_axes[2],
        title="Welch with zero-padded FFT",
        xlim=(450, 800),
        ylim=(-70, 5),
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, np, sampling_rate, t, welch_more_overlap, welch_reference, welch_signal, welch_zero_padded):
    reference_segment_count = 1 + (welch_signal.n_samples - 512) // 256
    overlap_segment_count = 1 + (welch_signal.n_samples - 512) // 128
    padded_segment_count = 1 + (welch_signal.n_samples - 512) // 256
    reference_db = np.asarray(welch_reference.dB)
    reference_linear = np.asarray(welch_reference.data)
    mo.md(
        t(
            "welch_evidence",
            reference_shape=welch_reference.shape,
            overlap_shape=welch_more_overlap.shape,
            padded_shape=welch_zero_padded.shape,
            reference_segments=reference_segment_count,
            overlap_segments=overlap_segment_count,
            padded_segments=padded_segment_count,
            reference_overlap=f"{512 - 256}",
            more_overlap=f"{512 - 128}",
            bin_spacing=f"{sampling_rate / welch_reference.n_fft:.2f}",
            padded_bin_spacing=f"{sampling_rate / welch_zero_padded.n_fft:.2f}",
            linear_peak=f"{float(np.max(reference_linear)):.3f}",
            db_peak=f"{float(np.max(reference_db)):.1f}",
            unit=welch_signal.channels[0].unit,
            reference=f"{welch_signal.channels[0].ref:g}",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("noct_section"))
    return


@app.cell
def _(np, wd):
    acoustic_sampling_rate = 16_000
    acoustic_time = np.arange(3 * acoustic_sampling_rate) / acoustic_sampling_rate
    acoustic_values = (
        0.020 * np.sqrt(2) * np.sin(2 * np.pi * 125 * acoustic_time)
        + 0.010 * np.sqrt(2) * np.sin(2 * np.pi * 250 * acoustic_time)
        + 0.006 * np.sqrt(2) * np.sin(2 * np.pi * 1_000 * acoustic_time)
        + 0.004 * np.sqrt(2) * np.sin(2 * np.pi * 4_000 * acoustic_time)
    )
    acoustic_signal = wd.from_numpy(
        data=acoustic_values.reshape(1, -1),
        sampling_rate=acoustic_sampling_rate,
        ch_labels=["acoustic"],
        ch_units=["Pa"],
    )
    return acoustic_signal


@app.cell
def _(acoustic_signal, np):
    noct_result = acoustic_signal.noct_spectrum(
        fmin=63,
        fmax=6_300,
        n=3,
        G=10,
        fr=1_000,
    )
    noct_band_centers = noct_result.freqs
    noct_ratio = 10 ** (3 / (10 * noct_result.n)) if noct_result.G == 10 else 2 ** (1 / noct_result.n)
    noct_band_widths = noct_band_centers * (np.sqrt(noct_ratio) - 1 / np.sqrt(noct_ratio))
    return noct_band_centers, noct_band_widths, noct_result


@app.cell
def _(noct_result, plt):
    _fig, (_ax_raw, _ax_weighted) = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
    _raw_axis = noct_result.plot(
        ax=_ax_raw,
        title="Third-octave band level",
        xlabel="Band center [Hz]",
        ylabel="Level [dB SPL re 20 uPa]",
    )
    _weighted_axis = noct_result.plot(
        ax=_ax_weighted,
        title="A-weighted third-octave level",
        xlabel="Band center [Hz]",
        ylabel="Level [dB SPL re 20 uPa]",
        Aw=True,
    )
    _raw_axis.set_xscale("log")
    _weighted_axis.set_xscale("log")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(acoustic_signal, mo, noct_band_centers, noct_band_widths, noct_result, np, t):
    noct_db = np.asarray(noct_result.dB)[0]
    noct_dba = np.asarray(noct_result.dBA)[0]
    reference_band_index = int(np.argmin(np.abs(noct_band_centers - noct_result.fr)))
    mo.md(
        t(
            "noct_evidence",
            band_count=len(noct_band_centers),
            center_start=f"{noct_band_centers[0]:.1f}",
            center_end=f"{noct_band_centers[-1]:.1f}",
            first_width=f"{noct_band_widths[0]:.1f}",
            reference_center=f"{noct_band_centers[reference_band_index]:.1f}",
            reference_level=f"{noct_db[reference_band_index]:.1f}",
            reference_a_level=f"{noct_dba[reference_band_index]:.1f}",
            n=noct_result.n,
            G=noct_result.G,
            fr=noct_result.fr,
            unit=acoustic_signal.channels[0].unit,
            reference=f"{acoustic_signal.channels[0].ref:g}",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("level_section"))
    return


@app.cell
def _(np, sampling_rate, wd):
    level_time = np.arange(4 * sampling_rate) / sampling_rate
    level_reference = 2e-5
    level_db = np.where(
        level_time < 2.0,
        60.0,
        np.where(level_time < 3.0, 80.0, 60.0),
    )
    level_rms = level_reference * 10 ** (level_db / 20)
    level_values = np.sqrt(2) * level_rms * np.sin(2 * np.pi * 1_000 * level_time)
    sound_data = wd.from_numpy(
        data=level_values.reshape(1, -1),
        sampling_rate=sampling_rate,
        ch_labels=["sound_pressure"],
        ch_units=["Pa"],
    )
    return level_time, sound_data


@app.cell
def _(sound_data):
    sound_level_linear = sound_data.sound_level(
        freq_weighting="A",
        time_weighting="Fast",
        dB=False,
    )
    sound_level_fast = sound_data.sound_level(
        freq_weighting="A",
        time_weighting="Fast",
        dB=True,
    )
    sound_level_slow = sound_data.sound_level(
        freq_weighting="A",
        time_weighting="Slow",
        dB=True,
    )
    return sound_level_fast, sound_level_linear, sound_level_slow


@app.cell
def _(plt, sound_level_fast, sound_level_linear, sound_level_slow):
    _fig, (_ax_linear, _ax_fast, _ax_slow) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    sound_level_linear.plot(
        ax=_ax_linear,
        title="Linear A/F RMS",
        ylabel="RMS [Pa]",
        ylim=(0, 0.25),
    )
    sound_level_fast.plot(
        ax=_ax_fast,
        title="A/F level",
        ylabel="Level [dB SPL re 20 uPa]",
        ylim=(40, 85),
    )
    sound_level_slow.plot(
        ax=_ax_slow,
        title="A/S level",
        ylabel="Level [dB SPL re 20 uPa]",
        ylim=(40, 85),
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(level_time, mo, np, sound_data, sound_level_fast, sound_level_linear, sound_level_slow, t):
    linear_values = np.asarray(sound_level_linear.data)
    fast_values = np.asarray(sound_level_fast.data)
    slow_values = np.asarray(sound_level_slow.data)
    low_slice = (level_time >= 0.75) & (level_time < 1.75)
    high_slice = (level_time >= 2.25) & (level_time < 2.75)
    mo.md(
        t(
            "level_evidence",
            linear_shape=sound_level_linear.shape,
            fast_shape=sound_level_fast.shape,
            slow_shape=sound_level_slow.shape,
            sampling_rate=sound_level_fast.sampling_rate,
            unit=sound_data.channels[0].unit,
            reference=f"{sound_data.channels[0].ref:g}",
            output_unit=sound_level_fast.channels[0].unit,
            linear_low=f"{float(np.mean(linear_values[low_slice])):.4f}",
            linear_high=f"{float(np.mean(linear_values[high_slice])):.4f}",
            fast_low=f"{float(np.mean(fast_values[low_slice])):.1f}",
            fast_high=f"{float(np.mean(fast_values[high_slice])):.1f}",
            slow_high=f"{float(np.mean(slow_values[high_slice])):.1f}",
            fast_peak=f"{float(np.max(fast_values)):.1f}",
            slow_peak=f"{float(np.max(slow_values)):.1f}",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("dataset_section"))
    return


@app.cell
def _(np, sampling_rate, wavfile):
    import atexit
    import tempfile
    from pathlib import Path

    temporary_directory = tempfile.TemporaryDirectory(prefix="wandas-learning-path-04-")
    atexit.register(temporary_directory.cleanup)
    comparison_root = Path(temporary_directory.name)
    comparison_time = np.arange(2 * sampling_rate) / sampling_rate
    comparison_values = {
        "baseline": 0.45 * np.sin(2 * np.pi * 440 * comparison_time),
        "changed": 0.45 * np.sin(2 * np.pi * 660 * comparison_time) + 0.15 * np.sin(2 * np.pi * 880 * comparison_time),
    }
    for condition, values in comparison_values.items():
        condition_directory = comparison_root / f"condition={condition}"
        condition_directory.mkdir(parents=True, exist_ok=True)
        pcm_values = np.asarray(values * np.iinfo(np.int16).max, dtype=np.int16)
        wavfile.write(condition_directory / f"{condition}.wav", sampling_rate, pcm_values)
    return comparison_root, temporary_directory


@app.cell
def _(comparison_root, wd):
    dataset = wd.from_folder(
        comparison_root,
        recursive=True,
        lazy_loading=True,
        path_metadata=True,
    )
    dataset_state_before = dataset.get_metadata()
    selected_dataset = dataset.select(condition="changed")
    selected_state_before_load = selected_dataset.get_metadata()
    selected_frame = selected_dataset[0]
    selected_state_after_item = selected_dataset.get_metadata()
    selected_materialized_values = selected_frame.data
    comparison_dataset = dataset.apply(
        lambda frame: frame.welch(
            n_fft=512,
            win_length=512,
            hop_length=256,
            window="hann",
            average="mean",
        )
    )
    comparison_frames = [comparison_dataset[index] for index in range(len(comparison_dataset))]
    comparison_contract = {
        "sampling_rates": sorted({float(frame.sampling_rate) for frame in comparison_frames}),
        "units": sorted({tuple(channel.unit for channel in frame.channels) for frame in comparison_frames}),
        "references": sorted({tuple(float(channel.ref) for channel in frame.channels) for frame in comparison_frames}),
        "shapes": [frame.shape for frame in comparison_frames],
    }
    return (
        comparison_contract,
        comparison_dataset,
        comparison_frames,
        dataset,
        dataset_state_before,
        selected_frame,
        selected_materialized_values,
        selected_state_after_item,
        selected_state_before_load,
        selected_dataset,
    )


@app.cell
def _(comparison_frames, plt):
    _fig, _ax = plt.subplots(figsize=(12, 5))
    for _frame in comparison_frames:
        _frame.plot(
            ax=_ax,
            overlay=True,
            label=str(_frame.metadata["condition"]),
            xlim=(300, 1_100),
            ylim=(-70, 5),
        )
    _ax.legend()
    _ax.set_title("Collection-level Welch comparison")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(
    comparison_contract,
    dataset_state_before,
    mo,
    selected_frame,
    selected_materialized_values,
    selected_state_after_item,
    selected_state_before_load,
    t,
):
    mo.md(
        t(
            "dataset_evidence",
            file_count=dataset_state_before["file_count"],
            loaded_before=dataset_state_before["loaded_count"],
            selected_count=selected_state_before_load["file_count"],
            selected_loaded_before=selected_state_before_load["loaded_count"],
            selected_loaded_after_item=selected_state_after_item["loaded_count"],
            materialized_type=type(selected_materialized_values).__name__,
            selected_condition=selected_frame.metadata["condition"],
            sampling_rates=comparison_contract["sampling_rates"],
            units=comparison_contract["units"],
            references=comparison_contract["references"],
            shapes=comparison_contract["shapes"],
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("summary"))
    return


@app.cell(hide_code=True)
def _(locale, mo, navigation_markdown):
    mo.md(navigation_markdown("04_advanced_processing", locale))
    return


if __name__ == "__main__":
    app.run()
