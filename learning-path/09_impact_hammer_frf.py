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
    catalog = load_catalog("09_impact_hammer_frf", locale)

    def t(key, **values):
        return catalog.text(key, **values)

    return catalog, docs_relative_href, language_switch_markdown, locale, mo, navigation_markdown, t


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(f"# {t('title')}\n\n{t('intro')}")
    return


@app.cell(hide_code=True)
def _(language_switch_markdown, locale, mo):
    mo.md(language_switch_markdown("09_impact_hammer_frf", locale))
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import signal

    import wandas as wd

    return np, pd, plt, signal, wd


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("sample"))
    return


@app.cell
def _(np, signal, wd):
    sampling_rate = 4096

    # A resonance at 250 Hz stands in for the structure under test.
    _r = np.exp(-0.02 * 2 * np.pi * 250 / sampling_rate)
    _theta = 2 * np.pi * 250 / sampling_rate * np.sqrt(1 - 0.02**2)
    structure = ([1 - _r], [1, -2 * _r * np.cos(_theta), _r**2])

    # Five hits; the fourth is hit too hard and the accelerometer clips.
    _force = np.zeros(6 * sampling_rate)
    for _time, _peak in zip([0.5, 1.5, 2.5, 3.5, 4.5], [50, 60, 55, 200, 52], strict=True):
        _start = int(_time * sampling_rate)
        _force[_start : _start + 9] += _peak * np.hanning(9)
    _response = signal.lfilter(*structure, _force)
    _response += np.random.default_rng(0).normal(0, 0.002, _response.size)
    _clip = 1.5 * np.abs(_response[: 3 * sampling_rate]).max()

    recording = wd.from_numpy(
        np.vstack([_force, np.clip(_response, -_clip, _clip)]),
        sampling_rate=sampling_rate,
        ch_labels=["force", "accel"],
        ch_units=["N", "m/s^2"],
    )
    return recording, sampling_rate, structure


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("cut_hits"))
    return


@app.cell
def _(recording):
    hits = recording.trigger(10.0, duration=0.5, pre=0.01, channel="force")
    return (hits,)


@app.cell(hide_code=True)
def _(hits, mo, pd, t):
    hit_table = pd.DataFrame(
        {
            t("table.event_id"): hits.event_ids,
            t("table.trigger_time"): hits.trigger_times.round(4),
        }
    )
    mo.vstack([mo.md(t("cut_result", count=len(hits), trigger_channel=hits.trigger_channel)), hit_table])
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("inspect"))
    return


@app.cell
def _(hits, plt):
    hits.plot()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("peaks"))
    return


@app.cell
def _(hits, np):
    # to_numpy() hands every hit to other tools as (hit, channel, sample).
    hit_peaks = np.abs(hits.to_numpy()).max(axis=-1)
    return (hit_peaks,)


@app.cell(hide_code=True)
def _(hit_peaks, hits, mo, pd, t):
    peak_table = pd.DataFrame(
        {
            t("table.event_id"): hits.event_ids,
            t("table.force_peak"): hit_peaks[:, 0].round(1),
            t("table.accel_peak"): hit_peaks[:, 1].round(2),
        }
    )
    mo.vstack([mo.md(t("peaks_result")), peak_table])
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t("average"))
    return


@app.cell
def _(hits):
    frf_all = hits.transfer_function()
    coherence_all = hits.coherence()

    good_hits = hits.drop([3])
    frf_good = good_hits.transfer_function()
    coherence_good = good_hits.coherence()
    return coherence_all, coherence_good, frf_all, frf_good, good_hits


@app.cell
def _(coherence_all, coherence_good, frf_all, frf_good, np, plt, sampling_rate, signal, structure):
    # The known structure gives the FRF that the hit average should recover.
    _, true_frf = signal.freqz(*structure, worN=frf_good.freqs, fs=sampling_rate)

    _fig, (_gain_ax, _coherence_ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    _gain_ax.semilogy(frf_all.freqs, np.abs(frf_all.data), label="all hits")
    _gain_ax.semilogy(frf_good.freqs, np.abs(frf_good.data), label="hit 3 dropped")
    _gain_ax.semilogy(frf_good.freqs, np.abs(true_frf), "k--", label="true structure")
    _gain_ax.set_ylabel("|H1| [(m/s^2)/N]")
    _gain_ax.set_ylim(0.01, 10)
    _gain_ax.legend()
    _coherence_ax.plot(coherence_all.freqs, coherence_all.data, label="all hits")
    _coherence_ax.plot(coherence_good.freqs, coherence_good.data, label="hit 3 dropped")
    _coherence_ax.set_ylabel("Coherence [-]")
    _coherence_ax.set_xlabel("Frequency [Hz]")
    _coherence_ax.set_xlim(20, 500)
    _coherence_ax.legend()
    _fig.tight_layout()
    _fig
    return (true_frf,)


@app.cell(hide_code=True)
def _(coherence_all, coherence_good, frf_all, frf_good, good_hits, hits, mo, np, pd, t, true_frf):
    _band = (frf_good.freqs >= 20) & (frf_good.freqs <= 500)

    def _gain_error(frf):
        return np.max(np.abs(np.abs(frf.data[_band]) - np.abs(true_frf[_band])) / np.abs(true_frf[_band]))

    def _min_coherence(coherence):
        return coherence.data[_band].min().round(3)

    comparison = pd.DataFrame(
        {
            t("table.hits"): [t("table.all_hits"), t("table.dropped")],
            t("table.n_hits"): [len(hits), len(good_hits)],
            t("table.min_coherence"): [_min_coherence(coherence_all), _min_coherence(coherence_good)],
            t("table.max_error"): [round(100 * _gain_error(frf_all), 1), round(100 * _gain_error(frf_good), 1)],
        }
    )
    mo.vstack([mo.md(t("average_result")), comparison])
    return


@app.cell(hide_code=True)
def _(catalog, docs_relative_href, locale, mo, t):
    suffix = f" ({catalog.text('navigation.japanese_only')})" if locale == "en" else ""
    api_link = f"[Frames API reference{suffix}]({docs_relative_href(locale, 'api/frames/')})"
    mo.md(t("summary", api_link=api_link))
    return


@app.cell(hide_code=True)
def _(locale, mo, navigation_markdown):
    mo.md(navigation_markdown("09_impact_hammer_frf", locale))
    return


if __name__ == "__main__":
    app.run()
