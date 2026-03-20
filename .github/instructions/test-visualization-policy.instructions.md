---
description: "Visualization test patterns: plot strategy dispatch, axes return types, parameter forwarding, and memory leak prevention"
applyTo: "tests/visualization/**"
---
# Wandas Test Policy: Visualization (`tests/visualization/`)

Visualization tests verify **plot generation correctness** and **consistency with frame methods**.
Since numerical accuracy is guaranteed by Processing tests, Visualization tests focus on
**whether correct data is passed to plots in the correct format**.

**Prerequisite**: This file is applied together with [test-grand-policy.instructions.md](test-grand-policy.instructions.md).

---

## Common Fixtures for Visualization Tests

Define the following fixtures in `conftest.py`. Set the non-interactive backend (`matplotlib.use("Agg")`) before importing matplotlib.

- **`channel_frame`**: Standard mono frame for plot tests. Use deterministic signals with known analytical solutions.
- **`stereo_frame`**: Multi-channel frame for plot tests. Use deterministic signals with known analytical solutions.
- **`cleanup_plots` (autouse=True)**: Cleanup fixture that clears all figures (`fig.clf()`) and calls `plt.close("all")` after each test. Auto-applied to all tests to prevent memory leaks.

---

## Visualization Test Strategy

### What to Test (and What NOT to Test)

**Test (Visualization responsibilities):**
- PlotStrategy dispatch is correct (`create_operation("waveform")` returns the correct Strategy)
- Axes objects are returned
- Correct number of subplots are generated (corresponding to channel count)
- Axis labels, titles, and legends are set
- Behavior changes with `overlay=True/False`

**Do NOT Test (Processing/Frame responsibilities):**
- Numerical accuracy of plotted data (guaranteed by Processing tests)
- FFT or STFT computation results (guaranteed by spectral tests)
- Metadata propagation (guaranteed by frame tests)

---

## PlotStrategy Dispatch Tests

Use `wandas.visualization.plotting.create_operation` to verify plot strategy dispatch:

- Passing `"waveform"` returns a Strategy object with a `plot` method
- All registered plot types such as `"frequency"`, `"spectrogram"`, `"describe"` can be dispatched
- Passing an unregistered string raises `ValueError`

---

## Axes Return Type Tests

Test using Matplotlib's non-interactive backend (Agg):

- `channel_frame.plot()` returns `matplotlib.axes.Axes` or `Iterator`
- `stereo_frame.plot(overlay=True)` returns a single `Axes` with `n_channels` or more lines drawn
- `stereo_frame.plot(overlay=False)` generates multiple subplots with `Axes` count corresponding to channel count

---

## Describe Method Tests

`describe()` is a composite plot method requiring special verification:

- `describe(is_close=False)` returns a list of `plt.Figure` objects with count matching channel count
- `describe(is_close=True)` returns `None` and figures are closed (default behavior)
- `describe(image_save=path)` saves files to the specified path
- For multi-channel cases, verify that a channel index suffix is appended to the filename (e.g., `test_0.png`, `test_1.png`)

---

## Plot Parameter Forwarding Tests

Verify that the following parameters are correctly reflected in plot results:

- `title`: `ax.get_title()` matches the specified string
- `xlabel` / `ylabel`: `ax.get_xlabel()` / `ax.get_ylabel()` match the specified strings
- `xlim` / `ylim`: `ax.get_xlim()` / `ax.get_ylim()` match the specified range with `pytest.approx`

---

## Memory Leak Prevention

To prevent figure memory leaks in Visualization tests:

- When a Figure is obtained in an individual test, clear its internal state with `fig.clf()` and then close the window with `plt.close(fig)`. Using only `plt.close("all")` is insufficient as it does not clear internal state.
- Using the `autouse=True` `cleanup_plots` fixture automatically applies cleanup to all tests (recommended).

---

## Cross-References
- [test-grand-policy.instructions.md](test-grand-policy.instructions.md) — 4 pillars and test pyramid
- [frames-design.instructions.md](frames-design.instructions.md) — Frame method behavior that plots depend on
