# Dependency Extras Split Design

## Context

Wandas currently installs marimo, broad visualization, WDF IO, machine learning, and psychoacoustic dependencies as runtime dependencies. This makes the core install heavy for a library whose central value is waveform-oriented data structures, `describe()`, and analysis operations.

The current import graph also pulls optional dependencies too early:

- `wandas.__init__` imports `ChannelFrame`.
- `ChannelFrame` imports `IPython.display` and some plotting helpers at module import time.
- `wandas.processing.__init__` imports all processing modules, which pulls in `librosa` and `mosqito`.
- Visualization should not import `librosa.display`; spectrogram rendering is core Matplotlib.

The result is that `import wandas` requires packages that are not part of the core data-structure experience.

## Goal

Use the balanced "take" option from the Matsu/Take/Ume split: keep the first-use waveform and `describe()` experience strong while removing marimo, WDF IO, machine learning, psychoacoustic, and non-core visualization dependencies from the core install.

`pip install wandas` must support:

- `import wandas`
- `ChannelFrame.from_numpy`
- `read_wav` and `to_wav`
- basic frame properties, slicing, arithmetic, and channel operations
- core statistics and SciPy-based processing
- filter, FFT, STFT, Welch, resample, RMS trend, and normalize operations
- basic plotting and figure/export-oriented `describe()` workflows

Use extras for workflows outside that core:

- `pip install "wandas[io]"`
- `pip install "wandas[effects]"`
- `pip install "wandas[marimo]"`
- `pip install "wandas[psychoacoustic]"`
- `pip install "wandas[ml]"`

## Dependency Layout

Core runtime dependencies:

```toml
dependencies = [
    "numpy>=2.0.2",
    "scipy>=1.13.0",
    "dask>=2024.8.0",
    "soundfile",
    "pandas",
    "xarray>=2025.6.1",
    "matplotlib>=3.9.0",
    "cattrs",
    "requests>=2.32.3",
]
```

Optional dependencies:

```toml
[project.optional-dependencies]
io = [
    "h5py>=3.13.0",
]
effects = [
    "librosa",
]
marimo = [
    "ipython",
    "marimo>=0.23.3",
]
psychoacoustic = [
    "mosqito",
]
ml = [
    "torch>=2.0.0",
    "tensorflow>=2.13.0",
]
```

Existing dependency groups for `test`, `docs`, and `dev` remain available for repository workflows. They may include packages from optional extras when those workflows need them.

## Architecture

The package should distinguish import-time availability from feature-time availability.

`import wandas` should only import core modules and core dependencies. Optional dependencies are imported inside the methods or modules that need them.

A small internal helper should centralize optional dependency errors, for example:

```python
require_optional_dependency("h5py", extra="io", feature="WDF load")
```

The helper should raise an `ImportError` with an actionable install command:

```text
WDF load requires optional dependency 'h5py'.
Install it with: pip install "wandas[io]"
```

This keeps error messages consistent and makes tests precise.

## Component Changes

### Packaging

Move marimo app packages, WDF-only IO packages, non-core visualization packages, `mosqito`, `torch`, and `tensorflow` out of core runtime dependencies and into extras. Keep `pandas`, `xarray`, and `matplotlib` in core because current core frame/data and `describe()` workflows depend on them.

Keep `soundfile` in core so `read_wav` remains part of the default Wandas experience.

Move `types-requests` out of runtime dependencies because it is only needed by typing or development workflows.

Remove unused runtime dependencies from core when implementation confirms they are not imported by package code. `pydantic` is unused after the metadata dataclass migration and should not remain in core. `tqdm` is only used by `ChannelFrameDataset`; either make that import lazy with a no-progress fallback or move the dataset progress feature behind an optional dependency.

### Core Imports

Remove optional imports from module top-level import paths:

- Move `matplotlib.pyplot`, `matplotlib.axes`, and `matplotlib.figure` imports out of `wandas.frames.channel` runtime imports.
- Move `IPython.display.Audio` and `display` imports into marimo display code paths.
- Keep `pandas` imports lazy or under `TYPE_CHECKING` where practical, but treat missing pandas as a broken core install rather than an `io` extra issue.
- Ensure `wandas.__init__` can complete without `IPython`, `h5py`, `librosa`, `mosqito`, `torch`, or `tensorflow`.

### IO

`read_wav` and `to_wav` remain core.

WDF features become `io` extra features:

- CSV uses core `pandas`.
- WDF load/save requires `h5py`.

Missing `h5py` should produce the shared helper error with `wandas[io]`. Missing `pandas` should be reported as a core dependency problem.

### Visualization And Marimo

Basic Matplotlib plotting, `SpectrogramFrame.plot()`, and figure/export-oriented `describe()` workflows remain core. Effect helpers that require `librosa.effects` require `wandas[effects]`.

marimo audio/display workflows require `wandas[marimo]`. `describe()` may still save or return figures when marimo dependencies are not installed; audio playback should be skipped or produce a marimo-specific missing dependency message depending on the call path.

### Processing Registry

Avoid importing every processing module from `wandas.processing.__init__` if that imports optional dependencies.

Core operations should be registered without requiring `librosa` or `mosqito`.

Psychoacoustic operations and NOct operations that use `mosqito` should lazy import their implementation and raise the shared missing dependency error for `wandas[psychoacoustic]` when unavailable.

### Librosa Isolation

Use PR #210 as implementation reference for removing `librosa` from core paths, but do not include Pyodide, marimo, or learning-path changes in this dependency split.

Core replacements:

- Replace `librosa.amplitude_to_db` with a NumPy implementation.
- Replace `librosa.resample` with a SciPy implementation such as `scipy.signal.resample_poly`.
- Replace `librosa.feature.rms` with a NumPy implementation.
- Replace `librosa.util.normalize` with a NumPy implementation.
- Replace `librosa.A_weighting` with a local A-weighting dB helper.

`SpectrogramFrame.plot()` and `describe()` should render spectrograms with Matplotlib directly so spectrogram visualization remains core. Keep `librosa.effects` in `effects` for HPSS features.

Do not vendor HPSS code as part of this change. HPSS remains an optional `librosa` feature for now.

## Data Flow

Default install:

1. User installs `wandas`.
2. `import wandas` succeeds with only core dependencies.
3. `wd.read_wav(...)` succeeds through the core audio IO path.
4. `ChannelFrame` operations and core processing work without optional extras.

IO extra:

1. User installs `wandas[io]`.
2. `wd.read_csv(...)` and WDF load/save become available.

Effects extra:

1. User installs `wandas[effects]`.
2. librosa-backed effect helpers such as HPSS become available.

marimo extra:

1. User installs `wandas[marimo]`.
2. IPython-backed audio display and marimo app workflows become available.

Psychoacoustic extra:

1. User installs `wandas[psychoacoustic]`.
2. `mosqito`-backed loudness, roughness, sharpness, and NOct operations become available.

## Error Handling

Optional dependency failures should be deterministic and user-facing.

Required behavior:

- `read_csv` without `pandas`: raise `ImportError` recommending `pip install "wandas[io]"`.
- WDF without `h5py`: raise `ImportError` recommending `pip install "wandas[io]"`.
- Plotting without `matplotlib`: raise `ImportError` recommending `pip install "wandas"` because Matplotlib is core. Spectrogram plotting must not require `librosa`.
- marimo display without `IPython`: raise `ImportError` recommending `pip install "wandas[marimo]"`, unless the feature can reasonably degrade without display.
- Psychoacoustic operations without `mosqito`: raise `ImportError` recommending `pip install "wandas[psychoacoustic]"`.
- Effects without `librosa`: raise `ImportError` recommending `pip install "wandas[effects]"`.

## Testing

### Packaging Tests

Add tests that parse `pyproject.toml` and verify:

- marimo packages are not core dependencies.
- Effect packages are not core dependencies.
- `pandas`, `h5py`, `mosqito`, `torch`, and `tensorflow` are not core dependencies.
- `io`, `effects`, `marimo`, `psychoacoustic`, and `ml` extras exist.

### Minimal Import Tests

Run `import wandas` in a subprocess while blocking optional modules:

- `matplotlib`
- `IPython`
- `pandas`
- `h5py`
- `librosa`
- `mosqito`

The import must succeed.

### Core Behavior Tests

With optional modules blocked, verify:

- `ChannelFrame.from_numpy`
- basic properties such as `n_channels`, `n_samples`, `duration`, `time`
- slicing and arithmetic
- core statistics
- filters
- FFT/STFT/Welch
- resample
- RMS trend
- normalize

### Optional Error Tests

With optional modules blocked, call optional features and assert the shared actionable `ImportError`.

### Numerical Regression Tests

For `librosa` replacements, add focused tests that compare against known expected behavior or previous behavior within tolerance:

- amplitude-to-dB conversion
- RMS trend shape and values
- resampling output length and simple signal behavior
- normalization behavior
- A-weighting dB values

## Out Of Scope

- Pyodide or WASM-specific learning-path changes.
- marimo setup changes.
- Vendoring HPSS from `librosa`.
- Redesigning the public processing API.
- Removing `soundfile` from core.

## Acceptance Criteria

- `pip install wandas` provides the core waveform frame experience.
- `import wandas` does not import optional marimo, visualization, IO, or psychoacoustic dependencies.
- `pip install "wandas[io]"` enables WDF workflows.
- `pip install "wandas[effects]"` enables librosa-backed audio effect workflows.
- `pip install "wandas[marimo]"` enables marimo display workflows.
- `pip install "wandas[psychoacoustic]"` enables `mosqito`-backed operations.
- `pip install "wandas[ml]"` enables Torch/TensorFlow tensor conversion workflows.
- Missing optional dependencies raise clear errors with the correct extra install command.
