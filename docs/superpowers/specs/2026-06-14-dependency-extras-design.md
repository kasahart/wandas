# Dependency Extras Split Design

## Context

Wandas currently installs notebook, visualization, IO, and psychoacoustic dependencies as runtime dependencies. This makes the core install heavy for a library whose central value is waveform-oriented data structures and analysis operations.

The current import graph also pulls optional dependencies too early:

- `wandas.__init__` imports `ChannelFrame`.
- `ChannelFrame` imports `matplotlib`, `pandas`, and `IPython.display` at module import time.
- `wandas.processing.__init__` imports all processing modules, which pulls in `librosa` and `mosqito`.
- Visualization imports `librosa.display` and `matplotlib` at module import time.

The result is that `import wandas` requires packages that are not part of the core data-structure experience.

## Goal

Use the balanced "take" option from the Matsu/Take/Ume split: keep the first-use waveform experience strong while removing notebook, visualization, broad IO, machine learning, and psychoacoustic dependencies from the core install.

`pip install wandas` must support:

- `import wandas`
- `ChannelFrame.from_numpy`
- `read_wav` and `to_wav`
- basic frame properties, slicing, arithmetic, and channel operations
- core statistics and SciPy-based processing
- filter, FFT, STFT, Welch, resample, RMS trend, and normalize operations

Use extras for workflows outside that core:

- `pip install "wandas[io]"`
- `pip install "wandas[viz]"`
- `pip install "wandas[notebook]"`
- `pip install "wandas[psychoacoustic]"`

## Dependency Layout

Core runtime dependencies:

```toml
dependencies = [
    "numpy>=2.0.2",
    "scipy>=1.13.0",
    "dask>=2024.8.0",
    "pydantic>=2.11.0",
    "soundfile",
]
```

Optional dependencies:

```toml
[project.optional-dependencies]
io = [
    "pandas",
    "h5py>=3.13.0",
]
viz = [
    "matplotlib>=3.9.0",
    "librosa",
    "japanize-matplotlib>=1.1.3",
]
notebook = [
    "ipykernel",
    "ipywidgets",
    "ipympl>=0.9.3",
    "ipycytoscape>=1.3.3",
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
require_optional_dependency("pandas", extra="io", feature="read_csv")
```

The helper should raise an `ImportError` with an actionable install command:

```text
read_csv requires optional dependency 'pandas'.
Install it with: pip install "wandas[io]"
```

This keeps error messages consistent and makes tests precise.

## Component Changes

### Packaging

Move notebook packages, visualization packages, `pandas`, `h5py`, `mosqito`, `torch`, and `tensorflow` out of core runtime dependencies and into extras.

Keep `soundfile` in core so `read_wav` remains part of the default Wandas experience.

Move `types-requests` out of runtime dependencies because it is only needed by typing or development workflows.

Remove unused runtime dependencies from core when implementation confirms they are not imported by package code. Current candidates are `cattrs` and `requests`. `tqdm` is only used by `ChannelFrameDataset`; either make that import lazy with a no-progress fallback or move the dataset progress feature behind an optional dependency.

### Core Imports

Remove optional imports from module top-level import paths:

- Move `matplotlib.pyplot`, `matplotlib.axes`, and `matplotlib.figure` imports out of `wandas.frames.channel` runtime imports.
- Move `IPython.display.Audio` and `display` imports into notebook-specific code paths.
- Move `pandas` imports in frame modules into `to_dataframe`, CSV, or IO-specific code paths, or guard them with `TYPE_CHECKING`.
- Ensure `wandas.__init__` can complete without `matplotlib`, `IPython`, `pandas`, `h5py`, `librosa`, or `mosqito`.

### IO

`read_wav` and `to_wav` remain core.

CSV and WDF features become `io` extra features:

- `read_csv` requires `pandas`.
- WDF load/save requires `h5py`.

Missing optional dependencies should produce the shared helper error with `wandas[io]`.

### Visualization And Notebook

Plotting features require `wandas[viz]`.

Notebook-specific audio display requires `wandas[notebook]`. `describe()` may still save figures when visualization dependencies are available but notebook dependencies are not; audio playback should be skipped or produce a notebook-specific missing dependency message depending on the call path.

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

Keep `librosa.display.specshow` in `viz`.

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

Visualization extra:

1. User installs `wandas[viz]`.
2. `frame.plot()`, `frame.describe()` visualization, and spectrogram rendering become available.

Notebook extra:

1. User installs `wandas[notebook]`.
2. IPython/Jupyter audio display and notebook-specific interactive behavior become available.

Psychoacoustic extra:

1. User installs `wandas[psychoacoustic]`.
2. `mosqito`-backed loudness, roughness, sharpness, and NOct operations become available.

## Error Handling

Optional dependency failures should be deterministic and user-facing.

Required behavior:

- `read_csv` without `pandas`: raise `ImportError` recommending `pip install "wandas[io]"`.
- WDF without `h5py`: raise `ImportError` recommending `pip install "wandas[io]"`.
- Plotting without `matplotlib` or `librosa`: raise `ImportError` recommending `pip install "wandas[viz]"`.
- Notebook display without `IPython`: raise `ImportError` recommending `pip install "wandas[notebook]"`, unless the feature can reasonably degrade without display.
- Psychoacoustic operations without `mosqito`: raise `ImportError` recommending `pip install "wandas[psychoacoustic]"`.
- HPSS without `librosa`: raise `ImportError` recommending `pip install "wandas[viz]"`.

## Testing

### Packaging Tests

Add tests that parse `pyproject.toml` and verify:

- notebook packages are not core dependencies.
- visualization packages are not core dependencies except where explicitly intended.
- `pandas`, `h5py`, `mosqito`, `torch`, and `tensorflow` are not core dependencies.
- `io`, `viz`, `notebook`, `psychoacoustic`, and `ml` extras exist.

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
- `import wandas` does not import optional notebook, visualization, IO, or psychoacoustic dependencies.
- `pip install "wandas[io]"` enables CSV and WDF workflows.
- `pip install "wandas[viz]"` enables plotting and spectrogram visualization workflows.
- `pip install "wandas[notebook]"` enables notebook display workflows.
- `pip install "wandas[psychoacoustic]"` enables `mosqito`-backed operations.
- Missing optional dependencies raise clear errors with the correct extra install command.
