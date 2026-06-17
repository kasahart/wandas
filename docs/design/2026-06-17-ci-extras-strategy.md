# CI Extras Strategy

PR #223 moved non-core runtime dependencies behind optional extras and made the default CI install path explicit. This note defines which dependency sets CI is expected to cover so future extras changes do not silently broaden the core install.

## Core Runtime Contract

The base package keeps signal-frame construction, basic waveform operations, CSV-backed workflows, Matplotlib plotting, and `describe()` figure generation available. `pandas` and `matplotlib` are treated as core dependencies for now; this project does not currently test or promise a pandas-free core install.

The core smoke tests should continue to block IO, notebook, psychoacoustic, visualization-adjacent, and ML-only imports such as `h5py`, `IPython`, `librosa`, `mosqito`, `torch`, and `tensorflow` where those packages are not required by the exercised path.

## Standard CI Install

The lint and test jobs install the test dependency group plus non-ML extras:

```bash
uv sync --no-dev --extra io --extra viz --extra notebook --extra psychoacoustic --group test
```

This path verifies the regular development and release surface for IO, plotting, notebook display, and psychoacoustic features without pulling in heavyweight ML frameworks.

The docs job installs the docs group plus the same non-ML extras:

```bash
uv sync --no-dev --extra io --extra viz --extra notebook --extra psychoacoustic --group docs
```

This keeps documentation examples aligned with the standard non-ML feature set while avoiding Torch and TensorFlow install cost in every documentation build.

## ML Extras

The `ml` extra owns Torch and TensorFlow tensor conversion support. These packages are intentionally excluded from the normal lint, docs, and test matrix because they materially increase install time and platform variance.

ML coverage should be added as a separate lightweight smoke job when needed. That job should install `wandas[ml]` and verify the tensor conversion boundary directly, rather than expanding the normal matrix to include Torch and TensorFlow everywhere.

## Registry Guardrail

`DEPENDENCY_REGISTRY` is the runtime source for dependency error messages and extra hints. Tests should keep checking that every registered dependency points at either the core dependencies or the matching optional extra in `pyproject.toml`, and that install hints stay aligned with the registered extra.
