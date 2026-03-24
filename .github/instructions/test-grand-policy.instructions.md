---
description: "Test grand policy: 4 pillars (immutability, metadata sync, math consistency, reference verification) and signal processing test pyramid"
applyTo: "tests/**"
---
# Wandas Test Grand Policy: AI-Driven Acoustic Testing

Wandas test code is not merely a bug detector — it is a guardrail that ensures **acoustic physical correctness** and **computational graph consistency** through AI agents.
To prevent AI hallucinations (plausible but incorrect calculations), adhere to the following **4 pillars** and **verification pyramid**.

This policy is automatically applied to all test files under `tests/**` (via `.instructions.md` + `applyTo` auto-injection).

---

## Common Fixtures — Standard Test Signals

Use test signals that are analytically predictable. Each component policy should extend the following standard fixtures.

### Fixture Naming Convention

| Suffix | Return Type | Usage Layer |
|--------|------------|-------------|
| (none) | `ChannelFrame` | Frame / Visualization tests |
| `_dask` | `(DaskArray, int)` | Processing tests (numerical verification without frame) |

**Example fixture without suffix (returns `ChannelFrame`):**

```python
import pytest
import numpy as np
from wandas.frames.channel import ChannelFrame

@pytest.fixture
def pure_sine_1khz() -> ChannelFrame:
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = np.sin(2 * np.pi * 1000 * t)
    return ChannelFrame.from_numpy(data.reshape(1, -1), sampling_rate=sr)
```

**Example fixture with `_dask` suffix (returns `(DaskArray, int)` tuple):**

```python
import pytest
import numpy as np
from wandas.utils.dask_helpers import da_from_array

@pytest.fixture
def pure_sine_1khz_dask() -> tuple:
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = np.sin(2 * np.pi * 1000 * t)
    dask_array = da_from_array(data.reshape(1, -1), chunks=(1, -1))
    return (dask_array, sr)
```

### Standard Fixtures

Define the following standard fixtures in `conftest.py`. Use deterministic signals with known analytical solutions for all signals.

- **Pure-tone fixture**: A single-frequency signal whose FFT peak position is analytically predictable.
- **Composite-tone fixture**: A signal with multiple frequency components. Used to verify filter pass/stop characteristics.
- **Calibrated pure-tone fixture**: A signal with a known sound pressure level conforming to IEC standards. Used as the standard signal for psychoacoustic tests.
- **Impulse fixture**: A unit impulse signal. Used to verify filter impulse responses.

Processing-layer fixtures (with `_dask` suffix) must return a `(DaskArray, int)` tuple.

---

## Pillar 1: Data Immutability & Lazy Evaluation Integrity

### Rule: Side-Effect-Free Operations
No operation should ever modify the internal data or metadata of the original instance. For each operation test, copy the original data before the operation, then verify that the original instance is unchanged after the operation and that the return value is a new instance.

### Rule: Dask Graph Protection
To preserve Dask lazy evaluation, verify that `result._data` is a `DaskArray` instance after the operation. Check that the operation does not unintentionally trigger a new `.compute()` call.

### Checklist
- [ ] Operation returns a new instance (`assert result is not original`)
- [ ] Original data unchanged after operation (`assert_array_equal`)
- [ ] Original metadata unchanged (sampling_rate, labels, operation_history)
- [ ] Dask graph preserved (no premature `.compute()`)
- [ ] `result._data` is a `DaskArray` instance after operation

---

## Pillar 2: Physical Domain Metadata Synchronization

### Rule: Metadata Auto-Tracking
After any operation, sampling rate, channel count, and labels must be correctly inherited or transformed. Verify that sampling rate is preserved for all operations other than resampling, and that channel count is preserved.

### Rule: Operation History Traceability
Verify that processing content and parameters are recorded in `operation_history`. For methods designed to record history, verify that the `operation_history` count increases by exactly 1, and that the last entry contains the correct **registry key** (e.g., `"lowpass_filter"`) and parameters. For structural operations that only modify the channel collection (`add_channel` / `remove_channel` / `rename_channels`, etc.), verify that `operation_history` remains unchanged.

Note: What is recorded in `operation_history` is the **registry key**, not the method name — it may differ from the frame method name (e.g., `low_pass_filter`).

### Rule: Domain Transition Metadata
For operations that change domain (time domain → frequency domain → time-frequency domain), verify that the number of axis dimensions matches the theoretical value of the domain conversion. For example, the number of frequency bins after FFT should be N/2+1 of the original time samples.

### Checklist
- [ ] Sampling rate preserved (or correctly updated for resampling)
- [ ] Channel labels propagated with operation annotation
- [ ] `operation_history` grows by exactly 1 entry per processing operation (registry-based); structural channel operations (add_channel / remove_channel / rename_channels) leave history unchanged
- [ ] History entry contains correct operation name (registry key) and params
- [ ] Domain transition produces correct axis dimensions

---

## Pillar 3: Mathematical Consistency & Transform Reversibility

### Rule: Domain Round-Trip Fidelity
For invertible transforms such as STFT→ISTFT, verify that the original signal can be recovered within `atol=1e-6` by the inverse transform.

### Rule: Shape & Type Consistency
Verify that the shape of output arrays matches the theoretical value. For example, the number of frequency bins in STFT should be `n_fft // 2 + 1`.

### Rule: Sampling Rate Constraint Enforcement
Invalid operations between frames with different sampling rates (addition, comparison, etc.) must raise `ValueError`.

### Checklist
- [ ] Round-trip transforms recover original within tolerance
- [ ] Output shapes match theoretical values
- [ ] Data types are correct (real for time, complex for spectral)
- [ ] Invalid cross-rate operations raise clear errors

---

## Pillar 4: Numerical Validity — Reference-Based Verification

AI agents risk generating "incorrect calculation code that looks logically correct" in complex signal processing formulas.
To prevent this, the following verification paradigms are mandatory.

### Rule: Wrapper Equivalence Testing
When Wandas wraps a library such as SciPy, compare results against direct calls to the external library. By comparing with the same algorithm and parameters as the implementation, exact match (`assert_allclose` default rtol=1e-7) can be required.

Example: `LowPassFilter` generates coefficients with `scipy.signal.butter(order, cutoff/nyquist, btype="low")` and applies them with `scipy.signal.filtfilt(b, a, x, axis=1)`, so comparing against the same call gives an exact match.

### Rule: Theoretical Value Verification
When implementing an algorithm from scratch, use analytically predictable signals (pure tone, impulse, DC) to compare against theoretical values. For example, for a 1 kHz pure tone FFT, verify that the peak appears in the 1 kHz bin (within ±1 bin).

### Rule: Tolerance Standards

| Category | Default Tolerance | Rationale |
|----------|------------------|-----------|
| Wrapper equivalence | Exact match (`assert_array_equal` or `assert_allclose` default) | Same algorithm, exact numeric result |
| Round-trip transforms | `atol=1e-6` | Windowing/overlap introduces small errors |
| Psychoacoustic metrics (MoSQITo wrapper) | Exact match | Wandas wraps MoSQITo directly — results must be identical |
| Theoretical value verification | `rtol=1e-6` | Known analytical result, numerical precision |
| Frequency peak detection | Within 1 FFT bin | Spectral leakage at bin boundaries |

Always document the rationale for tolerances in comments (e.g., `# Float32 precision tolerance`).

### Checklist
- [ ] Wrapper operations compared against direct library calls
- [ ] Custom algorithms verified against theoretical values or known signals
- [ ] Tolerance explicitly specified with rationale comment
- [ ] Known-signal tests use analytically predictable inputs (pure sine, impulse, DC)

---

## Signal Processing Test Pyramid

Structure test cases into the following 3 layers. Higher layers need fewer tests, but none may be omitted.

### Layer 1: Unit Tests (Logic) — Base
- **Target**: Individual class methods, utility functions
- **Purpose**: Boundary value testing, type checking, basic data flow correctness
- **Examples**: Cutoff frequency at zero or above Nyquist raises `ValueError`; 3D array input is rejected

### Layer 2: Domain Tests (Physics) — Middle
- **Target**: Physical models, JIS/IEC-compliant algorithms
- **Purpose**: Verify physical validity from an acoustic engineering perspective — e.g., time constant (Fast/Slow) response characteristics, filter stopband characteristics
- **Examples**: Low-pass filter sufficiently attenuates components above cutoff frequency; A-weighting is near 0 dB at 1 kHz

### Layer 3: Integration Tests (Wrapper & Authority) — Top
- **Target**: Integration with external libraries, end-to-end calculation pipelines
- **Purpose**: Validate implementation correctness by comparing against "external authorities" — existing libraries or theoretical formulas
- **Examples**: `cf.low_pass_filter()` result matches `scipy.signal.filtfilt()` called with same parameters

### Layer Distribution Guideline
When adding new processing, ensure the following minimum test configuration:
- **Unit**: Input validation + 2-3 error cases
- **Domain**: 1-2 physical validity smoke tests (using known signals)
- **Integration**: 1 equivalence test for wrappers, 1 theoretical value test for custom implementations

---

## Test Conventions

### Naming
Test function names must follow the pattern `test_{what}_{condition}_{expected_outcome}`.

Examples:
- `test_low_pass_filter_above_nyquist_raises_error`
- `test_fft_pure_sine_peak_at_correct_frequency`

### Error Message Assertions
When verifying error messages with `pytest.raises(ExceptionType, match=r"...")`, use patterns that match only the first line. The detail sections (HOW section, etc.) are subject to change, so verify only the core error type.

### Dask Array Creation
When creating Dask arrays in Processing-layer tests, use `wandas.utils.dask_helpers.da_from_array` with channel-wise chunks of `chunks=(1, -1)`.

---

## Anti-Patterns (Avoid These)

Avoid the following patterns as they undermine test reliability:

- **Direct floating-point comparison**: Always fails due to rounding errors. Use `np.testing.assert_allclose` and explicitly state tolerance and rationale.
- **Random data usage**: Not reproducible and cannot verify analytical correctness. Use deterministic known signals (pure tone, impulse, DC).
- **"No error" only verification**: Assertions like `result is not None` guarantee nothing. Verify actual output values, types, and shapes.
- **Magic numbers without rationale**: Unexplained tolerances like `atol=0.0001` cannot be reviewed. Show in comments why that value is used.

---

## Coverage-Aware Testing

The 4 pillars and the test pyramid are also the repository's coverage planning tools.

- Before implementing, identify which touched modules, branches, and metadata transitions are at risk of becoming newly uncovered.
- After running `uv run pytest -n auto --cov=wandas --cov-report=term-missing`, inspect the touched files in the coverage report and add focused tests for:
  - the main success path,
  - validation and error branches,
  - metadata/history updates,
  - lazy Dask paths or domain-transition paths when they apply.
- Treat coverage regressions on changed code as a warning that must be resolved or explicitly documented for review.
- If a localized coverage gap cannot be closed in the same change, record the missing lines and the follow-up test idea in the implementation handoff so reviewers can evaluate the risk.

---

## Cross-References
- [frames-design.instructions.md](frames-design.instructions.md) — Frame immutability and metadata rules
- [processing-api.instructions.md](processing-api.instructions.md) — Processing layer responsibilities
- [io-contracts.instructions.md](io-contracts.instructions.md) — I/O metadata preservation contracts
- [testing-workflow.instructions.md](testing-workflow.instructions.md) — TDD workflow and quality commands
