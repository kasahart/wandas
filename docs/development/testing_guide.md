# Wandas Testing Guide

**Last Updated**: 2025-11-05  
**Status**: Active  
**Related**: [Copilot Instructions](../../.github/copilot-instructions.md) | [Coding Standards](./coding_standards.md)

## Purpose

This guide provides comprehensive testing guidelines for the Wandas library. It covers test structure, validation techniques, fixtures, and best practices specific to signal processing and numerical computing.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Structure](#test-structure)
3. [Numerical Validation](#numerical-validation)
4. [Fixtures and Test Data](#fixtures-and-test-data)
5. [Metadata Testing](#metadata-testing)
6. [Coverage Requirements](#coverage-requirements)
7. [Performance Testing](#performance-testing)
8. [Testing Tools](#testing-tools)

## Testing Philosophy

### Core Principles

1. **Test Behavior, Not Implementation**: Focus on what the code does, not how
2. **Theoretical Validation**: Compare against known theoretical values
3. **Independence**: Tests run independently without shared state
4. **Clarity**: Test names and assertions are self-documenting
5. **Comprehensive**: Aim for 100% coverage (minimum 90%)

### Test-Driven Development (TDD)

Follow the Red-Green-Refactor cycle:

1. **Red**: Write a failing test
2. **Green**: Implement minimal code to pass
3. **Refactor**: Improve code while keeping tests green

## Test Structure

### Naming Convention

Test names should describe behavior:

```python
# Good: Describes behavior and expected result
def test_normalize_produces_unit_maximum()
def test_fft_preserves_energy()
def test_low_pass_filter_attenuates_high_frequencies()
def test_resample_raises_error_on_invalid_rate()

# Bad: Unclear what is being tested
def test_normalize()
def test_fft()
def test_filter()
```

### Organization

Group related tests in classes:

```python
class TestChannelFrame:
    """Tests for ChannelFrame class."""
    
    class TestInitialization:
        """Tests for ChannelFrame initialization."""
        
        def test_creates_frame_with_valid_data(self):
            ...
        
        def test_raises_error_on_invalid_sampling_rate(self):
            ...
    
    class TestOperations:
        """Tests for signal processing operations."""
        
        def test_fft_returns_spectral_frame(self):
            ...
        
        def test_fft_preserves_energy(self):
            ...
```

### Test Patterns

#### Pattern 1: Arrange-Act-Assert (AAA)

```python
def test_band_pass_filter_preserves_passband():
    """Test that band-pass filter preserves frequencies in passband."""
    # Arrange
    signal = generate_sin([100, 500, 1000, 5000], duration=1.0, sr=44100)
    filter_op = BandPassFilter(44100, low_cutoff=300, high_cutoff=3000)
    
    # Act
    filtered = filter_op(signal.data)
    filtered_spectrum = np.abs(np.fft.fft(filtered))
    
    # Assert
    # 500 Hz and 1000 Hz should be preserved (in passband)
    # 100 Hz and 5000 Hz should be attenuated (outside passband)
    freqs = np.fft.fftfreq(len(filtered), 1/44100)
    
    idx_100 = np.argmin(np.abs(freqs - 100))
    idx_500 = np.argmin(np.abs(freqs - 500))
    idx_5000 = np.argmin(np.abs(freqs - 5000))
    
    assert filtered_spectrum[idx_500] > filtered_spectrum[idx_100] * 10
    assert filtered_spectrum[idx_500] > filtered_spectrum[idx_5000] * 10
```

#### Pattern 2: Given-When-Then (BDD)

```python
def test_resample_changes_sampling_rate():
    """
    Given a signal at 44100 Hz
    When resampled to 22050 Hz
    Then the sampling rate is halved
    And the number of samples is halved
    And the frequency content is preserved (up to Nyquist)
    """
    # Given
    original = generate_sin([440], duration=1.0, sr=44100)
    
    # When
    resampled = original.resample(22050)
    
    # Then
    assert resampled.sampling_rate == 22050
    assert resampled.n_samples == original.n_samples // 2
    
    # Frequency content preserved
    original_spectrum = np.abs(np.fft.fft(original.data))
    resampled_spectrum = np.abs(np.fft.fft(resampled.data))
    
    # Check 440 Hz component is preserved
    original_peak = np.max(original_spectrum)
    resampled_peak = np.max(resampled_spectrum)
    assert np.abs(original_peak - resampled_peak) / original_peak < 0.01
```

## Numerical Validation

### Critical Rule: Validate Against Theory

**Always compare against theoretical values, not just existence or non-zero checks.**

#### Good: Theoretical Validation

```python
def test_fft_parseval_theorem(sample_signal: ChannelFrame):
    """
    Test that FFT preserves energy (Parseval's theorem).
    
    Parseval's theorem states that the total energy in time domain
    equals the total energy in frequency domain:
    
    sum(|x[n]|^2) = (1/N) * sum(|X[k]|^2)
    """
    # Time domain energy
    time_energy = np.sum(np.abs(sample_signal.data) ** 2)
    
    # Frequency domain energy
    spectrum = sample_signal.fft()
    freq_energy = np.sum(np.abs(spectrum.data) ** 2) / len(sample_signal.data)
    
    # Theoretical expectation: Both energies should be equal
    np.testing.assert_allclose(time_energy, freq_energy, rtol=1e-10)
```

```python
def test_normalize_unit_amplitude():
    """
    Test that normalization produces maximum amplitude of 1.0.
    
    Theoretical expectation: After normalization,
    max(|signal|) should be exactly 1.0.
    """
    signal = generate_sin([440], duration=1.0, sr=44100, amplitude=5.0)
    normalized = signal.normalize()
    
    max_amplitude = np.max(np.abs(normalized.data))
    
    # Theoretical: Maximum should be exactly 1.0
    assert np.abs(max_amplitude - 1.0) < 1e-10
```

#### Bad: Existence Checks Only

```python
def test_fft_works(sample_signal):
    """Test FFT produces output."""
    spectrum = sample_signal.fft()
    # Bad: Only checks that result exists
    assert spectrum is not None
    assert spectrum.data is not None
    # This doesn't validate correctness!
```

```python
def test_normalize_changes_data(signal):
    """Test normalization changes the data."""
    normalized = signal.normalize()
    # Bad: Only checks that data changed
    assert not np.array_equal(signal.data, normalized.data)
    # Doesn't verify it's normalized correctly!
```

### Floating-Point Comparisons

Use appropriate tolerance for floating-point comparisons:

```python
import numpy as np
import pytest

# Option 1: NumPy's assert_allclose
def test_computation_accuracy():
    result = compute_value()
    expected = 1.23456789
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-9)

# Option 2: pytest.approx
def test_computation_with_approx():
    result = compute_value()
    assert result == pytest.approx(1.23456789, rel=1e-7, abs=1e-9)

# Option 3: Manual comparison
def test_computation_manual():
    result = compute_value()
    expected = 1.23456789
    assert np.abs(result - expected) < 1e-7
```

### Known Values Testing

Test against known signal processing results:

```python
def test_sine_wave_fft():
    """
    Test FFT of pure sine wave produces single frequency peak.
    
    Theory: FFT of sin(2πft) should have a single peak at frequency f.
    """
    # Generate 440 Hz sine wave
    duration = 1.0
    sr = 44100
    freq = 440.0
    t = np.arange(int(duration * sr)) / sr
    signal_data = np.sin(2 * np.pi * freq * t)
    
    signal = ChannelFrame(signal_data, sampling_rate=sr)
    spectrum = signal.fft()
    
    # Find peak frequency
    freqs = spectrum.freqs
    magnitudes = np.abs(spectrum.data)
    peak_freq_idx = np.argmax(magnitudes)
    peak_freq = freqs[peak_freq_idx]
    
    # Theoretical: Peak should be at 440 Hz
    assert np.abs(peak_freq - freq) < 1.0  # Within 1 Hz
```

## Fixtures and Test Data

### Fixture Design

Use pytest fixtures to share test data:

```python
import pytest
import wandas as wd
from wandas.frames.channel import ChannelFrame

@pytest.fixture
def sample_rate() -> float:
    """Standard sampling rate for tests."""
    return 44100.0

@pytest.fixture
def sample_signal(sample_rate: float) -> ChannelFrame:
    """
    Generate a standard test signal.
    
    Returns a 1-second signal containing 440 Hz and 880 Hz sine waves.
    """
    return wd.generate_sin(
        freqs=[440, 880],
        duration=1.0,
        sampling_rate=sample_rate
    )

@pytest.fixture
def noisy_signal(sample_signal: ChannelFrame) -> ChannelFrame:
    """Add Gaussian noise to sample signal (SNR = 20 dB)."""
    noise = np.random.normal(0, 0.01, sample_signal.shape)
    return ChannelFrame(
        data=sample_signal.data + noise,
        sampling_rate=sample_signal.sampling_rate
    )

@pytest.fixture
def multi_channel_signal(sample_rate: float) -> ChannelFrame:
    """Generate multi-channel test signal."""
    n_channels = 4
    duration = 1.0
    n_samples = int(duration * sample_rate)
    
    # Different frequency per channel
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        freq = 440 * (ch + 1)
        t = np.arange(n_samples) / sample_rate
        data[ch] = np.sin(2 * np.pi * freq * t)
    
    return ChannelFrame(data, sampling_rate=sample_rate)
```

### Fixture Scope

Choose appropriate fixture scope:

```python
# Function scope (default): New instance per test
@pytest.fixture
def temp_signal():
    return generate_signal()

# Class scope: Shared within test class
@pytest.fixture(scope="class")
def expensive_computation():
    return compute_expensive_result()

# Module scope: Shared within module
@pytest.fixture(scope="module")
def large_dataset():
    return load_large_dataset()

# Session scope: Shared across entire test session
@pytest.fixture(scope="session")
def test_config():
    return load_config()
```

### Parametrized Tests

Test multiple scenarios with parametrize:

```python
@pytest.mark.parametrize("freq,expected_peak", [
    (100, 100),
    (440, 440),
    (1000, 1000),
    (5000, 5000),
])
def test_fft_frequency_detection(freq, expected_peak):
    """Test FFT correctly identifies different frequencies."""
    signal = generate_sin([freq], duration=1.0, sr=44100)
    spectrum = signal.fft()
    
    peak_freq = spectrum.freqs[np.argmax(np.abs(spectrum.data))]
    assert np.abs(peak_freq - expected_peak) < 1.0

@pytest.mark.parametrize("sr", [8000, 16000, 22050, 44100, 48000])
def test_resample_to_different_rates(sample_signal, sr):
    """Test resampling to various standard sampling rates."""
    resampled = sample_signal.resample(sr)
    assert resampled.sampling_rate == sr
    assert resampled.n_samples == int(sample_signal.duration * sr)
```

## Metadata Testing

### Operation History

Always verify operation history is updated:

```python
def test_filter_records_operation_history(sample_signal):
    """Test that filtering records operation in history."""
    # Original should have no operations (if newly generated)
    original_ops = len(sample_signal.operation_history)
    
    # Apply filter
    filtered = sample_signal.low_pass_filter(cutoff=1000)
    
    # Verify history extended
    assert len(filtered.operation_history) == original_ops + 1
    
    # Verify operation details
    last_op = filtered.operation_history[-1]
    assert last_op.name == "low_pass_filter"
    assert last_op.params["cutoff"] == 1000
    assert "timestamp" in vars(last_op)
```

### Frame Linking

Verify previous frame references:

```python
def test_operation_creates_linked_frame(sample_signal):
    """Test that operations create properly linked frames."""
    processed = sample_signal.normalize()
    
    # Verify link to previous
    assert processed.previous is sample_signal
    
    # Chain multiple operations
    filtered = processed.low_pass_filter(1000)
    assert filtered.previous is processed
    assert filtered.previous.previous is sample_signal
```

### Metadata Preservation

Test that metadata is preserved appropriately:

```python
def test_operation_preserves_metadata(sample_signal):
    """Test that operations preserve channel metadata."""
    # Set custom metadata
    sample_signal.channel_metadata.name = "Test Channel"
    sample_signal.channel_metadata.unit = "Pa"
    
    # Apply operation
    processed = sample_signal.normalize()
    
    # Verify metadata preserved
    assert processed.channel_metadata.name == "Test Channel"
    assert processed.channel_metadata.unit == "Pa"
    assert processed.sampling_rate == sample_signal.sampling_rate
```

## Coverage Requirements

### Target Coverage

- **Overall target**: 100% coverage
- **Minimum acceptable**: 90% coverage
- **Per-module target**: 95% coverage

### Measuring Coverage

```bash
# Generate coverage report
uv run pytest --cov=wandas --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html

# Check specific module
uv run pytest tests/frames/test_channel_frame.py \
    --cov=wandas.frames.channel \
    --cov-report=term-missing
```

### Acceptable Coverage Exceptions

Use `# pragma: no cover` only for:

1. **Platform-specific code**:
```python
if sys.platform == "win32":  # pragma: no cover
    path = windows_specific_path()
```

2. **Type checking blocks**:
```python
if TYPE_CHECKING:  # pragma: no cover
    from typing import SomeType
```

3. **Debug code** (should be minimal):
```python
if DEBUG:  # pragma: no cover
    print(f"Debug: {value}")
```

4. **Abstract methods that must be overridden**:
```python
def _process_array(self, x):  # pragma: no cover
    raise NotImplementedError("Subclass must implement")
```

## Performance Testing

### Using pytest-benchmark

```python
import pytest

def test_fft_performance(benchmark, sample_signal):
    """Benchmark FFT performance."""
    # benchmark automatically handles timing and statistics
    result = benchmark(sample_signal.fft)
    
    # Verify result is correct
    assert result is not None
    assert result.data.shape == (sample_signal.n_samples // 2 + 1,)

def test_filter_performance(benchmark):
    """Benchmark filter performance on large signal."""
    large_signal = generate_sin([440], duration=60.0, sr=44100)
    
    result = benchmark(large_signal.low_pass_filter, cutoff=1000)
    
    assert result.n_samples == large_signal.n_samples
```

### Performance Assertions

```python
import time

def test_lazy_evaluation_performance():
    """Test that Dask operations don't compute immediately."""
    large_signal = generate_large_signal(duration=600.0)  # 10 minutes
    
    # This should be fast (no computation)
    start = time.time()
    filtered = large_signal.low_pass_filter(1000)
    duration = time.time() - start
    
    # Should take <1 second without computation
    assert duration < 1.0
    
    # Computation happens here
    start = time.time()
    result = filtered.compute()
    duration = time.time() - start
    
    # This will be slower but should still be reasonable
    assert duration < 60.0  # Should process within 60 seconds
```

## Testing Tools

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/frames/test_channel_frame.py

# Run specific test
uv run pytest tests/frames/test_channel_frame.py::test_fft_preserves_energy

# Run tests matching pattern
uv run pytest -k "fft"

# Verbose output
uv run pytest -v

# Stop on first failure
uv run pytest -x

# Show print statements
uv run pytest -s
```

### Test Markers

Use markers to categorize tests:

```python
import pytest

@pytest.mark.slow
def test_large_signal_processing():
    """Test that takes a long time."""
    ...

@pytest.mark.integration
def test_end_to_end_workflow():
    """Integration test of multiple components."""
    ...

@pytest.mark.parametrize("rate", [8000, 16000, 44100])
def test_multiple_rates(rate):
    """Parametrized test."""
    ...
```

Run tests by marker:

```bash
# Run only fast tests (skip slow)
uv run pytest -m "not slow"

# Run only integration tests
uv run pytest -m integration

# Run slow tests in parallel
uv run pytest -m slow -n auto
```

### Test Configuration

Configure pytest in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--tb=short",
    "--cov=wandas",
    "--cov-report=term-missing",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
]
```

---

**Related Documentation**:
- [Coding Standards](./coding_standards.md) - Code quality requirements
- [Error Message Guide](./error_message_guide.md) - Error handling patterns
- [Copilot Instructions](../../.github/copilot-instructions.md) - Quick reference
