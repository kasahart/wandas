---
applyTo: 'tests/**/*.py'
---

# Test Standards for Wandas

## Validation Approach

**Validate against theoretical values, not just ranges.**

### Good Examples

```python
def test_fft_preserves_energy(sample_signal):
    """Test FFT preserves energy (Parseval's theorem)."""
    # Parseval's theorem: sum(|x[n]|^2) = (1/N) * sum(|X[k]|^2)
    time_energy = np.sum(np.abs(sample_signal.data) ** 2)
    spectrum = sample_signal.fft()
    freq_energy = np.sum(np.abs(spectrum.data) ** 2) / len(sample_signal.data)

    # Theoretical value: energies should be equal
    np.testing.assert_allclose(time_energy, freq_energy, rtol=1e-10)

def test_normalize_produces_unit_maximum(sample_signal):
    """Test normalization produces max amplitude of 1.0."""
    normalized = sample_signal.normalize()

    # Theoretical value: max should be exactly 1.0
    assert np.abs(np.max(np.abs(normalized.data)) - 1.0) < 1e-10
```

### Bad Examples (Don't Do This)

```python
def test_fft_output_not_zero(sample_signal):
    """Bad: Only checks non-zero, not actual values."""
    spectrum = sample_signal.fft()
    assert np.any(spectrum.data != 0)  # ❌ Too weak

def test_filter_reduces_amplitude(sample_signal):
    """Bad: Vague range check without theoretical basis."""
    filtered = sample_signal.low_pass_filter(1000)
    assert np.max(filtered.data) < np.max(sample_signal.data)  # ❌ Not specific
```

## Test Coverage Requirements

Each function must have tests for:

1. **Normal operation**: Expected behavior with typical inputs
2. **Boundary values**: Edge of valid ranges (0, max, min)
3. **Error cases**: Invalid inputs that should raise exceptions
4. **Metadata preservation**: Operation history and channel metadata
5. **Theoretical validation**: Compare against known mathematical properties

## Test Structure

### Naming Convention

```python
def test_<function>_<scenario>():
    """Test that <function> <expected behavior>."""
```

Examples:
- `test_low_pass_filter_preserves_shape()`
- `test_low_pass_filter_attenuates_high_frequencies()`
- `test_low_pass_filter_with_invalid_cutoff_raises_error()`

### Using Fixtures

```python
import pytest
import wandas as wd

@pytest.fixture
def sample_signal():
    """Generate standard test signal."""
    return wd.generate_sin(freqs=[440, 880], duration=1.0, sampling_rate=44100)

@pytest.fixture
def sample_noise(sample_signal):
    """Add noise to sample signal."""
    noise = np.random.normal(0, 0.01, sample_signal.shape)
    return sample_signal + noise

def test_uses_fixtures(sample_signal, sample_noise):
    """Tests can use multiple fixtures."""
    ...
```

### Testing Exceptions

```python
def test_invalid_parameter_raises_error():
    """Test that invalid cutoff frequency raises ValueError."""
    signal = wd.generate_sin(440, sampling_rate=44100)

    with pytest.raises(ValueError, match="Cutoff frequency out of range"):
        signal.low_pass_filter(cutoff=50000)  # Above Nyquist

def test_error_message_contains_values():
    """Test error message includes actual values."""
    signal = wd.generate_sin(440, sampling_rate=44100)

    with pytest.raises(ValueError) as exc_info:
        signal.low_pass_filter(cutoff=50000)

    error_msg = str(exc_info.value)
    assert "50000" in error_msg  # Actual value
    assert "22050" in error_msg  # Nyquist frequency
    assert "Hz" in error_msg  # Units
```

## Mathematical Validation Examples

### FFT Tests

```python
def test_fft_parseval_theorem(signal):
    """Validate Parseval's theorem: time and frequency energy are equal."""
    ...

def test_fft_linearity(signal1, signal2):
    """Validate FFT(a + b) = FFT(a) + FFT(b)."""
    ...

def test_ifft_reconstruction(signal):
    """Validate IFFT(FFT(x)) = x."""
    ...
```

### Filter Tests

```python
def test_filter_frequency_response(signal):
    """Validate filter attenuates frequencies above cutoff."""
    cutoff = 1000
    filtered = signal.low_pass_filter(cutoff)

    # Check frequency response
    spectrum = filtered.fft()
    high_freq_idx = spectrum.freqs > cutoff * 1.5
    assert np.max(np.abs(spectrum.data[high_freq_idx])) < threshold
```

## Coverage Target

- Target: 100% code coverage
- Minimum: 90% for PR approval
- Use `pytest --cov=wandas --cov-report=term-missing` to identify gaps

## Floating Point Comparison

Always use appropriate tolerances:

```python
# Good
np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
assert actual == pytest.approx(expected, rel=1e-9)

# Bad
assert actual == expected  # ❌ Fails due to floating point errors
```
