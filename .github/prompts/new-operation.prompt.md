# Create New Signal Processing Operation

Generate a new signal processing operation for Wandas following the AudioOperation pattern.

## Requirements

Ask for:
1. Operation name (e.g., "bandpass_filter", "echo_effect")
2. Input/output types (usually `NDArrayReal` or `NDArrayComplex`)
3. Parameters (e.g., cutoff frequency, delay time, gain)
4. Processing algorithm description

## Implementation Template

Create both the operation class and comprehensive test suite:

### Operation Class

```python
from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayReal
from typing import Optional

@register_operation
class <OperationName>(AudioOperation[NDArrayReal, NDArrayReal]):
    """
    <Brief description of what this operation does>

    <Extended description with mathematical background if applicable>

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the input signal in Hz.
    <param_name> : <type>
        <Description of parameter>

    Raises
    ------
    ValueError
        If <condition that raises error>

    Examples
    --------
    >>> signal = wd.read_wav("audio.wav")
    >>> processed = signal.<operation_name>(<params>)
    """

    name = "<operation_name>"

    def __init__(self, sampling_rate: float, <params>) -> None:
        # Validate parameters
        if <invalid_condition>:
            raise ValueError(
                f"<WHAT: Problem description>\n"
                f"  Got: {<actual_value>}\n"
                f"  Expected: {<expected_value>}\n"
                f"<HOW: Solution>"
            )

        super().__init__(sampling_rate, <param>=<value>)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """
        Process the input array.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array.

        Returns
        -------
        NDArrayReal
            Processed signal array.
        """
        # Implementation
        result = <processing_logic>
        return result
```

### Test Suite

```python
import pytest
import numpy as np
import wandas as wd
from wandas.frames.channel import ChannelFrame

@pytest.fixture
def sample_signal() -> ChannelFrame:
    """Generate test signal for operation."""
    return wd.generate_sin(freqs=[440, 880], duration=1.0, sampling_rate=44100)

def test_<operation>_preserves_shape(sample_signal):
    """Test that operation preserves signal shape."""
    result = sample_signal.<operation>(<params>)
    assert result.n_samples == sample_signal.n_samples
    assert result.n_channels == sample_signal.n_channels

def test_<operation>_preserves_sampling_rate(sample_signal):
    """Test that operation preserves sampling rate."""
    result = sample_signal.<operation>(<params>)
    assert result.sampling_rate == sample_signal.sampling_rate

def test_<operation>_records_history(sample_signal):
    """Test that operation is recorded in history."""
    result = sample_signal.<operation>(<params>)
    assert len(result.operation_history) == len(sample_signal.operation_history) + 1
    assert result.operation_history[-1].name == "<operation_name>"

def test_<operation>_with_invalid_param_raises_error(sample_signal):
    """Test that invalid parameter raises ValueError."""
    with pytest.raises(ValueError, match="<error_pattern>"):
        sample_signal.<operation>(<invalid_params>)

def test_<operation>_theoretical_validation(sample_signal):
    """Test operation against theoretical/mathematical properties."""
    result = sample_signal.<operation>(<params>)

    # Validate against theoretical expectation
    # Example: Check energy preservation, frequency response, etc.
    expected = <calculate_theoretical_value>
    actual = <measure_from_result>

    np.testing.assert_allclose(actual, expected, rtol=1e-10)
```

## Checklist

Ensure the implementation includes:

- [ ] Type hints on all parameters and return values
- [ ] NumPy-style docstring in English
- [ ] Parameter validation in `__init__`
- [ ] 3-element error messages (WHAT/WHY/HOW)
- [ ] Immutability (returns new frame)
- [ ] Operation history tracking
- [ ] Comprehensive tests (normal/boundary/error/theoretical)
- [ ] Usage examples in docstring
- [ ] Test coverage: 100% of new code

## Common Patterns

### Filter Operations
- Validate cutoff frequency < Nyquist
- Check filter order > 0
- Use scipy.signal for implementations

### Time-domain Effects
- Validate delay time > 0
- Check gain/mix parameters in [0, 1]
- Handle edge effects (padding/windowing)

### Frequency-domain Operations
- Ensure FFT size is power of 2
- Handle DC component appropriately
- Consider phase preservation

## After Implementation

1. Run tests: `uv run pytest tests/processing/test_<operation>.py -v`
2. Check coverage: `uv run pytest --cov=wandas.processing.<module> --cov-report=term-missing`
3. Type check: `uv run mypy wandas/processing/<module>.py`
4. Lint: `uv run ruff check wandas/processing/<module>.py`
5. Add usage example to `examples/` if appropriate
