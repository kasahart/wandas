# Fix Error Message

Transform error message to follow Wandas 3-element pattern: WHAT/WHY/HOW

## The Pattern

Every error message must contain:

1. **WHAT**: Clear statement of the problem
2. **WHY**: Show actual vs expected values
3. **HOW**: Actionable solution

## Template

```python
raise <ErrorType>(
    f"<WHAT: Clear problem description>\n"
    f"  Got: {actual_value}\n"
    f"  Expected: {expected_value}\n"
    f"<HOW: Specific, actionable solution>"
)
```

## Examples

### Before (Bad)

```python
# Too vague, no context
raise ValueError("Invalid cutoff")

# Japanese text
raise ValueError("sampling_rate不一致")

# Missing solution
raise ValueError(f"overlap must be in [0.0, 1.0], got {overlap}")
```

### After (Good)

```python
# Clear, complete, actionable
raise ValueError(
    f"Cutoff frequency out of range\n"
    f"  Got: {cutoff} Hz\n"
    f"  Expected: 0 to {nyquist} Hz (Nyquist frequency)\n"
    f"Use cutoff < sampling_rate / 2 to fix this error."
)

# English, with context
raise ValueError(
    f"Sampling rate mismatch\n"
    f"  Signal 1: {self.sampling_rate} Hz\n"
    f"  Signal 2: {other.sampling_rate} Hz\n"
    f"Use signal.resample({other.sampling_rate}) to match sampling rates."
)

# With explanation
raise ValueError(
    f"Overlap parameter out of range\n"
    f"  Got: {overlap}\n"
    f"  Expected: 0.0 to 1.0 (0% to 100% overlap)\n"
    f"Please set overlap to a value between 0.0 and 1.0."
)
```

## Common Error Types

### ValueError
Used when value is of correct type but invalid:

```python
raise ValueError(
    f"<Parameter> out of range\n"
    f"  Got: {value}\n"
    f"  Expected: {valid_range}\n"
    f"<How to fix>"
)
```

### TypeError
Used when value has wrong type:

```python
raise TypeError(
    f"Invalid type for <parameter>\n"
    f"  Got: {type(value).__name__}\n"
    f"  Expected: {expected_types}\n"
    f"<How to fix>"
)
```

### FileNotFoundError
Used when file doesn't exist:

```python
raise FileNotFoundError(
    f"Audio file not found\n"
    f"  Path: {filepath.absolute()}\n"
    f"  Current directory: {Path.cwd()}\n"
    f"Please check the file path and ensure the file exists."
)
```

### Custom Exceptions
For domain-specific errors:

```python
class InvalidSamplingRateError(ValueError):
    """Raised when sampling rate is invalid."""
    pass

raise InvalidSamplingRateError(
    f"Sampling rate mismatch\n"
    f"  File: {file_sr} Hz\n"
    f"  Expected: {expected_sr} Hz\n"
    f"Use signal.resample({expected_sr}) or set sampling_rate=None."
)
```

## Checklist

- [ ] Error message in English
- [ ] Shows actual value received
- [ ] Shows expected value/range
- [ ] Provides specific solution (not generic)
- [ ] Includes units where applicable (Hz, seconds, etc.)
- [ ] Uses consistent formatting (newlines, indentation)

## Additional Tips

### Show Context

```python
# Good: Shows both values
raise ValueError(
    f"Channel index out of range\n"
    f"  Requested: {channel}\n"
    f"  Available: 0 to {n_channels - 1}\n"
    f"  Total channels: {n_channels}\n"
    f"Use signal.n_channels to check available channels."
)
```

### Provide Examples

```python
# Good: Shows how to fix
raise TypeError(
    f"Invalid channel specification\n"
    f"  Got: {type(channel).__name__}\n"
    f"  Expected: int, list, or None\n"
    f"Examples:\n"
    f"  - channel=0  # First channel\n"
    f"  - channel=[0, 2]  # Channels 0 and 2\n"
    f"  - channel=None  # All channels"
)
```

### Include Constraints

```python
# Good: Explains why
raise ValueError(
    f"FFT size must be power of 2\n"
    f"  Got: {n_fft}\n"
    f"  Nearest power of 2: {nearest_power}\n"
    f"Power of 2 values give optimal FFT performance.\n"
    f"Use n_fft={nearest_power} for best results."
)
```

## Testing Error Messages

Always test that error messages contain expected information:

```python
def test_error_message_quality():
    """Test that error message follows WHAT/WHY/HOW pattern."""
    signal = wd.generate_sin(440, sampling_rate=44100)

    with pytest.raises(ValueError) as exc_info:
        signal.low_pass_filter(cutoff=50000)

    error_msg = str(exc_info.value)

    # WHAT: Problem description
    assert "out of range" in error_msg.lower()

    # WHY: Actual and expected values
    assert "50000" in error_msg  # Actual
    assert "22050" in error_msg  # Nyquist

    # HOW: Solution
    assert "Hz" in error_msg
```

## Reference

See `docs/development/error_message_guide.md` for comprehensive guidelines.
