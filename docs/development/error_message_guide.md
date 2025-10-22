# Error Message Guidelines

**Last Updated**: 2025-10-22  
**Status**: Active  
**Related**: [Copilot Instructions](.github/copilot-instructions.md)

## Purpose

This guide provides comprehensive guidelines for writing user-friendly, actionable error messages in the Wandas library. Good error messages help users quickly understand what went wrong and how to fix it.

## Table of Contents

1. [The Three-Element Rule](#the-three-element-rule)
2. [Error Message Template](#error-message-template)
3. [Error Types and Categories](#error-types-and-categories)
4. [Examples: Good vs Bad](#examples-good-vs-bad)
5. [Priority Classification](#priority-classification)
6. [Implementation Guidelines](#implementation-guidelines)
7. [Testing Error Messages](#testing-error-messages)

## The Three-Element Rule

Every error message should contain **three elements**:

### 1. **WHAT** - What is the problem?
Clear description of what went wrong or what condition failed.

### 2. **WHY** - Why is it a problem?
Explain the constraint, requirement, or expected condition.

### 3. **HOW** - How can the user fix it?
Provide actionable guidance or suggest alternatives.

## Error Message Template

### Standard Template

```python
raise ErrorType(
    f"<WHAT: Clear statement of the problem>\n"
    f"  Current: <actual value/state>\n"
    f"  Expected: <expected value/state>\n"
    f"<HOW: Actionable suggestion to fix>"
)
```

### Extended Template (for complex errors)

```python
raise ErrorType(
    f"<WHAT: Clear statement of the problem>\n"
    f"  Current: <actual value/state>\n"
    f"  Expected: <expected value/state>\n"
    f"  Reason: <WHY this constraint exists>\n"
    f"<HOW: Actionable suggestion to fix>\n"
    f"  Suggestion: <specific action>\n"
    f"  Example: <code example or use case>"
)
```

## Error Types and Categories

### ValueError
Used when the value is of correct type but has an invalid value.

**Common Cases:**
- Out of range parameters
- Invalid combinations of parameters
- Dimension mismatches

**Template:**
```python
raise ValueError(
    f"<Parameter name> must be <constraint>\n"
    f"  Got: {actual_value}\n"
    f"  Expected: <valid range or condition>\n"
    f"<Actionable suggestion>"
)
```

### TypeError
Used when a value has the wrong type.

**Common Cases:**
- Wrong argument type
- Incompatible data types
- Missing required attributes

**Template:**
```python
raise TypeError(
    f"<Parameter name> must be <expected type>\n"
    f"  Got: {type(value).__name__}\n"
    f"  Expected: <valid types>\n"
    f"<Actionable suggestion>"
)
```

### RuntimeError
Used when an error occurs during execution that doesn't fit other categories.

**Common Cases:**
- State conflicts
- Operation prerequisites not met
- Unexpected conditions

**Template:**
```python
raise RuntimeError(
    f"<Operation> failed: <reason>\n"
    f"  Current state: <description>\n"
    f"  Required state: <description>\n"
    f"<Actionable suggestion>"
)
```

### FileNotFoundError
Used when a file or path doesn't exist.

**Template:**
```python
raise FileNotFoundError(
    f"File not found: {filepath}\n"
    f"  Path: {absolute_path}\n"
    f"Please check the file path and try again."
)
```

### IndexError
Used when an index is out of range.

**Template:**
```python
raise IndexError(
    f"Index out of range\n"
    f"  Index: {index}\n"
    f"  Valid range: 0 to {max_index}\n"
    f"Please use an index within the valid range."
)
```

### KeyError
Used when a dictionary key or label is not found.

**Template:**
```python
raise KeyError(
    f"Key not found: {key}\n"
    f"  Available keys: {list(available_keys)}\n"
    f"Please use one of the available keys."
)
```

### NotImplementedError
Used for features not yet implemented or abstract methods.

**Template:**
```python
raise NotImplementedError(
    f"<Feature/Method> is not implemented\n"
    f"  Reason: <why not implemented>\n"
    f"<Alternative or future plan>"
)
```

### Custom Exceptions
For domain-specific errors, create custom exception classes.

**Example:**
```python
class InvalidSamplingRateError(ValueError):
    """Raised when sampling rate is invalid or mismatched."""
    pass

raise InvalidSamplingRateError(
    f"Sampling rate mismatch\n"
    f"  Signal 1: {sr1} Hz\n"
    f"  Signal 2: {sr2} Hz\n"
    f"Consider using resample() to match sampling rates."
)
```

## Examples: Good vs Bad

### Example 1: Parameter Validation

❌ **Bad** (Score: 0/3)
```python
raise ValueError("sampling_rate不一致")
```

Issues:
- Japanese text (should be English)
- No context about actual values
- No suggestion for fixing

✅ **Good** (Score: 3/3)
```python
raise ValueError(
    f"Sampling rate mismatch\n"
    f"  Signal 1: {self.sampling_rate} Hz\n"
    f"  Signal 2: {data.sampling_rate} Hz\n"
    f"Consider using signal.resample({data.sampling_rate}) to match sampling rates."
)
```

Benefits:
- **WHAT**: Clear statement of mismatch
- **WHY**: Shows actual values
- **HOW**: Specific method to fix

### Example 2: Range Validation

❌ **Bad** (Score: 1/3)
```python
raise ValueError(f"overlap must be in [0.0, 1.0], got {overlap}")
```

Issues:
- Has WHAT and WHY
- Missing HOW (suggestion)

✅ **Good** (Score: 3/3)
```python
raise ValueError(
    f"Overlap parameter out of range\n"
    f"  Got: {overlap}\n"
    f"  Expected: 0.0 to 1.0 (0% to 100% overlap)\n"
    f"Please set overlap to a value between 0.0 and 1.0."
)
```

Benefits:
- Explains what overlap means (0% to 100%)
- Clear valid range
- Actionable suggestion

### Example 3: Type Validation

❌ **Bad** (Score: 0/3)
```python
raise TypeError("channel must be int, list, or None")
```

Issues:
- No actual value shown
- No context
- No suggestion

✅ **Good** (Score: 3/3)
```python
raise TypeError(
    f"Invalid channel specification type\n"
    f"  Got: {type(channel).__name__}\n"
    f"  Expected: int (channel index), list (multiple channels), or None (all channels)\n"
    f"Examples:\n"
    f"  - channel=0  # First channel\n"
    f"  - channel=[0, 2]  # Channels 0 and 2\n"
    f"  - channel=None  # All channels"
)
```

Benefits:
- Shows actual type received
- Explains each valid type
- Provides concrete examples

### Example 4: Dimension Validation

❌ **Bad** (Score: 0/3)
```python
raise ValueError(
    f"データは2次元または3次元である必要があります。形状: {data.shape}"
)
```

Issues:
- Japanese text
- No suggestion
- Doesn't explain what each dimension means

✅ **Good** (Score: 3/3)
```python
raise ValueError(
    f"Invalid data shape for spectrogram\n"
    f"  Got: {data.shape} ({data.ndim}D)\n"
    f"  Expected: 2D (frequency, time) or 3D (channel, frequency, time)\n"
    f"If you have 1D time-domain data, use .stft() to convert to spectrogram first.\n"
    f"Example: signal.stft(n_fft=2048).to_spectrogram()"
)
```

Benefits:
- English text
- Explains dimension meaning
- Shows current shape
- Provides conversion method

### Example 5: File Operations

❌ **Bad** (Score: 1/3)
```python
raise FileNotFoundError(f"File not found: {path}")
```

Issues:
- Minimal context
- No troubleshooting help

✅ **Good** (Score: 3/3)
```python
raise FileNotFoundError(
    f"Audio file not found\n"
    f"  Path: {path.absolute()}\n"
    f"  Current directory: {Path.cwd()}\n"
    f"Please check:\n"
    f"  - File path is correct\n"
    f"  - File exists at the specified location\n"
    f"  - You have read permissions"
)
```

Benefits:
- Shows absolute path
- Shows current directory for context
- Provides troubleshooting checklist

## Priority Classification

Based on the analysis of 100 errors in the codebase:

### Priority: HIGH (70 errors - need improvement)
**Criteria**: Score 0-1 (missing 2-3 elements)

**Modules with highest priority:**
- `frames.channel` (23 errors, avg score: 0.7/3)
- `frames.spectrogram` (4 errors, avg score: 0.0/3)
- `utils.frame_dataset` (7 errors, avg score: 0.6/3)
- `io.wdf_io` (5 errors, avg score: 0.6/3)

**Action**: Complete rewrite following the template

### Priority: MEDIUM (30 errors - need enhancement)
**Criteria**: Score 2 (missing 1 element)

**Modules:**
- `core.base_frame` (13 errors, avg score: 1.3/3)
- `processing.filters` (5 errors, avg score: 2.0/3)

**Action**: Add missing element (usually HOW)

### Priority: LOW (0 errors - already good)
**Criteria**: Score 3 (all elements present)

**Action**: Review and maintain quality in future changes

## Implementation Guidelines

### 1. Write Error Messages in English
All error messages must be in English for consistency and international accessibility.

```python
# ❌ Bad
raise ValueError("データ長不一致")

# ✅ Good
raise ValueError("Data length mismatch")
```

### 2. Include Actual and Expected Values
Always show what was received vs what was expected.

```python
# ❌ Bad
raise ValueError("Invalid cutoff frequency")

# ✅ Good
raise ValueError(
    f"Cutoff frequency out of range\n"
    f"  Got: {cutoff} Hz\n"
    f"  Expected: 0 to {nyquist} Hz (Nyquist frequency)"
)
```

### 3. Be Specific with Suggestions
Provide concrete, actionable suggestions.

```python
# ❌ Bad
raise ValueError("Fix the sampling rate")

# ✅ Good
raise ValueError(
    f"Sampling rate mismatch\n"
    f"  Signal A: {sr_a} Hz\n"
    f"  Signal B: {sr_b} Hz\n"
    f"Use signal_b.resample({sr_a}) to match sampling rates."
)
```

### 4. Use Multi-line Messages for Clarity
Break complex error messages into multiple lines for readability.

```python
# ❌ Bad
raise ValueError(f"Expected 2D array with shape (n_channels, n_samples) but got {shape}")

# ✅ Good
raise ValueError(
    f"Invalid data shape\n"
    f"  Got: {shape} ({ndim}D)\n"
    f"  Expected: (n_channels, n_samples) (2D)\n"
    f"Reshape your data using data.reshape(n_channels, -1)"
)
```

### 5. Provide Examples When Helpful
Include code examples for complex operations.

```python
raise ValueError(
    f"Invalid window function: {window}\n"
    f"  Supported windows: hann, hamming, blackman, bartlett\n"
    f"Example: signal.stft(window='hann')"
)
```

### 6. Consider Context in Error Messages
Include relevant context that helps debugging.

```python
raise ValueError(
    f"Channel index out of range\n"
    f"  Requested channel: {channel}\n"
    f"  Available channels: 0 to {n_channels - 1}\n"
    f"  Total channels: {n_channels}\n"
    f"Use signal.n_channels to check available channels."
)
```

## Testing Error Messages

### Test That Errors Are Raised
Always test that appropriate errors are raised for invalid inputs.

```python
def test_invalid_cutoff_raises_error():
    """Test that invalid cutoff frequency raises ValueError."""
    signal = generate_sin(freq=440, sampling_rate=44100)
    
    with pytest.raises(ValueError, match="Cutoff frequency out of range"):
        signal.low_pass_filter(cutoff=50000)  # Above Nyquist
```

### Test Error Message Content
Verify that error messages contain expected information.

```python
def test_error_message_contains_values():
    """Test that error message includes actual values."""
    signal = generate_sin(freq=440, sampling_rate=44100)
    
    with pytest.raises(ValueError) as exc_info:
        signal.low_pass_filter(cutoff=50000)
    
    error_msg = str(exc_info.value)
    assert "50000" in error_msg  # Actual value
    assert "22050" in error_msg  # Nyquist frequency
    assert "Hz" in error_msg  # Units
```

### Test Multilingual Support
If supporting multiple languages in the future, ensure error messages can be localized.

```python
# Future consideration
def test_error_message_localization():
    """Test error message localization (future feature)."""
    with set_language('ja'):
        with pytest.raises(ValueError) as exc_info:
            invalid_operation()
        assert "不正な値" in str(exc_info.value)
```

## Common Patterns and Anti-Patterns

### ✅ DO: Use f-strings for Dynamic Values
```python
raise ValueError(
    f"Invalid parameter: {param_name}\n"
    f"  Got: {value}\n"
    f"  Expected: {expected}"
)
```

### ❌ DON'T: Use Plain Strings Without Context
```python
raise ValueError("Invalid parameter")
```

### ✅ DO: Explain Technical Terms
```python
raise ValueError(
    f"FFT size must be power of 2 for optimal performance\n"
    f"  Got: {n_fft}\n"
    f"  Suggested: {nearest_power_of_2}\n"
    f"Power of 2 values: 256, 512, 1024, 2048, 4096..."
)
```

### ❌ DON'T: Assume User Knows Technical Details
```python
raise ValueError(f"n_fft must be power of 2, got {n_fft}")
```

### ✅ DO: Provide Multiple Fix Options
```python
raise ValueError(
    f"Sampling rate mismatch\n"
    f"  File: {file_sr} Hz\n"
    f"  Expected: {expected_sr} Hz\n"
    f"Options:\n"
    f"  1. Resample: signal.resample({expected_sr})\n"
    f"  2. Accept any rate: read_wav(path, sampling_rate=None)\n"
    f"  3. Convert file: use audio editing software"
)
```

### ❌ DON'T: Give Only One Option
```python
raise ValueError("Sampling rates don't match")
```

## Migration Plan

### Phase 1: Investigation and Guidelines (Current)
- ✅ Extract and categorize all error messages
- ✅ Create comprehensive guidelines
- ✅ Update contribution guidelines

### Phase 2: Prioritized Improvement (Future)
1. **High Priority** (Score 0-1): 70 errors
   - Focus on most-used modules first
   - Complete rewrite following template
   
2. **Medium Priority** (Score 2): 30 errors
   - Add missing element (usually HOW)
   - Quick wins

3. **Low Priority** (Score 3): 0 errors
   - Maintain quality in new code

### Phase 3: Enforcement (Future)
- Add error message quality checks to CI
- Create automated analysis tool
- Regular quality reviews

## References

### Internal
- [Copilot Instructions](.github/copilot-instructions.md)
- [API Documentation](../src/api/)
- [Design Documents](../design/)

### External
- [Python Exception Best Practices](https://docs.python.org/3/tutorial/errors.html)
- [Google Python Style Guide - Exceptions](https://google.github.io/styleguide/pyguide.html#24-exceptions)
- [Writing Great Error Messages](https://uxdesign.cc/how-to-write-good-error-messages-858e4551cd4)

## Appendix: Error Analysis Summary

### Statistics (as of 2025-10-22)
- **Total Errors**: 100
- **By Type**:
  - ValueError: 60
  - TypeError: 16
  - NotImplementedError: 13
  - IndexError: 5
  - FileNotFoundError: 3
  - KeyError: 2
  - FileExistsError: 1

### Quality Distribution
- **High Quality** (Score 2-3): 30 errors (30%)
- **Medium Quality** (Score 1): 39 errors (39%)
- **Low Quality** (Score 0): 31 errors (31%)

### Top Modules for Improvement
1. `frames.channel` - 23 errors (avg: 0.7/3)
2. `core.base_frame` - 13 errors (avg: 1.3/3)
3. `frames.roughness` - 7 errors (avg: 1.0/3)
4. `utils.frame_dataset` - 7 errors (avg: 0.6/3)
5. `visualization.plotting` - 7 errors (avg: 1.0/3)

---

**Note**: This is a living document. Update it as error message patterns evolve and new best practices emerge.
