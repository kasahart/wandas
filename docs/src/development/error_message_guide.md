# Error Message Guidelines

## Overview

This document provides comprehensive guidelines for writing user-friendly error messages in the Wandas library. Well-crafted error messages help users quickly understand and resolve issues, improving the overall developer experience.

## The Three-Element Structure

Every error message should contain these three essential elements:

1. **What is the problem?** - Clearly state what went wrong
2. **Why is it a problem?** - Explain the constraint or condition that was violated
3. **How to fix it?** - Provide actionable guidance on resolving the issue

### Template

```python
raise SomeError(
    f"[WHAT] {problem_description}.\n"
    f"[WHY] {constraint_explanation}.\n"
    f"[HOW] {solution_guidance}"
)
```

## Error Type Guidelines

### ValueError

**When to use:** Invalid values, out-of-range parameters, or inappropriate argument values.

**Good Example:**
```python
def set_cutoff_frequency(self, cutoff: float) -> None:
    """Set cutoff frequency for the filter."""
    nyquist = self.sampling_rate / 2
    if cutoff <= 0 or cutoff >= nyquist:
        raise ValueError(
            f"Cutoff frequency must be between 0 Hz and {nyquist} Hz, got {cutoff} Hz.\n"
            f"The cutoff frequency must be positive and less than the Nyquist frequency "
            f"({nyquist} Hz) to avoid aliasing.\n"
            f"Please provide a cutoff frequency in the valid range: 0 < cutoff < {nyquist}."
        )
```

**Bad Example:**
```python
# Too vague
raise ValueError("Invalid cutoff")

# Missing solution
raise ValueError(f"Cutoff frequency {cutoff} is invalid")

# No context about valid range
raise ValueError("Cutoff must be less than Nyquist frequency")
```

### TypeError

**When to use:** Type mismatches or incorrect types for arguments.

**Good Example:**
```python
def process_signal(self, data: Union[np.ndarray, DaskArray]) -> ChannelFrame:
    """Process audio signal data."""
    if not isinstance(data, (np.ndarray, DaskArray)):
        raise TypeError(
            f"Expected numpy.ndarray or dask.array, got {type(data).__name__}.\n"
            f"The signal processing pipeline requires array-like data structures.\n"
            f"Please convert your data to numpy.ndarray or dask.array before processing."
        )
```

**Bad Example:**
```python
# Unhelpful type information
raise TypeError("Wrong type")

# Missing guidance
raise TypeError(f"data must be array, got {type(data)}")
```

### FileNotFoundError

**When to use:** File or directory doesn't exist.

**Good Example:**
```python
def load_audio_file(self, filepath: Path) -> ChannelFrame:
    """Load audio file from disk."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Audio file not found: {filepath}\n"
            f"The file path must point to an existing audio file (WAV, MP3, etc.).\n"
            f"Please check:\n"
            f"  1. The file path is correct: {filepath.absolute()}\n"
            f"  2. The file exists in the specified location\n"
            f"  3. You have read permissions for the file"
        )
```

**Bad Example:**
```python
# Too minimal
raise FileNotFoundError(str(filepath))

# Missing helpful context
raise FileNotFoundError(f"File not found: {filepath}")
```

### IndexError

**When to use:** Index out of bounds or invalid indexing.

**Good Example:**
```python
def get_channel(self, channel_index: int) -> ChannelFrame:
    """Get data for a specific channel."""
    if not 0 <= channel_index < self.n_channels:
        raise IndexError(
            f"Channel index {channel_index} is out of range for {self.n_channels} channels.\n"
            f"Valid channel indices are 0 to {self.n_channels - 1} (0-based indexing).\n"
            f"Please use an index in the range [0, {self.n_channels - 1}]."
        )
```

**Bad Example:**
```python
# Not informative
raise IndexError("Invalid channel")

# Missing valid range
raise IndexError(f"Channel {channel_index} out of bounds")
```

### NotImplementedError

**When to use:** Feature not yet implemented or abstract method that must be overridden.

**Good Example:**
```python
def save_to_format(self, format: str) -> None:
    """Save frame to specified format."""
    if format not in ["hdf5", "wav"]:
        raise NotImplementedError(
            f"Export to '{format}' format is not yet supported.\n"
            f"Currently supported formats: 'hdf5', 'wav'.\n"
            f"Please use one of the supported formats, or consider opening a feature "
            f"request at https://github.com/kasahart/wandas/issues"
        )
```

**Bad Example:**
```python
# Unhelpful
raise NotImplementedError("Not supported")

# Missing alternatives
raise NotImplementedError(f"Format {format} not implemented")
```

### KeyError

**When to use:** Missing dictionary key or metadata attribute.

**Good Example:**
```python
def get_metadata_value(self, key: str) -> Any:
    """Get value from metadata dictionary."""
    if key not in self.metadata:
        available_keys = ", ".join(sorted(self.metadata.keys()))
        raise KeyError(
            f"Metadata key '{key}' not found.\n"
            f"Available metadata keys: {available_keys}\n"
            f"Please check the spelling or use one of the available keys."
        )
```

**Bad Example:**
```python
# Standard KeyError is too terse
raise KeyError(key)

# Missing available keys
raise KeyError(f"Key {key} not found in metadata")
```

## Quality Checklist

Before committing error messages, verify:

- [ ] **Clarity**: Is the problem clearly stated?
- [ ] **Context**: Are relevant values (actual vs. expected) included?
- [ ] **Guidance**: Does it tell the user how to fix the issue?
- [ ] **Formatting**: Is the message well-formatted and readable?
- [ ] **Completeness**: Does it include all three elements (What, Why, How)?
- [ ] **Accuracy**: Are all constraints and conditions correctly stated?
- [ ] **Specificity**: Does it provide specific values rather than vague descriptions?

## Common Patterns

### Range Validation

```python
if not min_value <= value <= max_value:
    raise ValueError(
        f"{param_name} must be between {min_value} and {max_value}, got {value}.\n"
        f"{explanation_of_constraint}.\n"
        f"Please provide a value in the range [{min_value}, {max_value}]."
    )
```

### Dimension Validation

```python
if data.ndim not in expected_dims:
    expected_str = " or ".join(f"{d}D" for d in expected_dims)
    raise ValueError(
        f"Expected {expected_str} array, got {data.ndim}D array with shape {data.shape}.\n"
        f"{explanation_of_why_dimension_matters}.\n"
        f"Please reshape your data to match one of the expected dimensions."
    )
```

### Type Validation

```python
if not isinstance(obj, expected_types):
    type_names = " or ".join(t.__name__ for t in expected_types)
    raise TypeError(
        f"Expected {type_names}, got {type(obj).__name__}.\n"
        f"{explanation_of_type_requirement}.\n"
        f"Please convert your object to one of the accepted types."
    )
```

### Sampling Rate Matching

```python
if self.sampling_rate != other.sampling_rate:
    raise ValueError(
        f"Sampling rates do not match: {self.sampling_rate} Hz vs {other.sampling_rate} Hz.\n"
        f"Operations on audio signals require matching sampling rates to maintain temporal alignment.\n"
        f"Please resample one of the signals using the .resample() method to match sampling rates."
    )
```

## Priority Classification

Based on the analysis of current error messages in the codebase:

### High Priority (Should be improved immediately)

1. **Input validation errors** - Most frequently encountered by users
   - File path validation
   - Parameter range validation
   - Dimension validation
   - Type validation

2. **User-facing API errors** - Direct user interaction
   - ChannelFrame operations
   - File I/O operations
   - Signal processing operations

### Medium Priority (Should be improved soon)

1. **Processing pipeline errors** - Encountered during data processing
   - Filter operations
   - Transform operations
   - Metadata operations

2. **Configuration errors** - Setup and initialization
   - Invalid configurations
   - Incompatible settings

### Low Priority (Can be improved over time)

1. **Internal errors** - Rare edge cases
   - NotImplementedError for internal methods
   - Internal state validation

2. **Debug-level errors** - Developer-facing
   - Type checking errors
   - Assertion failures

## Examples from Wandas Codebase

### Current State Analysis

We analyzed 100 error messages across the Wandas codebase:

- **ValueError**: 60 (60%) - Most common, needs consistent improvement
- **TypeError**: 16 (16%) - Needs better type guidance
- **NotImplementedError**: 13 (13%) - Should guide to alternatives
- **IndexError**: 5 (5%) - Should include valid ranges
- **FileNotFoundError**: 3 (3%) - Should include path troubleshooting
- **KeyError**: 2 (2%) - Should list available keys
- **FileExistsError**: 1 (1%) - Should suggest alternatives

### Top Files Needing Improvement

1. `wandas/frames/channel.py` (23 errors) - Core user-facing API
2. `wandas/core/base_frame.py` (13 errors) - Base functionality
3. `wandas/visualization/plotting.py` (7 errors) - Visualization errors
4. `wandas/utils/frame_dataset.py` (7 errors) - Dataset operations
5. `wandas/frames/roughness.py` (7 errors) - Psychoacoustic operations

### Improvement Examples

#### Before: Basic Error Message
```python
# wandas/frames/channel.py
if data.ndim > 2:
    raise ValueError(f"Data must be 1-dimensional or 2-dimensional. Shape: {data.shape}")
```

#### After: Improved Error Message
```python
# wandas/frames/channel.py
if data.ndim > 2:
    raise ValueError(
        f"Data must be 1D or 2D, got {data.ndim}D array with shape {data.shape}.\n"
        f"ChannelFrame expects 1D arrays (single channel) or 2D arrays "
        f"(multiple channels with shape [n_channels, n_samples]).\n"
        f"Please reshape your {data.ndim}D array to 2D using data.reshape(n_channels, -1) "
        f"or select a specific slice that results in 1D or 2D data."
    )
```

#### Before: Minimal Type Error
```python
# wandas/frames/channel.py
raise TypeError("channel must be int, list, or None")
```

#### After: Comprehensive Type Error
```python
# wandas/frames/channel.py
raise TypeError(
    f"Channel selector must be int, list, or None, got {type(channel).__name__}.\n"
    f"Use int to select a single channel (e.g., channel=0), "
    f"list to select multiple channels (e.g., channel=[0, 1]), "
    f"or None to select all channels.\n"
    f"Please convert your channel selector to one of the supported types."
)
```

## Implementation Strategy

### Step-by-Step Process

1. **Identify** error messages that need improvement
2. **Analyze** the context and common user scenarios
3. **Draft** improved message with all three elements
4. **Review** against the quality checklist
5. **Test** with example scenarios
6. **Document** in code comments if the error logic is complex

### Testing Error Messages

Always test error messages to ensure:

```python
def test_error_message_quality():
    """Test that error messages are helpful."""
    with pytest.raises(ValueError) as exc_info:
        invalid_operation()
    
    error_msg = str(exc_info.value)
    
    # Check for key components
    assert "what" in error_msg.lower() or "expected" in error_msg.lower()
    assert "why" in error_msg.lower() or "because" in error_msg.lower() or "must" in error_msg.lower()
    assert "please" in error_msg.lower() or "try" in error_msg.lower() or "use" in error_msg.lower()
    
    # Check for specific values
    assert str(actual_value) in error_msg
    assert str(expected_value) in error_msg
```

## References

- [Python Exception Best Practices](https://docs.python.org/3/tutorial/errors.html)
- [Error Message Guidelines - Google](https://developers.google.com/tech-writing/error-messages)
- [Writing Helpful Error Messages](https://uxplanet.org/how-to-write-good-error-messages-858e4551cd4)

## Revision History

- **2024-10-22**: Initial version - Phase 1 error message improvement
