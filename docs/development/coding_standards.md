# Wandas Coding Standards

**Last Updated**: 2025-11-05  
**Status**: Active  
**Related**: [Copilot Instructions](../../.github/copilot-instructions.md) | [Error Message Guide](./error_message_guide.md)

## Purpose

This document provides comprehensive coding standards for the Wandas library. For quick reference during development, see the Copilot instructions. This document contains detailed explanations, rationale, and extensive examples.

## Table of Contents

1. [Type Hints and Type Safety](#type-hints-and-type-safety)
2. [Array Handling (NumPy/Dask)](#array-handling-numpydask)
3. [Signal Processing Operations](#signal-processing-operations)
4. [Metadata Management](#metadata-management)
5. [Error Handling](#error-handling)
6. [Documentation Standards](#documentation-standards)
7. [Performance Optimization](#performance-optimization)
8. [Code Style](#code-style)

## Type Hints and Type Safety

### Requirements

**All code must:**
- Use complete type hints on all functions and methods
- Pass mypy strict mode (`strict = true`)
- Use type aliases from `wandas.utils.types`
- Include return type annotations (including `None`)

### Type Aliases

Use standardized type aliases from `wandas.utils.types`:

```python
from wandas.utils.types import (
    NDArrayReal,      # numpy.ndarray with real values
    NDArrayComplex,   # numpy.ndarray with complex values
    DaskArray,        # dask.array.Array
    PathLike,         # Union[str, Path]
)
```

### Examples

#### Good: Complete Type Hints

```python
from typing import Optional, Union
from wandas.utils.types import NDArrayReal, DaskArray
import dask.array as da

def apply_window(
    data: Union[NDArrayReal, DaskArray],
    window_type: str = "hann",
    axis: int = -1
) -> Union[NDArrayReal, DaskArray]:
    """
    Apply window function to signal.
    
    Parameters
    ----------
    data : NDArrayReal or DaskArray
        Input signal array.
    window_type : str, default="hann"
        Type of window to apply.
    axis : int, default=-1
        Axis along which to apply window.
        
    Returns
    -------
    NDArrayReal or DaskArray
        Windowed signal, same type as input.
    """
    window = create_window(window_type, data.shape[axis])
    return data * window
```

#### Bad: Missing Type Hints

```python
def apply_window(data, window_type="hann", axis=-1):
    """Apply window function."""  # No type information!
    window = create_window(window_type, data.shape[axis])
    return data * window
```

### Generic Types for Operations

Use TypeVars for generic operations:

```python
from typing import TypeVar, Generic
from wandas.utils.types import NDArrayReal, NDArrayComplex

InputT = TypeVar('InputT', NDArrayReal, NDArrayComplex)
OutputT = TypeVar('OutputT', NDArrayReal, NDArrayComplex)

class AudioOperation(Generic[InputT, OutputT]):
    """Base class for audio operations."""
    
    def _process_array(self, x: InputT) -> OutputT:
        """Process input array."""
        raise NotImplementedError
```

### Type Checking

Run mypy regularly during development:

```bash
# Check specific file
uv run mypy wandas/frames/channel.py

# Check entire project
uv run mypy --config-file=pyproject.toml

# Check with verbose output
uv run mypy wandas --verbose
```

## Array Handling (NumPy/Dask)

### Principles

1. **Prefer Dask**: Use Dask arrays for lazy evaluation and memory efficiency
2. **NumPy compatibility**: Support both NumPy and Dask arrays
3. **Explicit axes**: Always specify axis parameter explicitly
4. **Shape validation**: Validate array shapes at function entry

### Dask vs NumPy

**Use Dask when:**
- Processing large datasets (>1GB)
- Chaining multiple operations
- Delaying computation until necessary

**Use NumPy when:**
- Small arrays (<100MB)
- Immediate results needed
- Simple, one-off operations

### Examples

#### Array Type Handling

```python
import numpy as np
import dask.array as da
from typing import Union
from wandas.utils.types import NDArrayReal, DaskArray

def process_audio(
    data: Union[NDArrayReal, DaskArray],
    chunk_size: int = 1024
) -> DaskArray:
    """
    Process audio data with automatic Dask conversion.
    
    Parameters
    ----------
    data : NDArrayReal or DaskArray
        Input audio data.
    chunk_size : int, default=1024
        Chunk size for Dask operations.
        
    Returns
    -------
    DaskArray
        Processed audio as Dask array (not computed).
    """
    # Convert NumPy to Dask if needed
    if isinstance(data, np.ndarray):
        data = da.from_array(data, chunks=chunk_size)
    
    # Process with Dask operations
    result = da.fft.fft(data, axis=-1)
    
    # Return without computing (lazy evaluation)
    return result
```

#### Axis Handling

```python
def apply_filter(
    data: DaskArray,
    axis: int = -1
) -> DaskArray:
    """
    Apply filter along specified axis.
    
    Parameters
    ----------
    data : DaskArray
        Input data array.
    axis : int, default=-1
        Axis along which to filter. Use -1 for last axis (time).
        
    Returns
    -------
    DaskArray
        Filtered data.
        
    Raises
    ------
    ValueError
        If axis is out of bounds for array dimensions.
    """
    # Validate axis
    if axis >= data.ndim or axis < -data.ndim:
        raise ValueError(
            f"Invalid axis for array\n"
            f"  Axis: {axis}\n"
            f"  Array dimensions: {data.ndim}\n"
            f"Valid range: {-data.ndim} to {data.ndim - 1}"
        )
    
    # Normalize negative axis
    if axis < 0:
        axis = data.ndim + axis
    
    # Apply operation along axis
    return da.apply_along_axis(filter_func, axis, data)
```

#### Shape Validation

```python
def validate_array_shape(
    data: Union[NDArrayReal, DaskArray],
    expected_dims: tuple[int, ...],
    name: str = "data"
) -> None:
    """
    Validate array has expected dimensions.
    
    Parameters
    ----------
    data : NDArrayReal or DaskArray
        Array to validate.
    expected_dims : tuple of int
        Tuple of allowed dimensions (e.g., (1, 2) for 1D or 2D).
    name : str, default="data"
        Parameter name for error messages.
        
    Raises
    ------
    ValueError
        If array dimensions don't match expectations.
    """
    if data.ndim not in expected_dims:
        dims_str = " or ".join(f"{d}D" for d in expected_dims)
        raise ValueError(
            f"Invalid array dimensions for {name}\n"
            f"  Got: {data.ndim}D array with shape {data.shape}\n"
            f"  Expected: {dims_str}\n"
            f"Reshape or select appropriate data."
        )
```

## Signal Processing Operations

### Operation Pattern

All signal processing operations follow this pattern:

1. Inherit from `AudioOperation[InputT, OutputT]`
2. Add `@register_operation` decorator
3. Implement `__init__` with parameter validation
4. Implement `_process_array` method
5. Add comprehensive docstring

### Base Class Structure

```python
from typing import TypeVar, Generic, Any, Dict
from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayReal

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')

class AudioOperation(Generic[InputT, OutputT]):
    """
    Base class for audio processing operations.
    
    Subclasses must implement:
    - name: str (class attribute)
    - _process_array: method
    """
    
    name: str = "base_operation"
    
    def __init__(self, sampling_rate: float, **params: Any) -> None:
        self.sampling_rate = sampling_rate
        self.params = params
    
    def _process_array(self, x: InputT) -> OutputT:
        """Process the input array. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def __call__(self, x: InputT) -> OutputT:
        """Apply operation to input."""
        return self._process_array(x)
```

### Example: Band-Pass Filter

```python
from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayReal
import numpy as np
from scipy import signal

@register_operation
class BandPassFilter(AudioOperation[NDArrayReal, NDArrayReal]):
    """
    Apply band-pass filter to signal.
    
    This filter allows frequencies within a specified band to pass
    while attenuating frequencies outside the band.
    
    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the input signal in Hz.
    low_cutoff : float
        Lower cutoff frequency in Hz.
    high_cutoff : float
        Upper cutoff frequency in Hz.
    order : int, default=5
        Filter order. Higher order gives sharper cutoff.
        
    Raises
    ------
    ValueError
        If cutoff frequencies are invalid or exceed Nyquist frequency.
        
    Examples
    --------
    >>> filter_op = BandPassFilter(44100, low_cutoff=300, high_cutoff=3000)
    >>> filtered_signal = signal.apply_operation(filter_op)
    """
    
    name = "band_pass_filter"
    
    def __init__(
        self,
        sampling_rate: float,
        low_cutoff: float,
        high_cutoff: float,
        order: int = 5
    ) -> None:
        # Validate parameters
        nyquist = sampling_rate / 2
        
        if low_cutoff <= 0:
            raise ValueError(
                f"Invalid low cutoff frequency\n"
                f"  Got: {low_cutoff} Hz\n"
                f"  Expected: Positive value\n"
                f"Use a positive frequency value."
            )
        
        if high_cutoff >= nyquist:
            raise ValueError(
                f"High cutoff exceeds Nyquist frequency\n"
                f"  High cutoff: {high_cutoff} Hz\n"
                f"  Nyquist frequency: {nyquist} Hz\n"
                f"Reduce high_cutoff to below {nyquist} Hz."
            )
        
        if low_cutoff >= high_cutoff:
            raise ValueError(
                f"Invalid frequency band\n"
                f"  Low cutoff: {low_cutoff} Hz\n"
                f"  High cutoff: {high_cutoff} Hz\n"
                f"Ensure low_cutoff < high_cutoff."
            )
        
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.order = order
        
        # Store parameters for operation history
        super().__init__(
            sampling_rate,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            order=order
        )
        
        # Pre-compute filter coefficients
        self.sos = signal.butter(
            order,
            [low_cutoff, high_cutoff],
            btype='band',
            fs=sampling_rate,
            output='sos'
        )
    
    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """
        Apply band-pass filter to input array.
        
        Parameters
        ----------
        x : NDArrayReal
            Input signal array.
            
        Returns
        -------
        NDArrayReal
            Filtered signal array.
        """
        # Apply filter
        return signal.sosfilt(self.sos, x, axis=-1)
```

### Operation Registration

The `@register_operation` decorator makes operations discoverable:

```python
# operations are automatically registered and can be listed
from wandas.processing import list_operations

available_ops = list_operations()
# ['band_pass_filter', 'low_pass_filter', ...]
```

## Metadata Management

### Operation History

Every operation must record its execution in the operation history:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

@dataclass
class OperationRecord:
    """Record of a single operation."""
    
    name: str
    params: Dict[str, Any]
    timestamp: datetime
```

### Creating Processed Frames

When an operation creates a new frame, it must:
1. Copy and extend operation_history
2. Preserve or update channel_metadata
3. Link to previous frame

```python
from typing import Any
from datetime import datetime

def create_processed_frame(
    self,
    data: NDArrayReal,
    operation_name: str,
    **params: Any
) -> "ChannelFrame":
    """
    Create new frame with updated metadata.
    
    Parameters
    ----------
    data : NDArrayReal
        Processed signal data.
    operation_name : str
        Name of the operation that was applied.
    **params : Any
        Parameters used in the operation.
        
    Returns
    -------
    ChannelFrame
        New frame with updated operation history.
    """
    # Extend operation history
    new_history = self.operation_history.copy()
    new_history.append(
        OperationRecord(
            name=operation_name,
            params=params,
            timestamp=datetime.now()
        )
    )
    
    # Create new frame
    return ChannelFrame(
        data=data,
        sampling_rate=self.sampling_rate,
        channel_metadata=self.channel_metadata,  # Preserved
        operation_history=new_history,
        previous=self  # Link to previous frame
    )
```

## Error Handling

See [Error Message Guide](./error_message_guide.md) for comprehensive guidance.

### Quick Reference

All errors must follow the 3-element rule:

1. **WHAT**: Clearly state the problem
2. **WHY**: Explain the constraint or requirement
3. **HOW**: Provide actionable solution

```python
# Template
raise ExceptionType(
    f"<WHAT: problem description>\n"
    f"  Got: {actual_value}\n"
    f"  Expected: {expected_value}\n"
    f"<HOW: specific actionable solution>"
)
```

## Documentation Standards

### Docstring Format

Use NumPy/Google style docstrings in English:

```python
def function_name(param1: type1, param2: type2) -> ReturnType:
    """
    Brief one-line description.
    
    Longer description providing context, algorithm details,
    and usage guidance. Can span multiple paragraphs.
    
    Parameters
    ----------
    param1 : type1
        Description of param1. Include units if applicable.
        Can span multiple lines.
    param2 : type2, default=value
        Description of param2.
        
    Returns
    -------
    ReturnType
        Description of return value.
        
    Raises
    ------
    ValueError
        When and why this error occurs.
    TypeError
        When and why this error occurs.
        
    See Also
    --------
    related_function : Brief description
    
    Notes
    -----
    Additional technical details, algorithm complexity,
    or implementation notes.
    
    Mathematical formulas in LaTeX:
    .. math:: E = mc^2
    
    Examples
    --------
    >>> result = function_name(arg1, arg2)
    >>> print(result)
    expected_output
    """
    ...
```

### Mathematical Notation

Use LaTeX for mathematical formulas:

```python
def compute_rms(signal: NDArrayReal) -> float:
    """
    Compute Root Mean Square of signal.
    
    The RMS is calculated as:
    
    .. math::
        \\text{RMS} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} x_i^2}
    
    where :math:`N` is the number of samples and :math:`x_i`
    are the signal values.
    
    Parameters
    ----------
    signal : NDArrayReal
        Input signal array.
        
    Returns
    -------
    float
        RMS value of the signal.
    """
    return np.sqrt(np.mean(signal ** 2))
```

## Performance Optimization

### Lazy Evaluation

Prefer returning Dask arrays without computing:

```python
# Good: Returns lazy Dask array
def process_large_signal(data: DaskArray) -> DaskArray:
    """Process signal with lazy evaluation."""
    # Chain operations without .compute()
    return da.fft.fft(data).apply(window_function)

# Bad: Forces computation
def process_large_signal(data: DaskArray) -> NDArrayReal:
    """Process signal (forces computation)."""
    result = da.fft.fft(data)
    return result.compute()  # Forces immediate computation!
```

### Vectorization

Use NumPy/Dask vectorized operations instead of Python loops:

```python
# Good: Vectorized
def normalize_channels(data: NDArrayReal) -> NDArrayReal:
    """Normalize each channel independently."""
    max_vals = np.max(np.abs(data), axis=-1, keepdims=True)
    return data / max_vals

# Bad: Python loop
def normalize_channels(data: NDArrayReal) -> NDArrayReal:
    """Normalize each channel independently."""
    result = np.zeros_like(data)
    for i in range(data.shape[0]):
        max_val = np.max(np.abs(data[i]))
        result[i] = data[i] / max_val
    return result
```

### Memory Efficiency

Consider memory usage for large arrays:

```python
import dask.array as da

def efficient_chunking(
    large_array: NDArrayReal,
    chunk_size: int = 10000
) -> DaskArray:
    """
    Convert large array to Dask with optimal chunking.
    
    Parameters
    ----------
    large_array : NDArrayReal
        Large NumPy array to process.
    chunk_size : int, default=10000
        Chunk size along last axis.
        
    Returns
    -------
    DaskArray
        Dask array with optimal chunking.
    """
    # Chunk along time axis (last dimension)
    chunks = tuple(s if i < large_array.ndim - 1 else chunk_size 
                   for i, s in enumerate(large_array.shape))
    return da.from_array(large_array, chunks=chunks)
```

## Code Style

### Formatting

Use Ruff for automatic formatting:

```bash
# Format code
uv run ruff format wandas tests

# Check without modifying
uv run ruff format --check wandas tests
```

### Linting

Use Ruff for linting:

```bash
# Lint and auto-fix
uv run ruff check wandas tests --fix

# Check without modifying
uv run ruff check wandas tests
```

### Import Organization

Organize imports in this order:

1. Standard library
2. Third-party packages
3. Local/project imports

```python
# Standard library
import os
from pathlib import Path
from typing import Optional, Union

# Third-party
import numpy as np
import dask.array as da
from scipy import signal

# Local
from wandas.utils.types import NDArrayReal
from wandas.processing.base import AudioOperation
```

### Naming Conventions

- **Classes**: PascalCase (`ChannelFrame`, `AudioOperation`)
- **Functions/methods**: snake_case (`read_wav`, `apply_filter`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_SAMPLING_RATE`)
- **Private members**: Leading underscore (`_process_array`)
- **Type variables**: PascalCase with T suffix (`InputT`, `OutputT`)

---

**Related Documentation**:
- [Error Message Guide](./error_message_guide.md) - Detailed error handling patterns
- [Testing Guide](./testing_guide.md) - Testing best practices
- [Copilot Instructions](../../.github/copilot-instructions.md) - Quick reference for AI
