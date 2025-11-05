---
applyTo: 'wandas/**/*.py'
---

# Python Code Standards for Wandas

## Type Safety

- All functions require type hints (parameters and return values)
- Use type aliases from `wandas.utils.types` (`NDArrayReal`, `NDArrayComplex`, `DaskArray`)
- Comply with mypy strict mode
- Use `Optional[T]` for nullable types, `Union[T1, T2]` for alternatives

## Signal Processing Patterns

### AudioOperation Base Class

Extend `AudioOperation[InputType, OutputType]` for new operations:

```python
from wandas.processing.base import AudioOperation, register_operation

@register_operation
class MyFilter(AudioOperation[NDArrayReal, NDArrayReal]):
    name = "my_filter"

    def __init__(self, sampling_rate: float, cutoff: float) -> None:
        # Validate parameters
        if cutoff >= sampling_rate / 2:
            raise ValueError(
                f"Cutoff frequency out of range\n"
                f"  Got: {cutoff} Hz\n"
                f"  Expected: < {sampling_rate / 2} Hz (Nyquist)\n"
                f"Use cutoff < sampling_rate / 2"
            )
        super().__init__(sampling_rate, cutoff=cutoff)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        # Implementation
        ...
```

### Key Principles

- **Immutability**: Return new frames, never modify input
- **Metadata**: Track operations in `operation_history`
- **Lazy Evaluation**: Use Dask arrays where possible
- **Validation**: Check parameters in `__init__`

## Error Handling

Use 3-element pattern (WHAT/WHY/HOW):

```python
raise ValueError(
    f"<WHAT: Problem description>\n"
    f"  Got: {actual_value}\n"
    f"  Expected: {expected_value}\n"
    f"<HOW: Actionable solution>"
)
```

Show actual vs expected values. Provide specific solutions, not generic advice.

## Docstrings

Use NumPy format, English only:

```python
def process(self, cutoff: float, order: int = 5) -> "ChannelFrame":
    """
    Apply low-pass filter to signal.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz. Must be less than Nyquist frequency.
    order : int, default=5
        Filter order. Higher values give steeper roll-off.

    Returns
    -------
    ChannelFrame
        Filtered signal with operation recorded in metadata.

    Raises
    ------
    ValueError
        If cutoff >= Nyquist frequency or order < 1.

    Examples
    --------
    >>> signal = wd.read_wav("audio.wav")
    >>> filtered = signal.low_pass_filter(cutoff=1000)
    ```
```

Include: Parameters, Returns, Raises (with conditions), Examples (working code).

## Method Chaining

Enable fluent API by using `apply_operation`:

```python
def normalize(
    self: T_Processing,
    norm: float = float("inf"),
    axis: int = -1
) -> T_Processing:
    """Normalize signal amplitude.

    Parameters
    ----------
    norm : float, default=inf
        Norm type for normalization.
    axis : int, default=-1
        Axis along which to normalize.

    Returns
    -------
    T_Processing
        New frame with normalized signal.
    """
    logger.debug(f"Setting up normalize: norm={norm}, axis={axis} (lazy)")
    result = self.apply_operation("normalize", norm=norm, axis=axis)
    return cast(T_Processing, result)
```

## Performance

- Use vectorized NumPy operations
- Avoid explicit loops where possible
- Use Dask for large arrays (lazy evaluation)
- Don't call `.compute()` unless necessary
