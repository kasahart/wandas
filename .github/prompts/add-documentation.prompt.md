# Add API Documentation

Generate NumPy-style docstring following Wandas standards.

## Required Sections

1. **Brief description** (one line)
2. **Extended description** (optional, for complex operations)
3. **Parameters** (all arguments with types and descriptions)
4. **Returns** (type and description)
5. **Raises** (all possible exceptions with conditions)
6. **Examples** (working code snippets)

## Template

```python
def function_name(
    param1: Type1,
    param2: Type2,
    optional_param: Type3 = default_value
) -> ReturnType:
    """
    Brief one-line description of what the function does.

    Extended description providing more context, mathematical background,
    or usage guidelines. This section is optional but recommended for
    complex operations.

    Parameters
    ----------
    param1 : Type1
        Description of param1. Explain what it represents, valid ranges,
        and any constraints.
    param2 : Type2
        Description of param2.
    optional_param : Type3, default=<default_value>
        Description of optional parameter. Explain default behavior.

    Returns
    -------
    ReturnType
        Description of return value. Explain what it contains and
        how it relates to input.

    Raises
    ------
    ValueError
        If <specific condition that raises ValueError>.
    TypeError
        If <specific condition that raises TypeError>.

    See Also
    --------
    related_function : Related functionality
    another_function : Alternative approach

    Notes
    -----
    Additional information about implementation details, performance
    characteristics, or theoretical background.

    Mathematical formulas can be included using LaTeX:

    .. math::
        X(f) = \int_{-\infty}^{\infty} x(t) e^{-j2\pi ft} dt

    Examples
    --------
    Basic usage:

    >>> signal = wd.read_wav("audio.wav")
    >>> result = signal.function_name(param1, param2)
    >>> print(result.shape)
    (44100, 2)

    Advanced usage with optional parameters:

    >>> result = signal.function_name(
    ...     param1=value1,
    ...     param2=value2,
    ...     optional_param=custom_value
    ... )

    Method chaining:

    >>> result = (
    ...     signal
    ...     .function_name(param1, param2)
    ...     .another_operation()
    ... )
    """
    # Implementation
    ...
```

## Style Guidelines

### Language
- **English only** for all docstrings
- Clear, concise technical writing
- Use active voice ("Applies filter" not "Filter is applied")

### Types
- Use full type names in Parameters section
- Show default values explicitly
- Indicate optional with `optional` or `default=value`

### Descriptions
- Start parameter descriptions with capital letter
- End with period
- Explain constraints and valid ranges
- Include units (Hz, seconds, dB, etc.)

### Examples
- Show working code that can be copy-pasted
- Include expected outputs where helpful
- Demonstrate both basic and advanced usage
- Use `>>>` for Python prompt, `...` for continuation

## Common Patterns

### Signal Processing Functions

```python
def low_pass_filter(
    self,
    cutoff: float,
    order: int = 5
) -> "ChannelFrame":
    """
    Apply low-pass Butterworth filter to signal.

    Filters out frequency components above the cutoff frequency
    using a Butterworth filter design. Higher order values provide
    steeper roll-off but may introduce phase distortion.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz. Must be less than Nyquist frequency
        (sampling_rate / 2).
    order : int, default=5
        Filter order. Higher values give steeper roll-off.
        Must be positive integer.

    Returns
    -------
    ChannelFrame
        Filtered signal with operation recorded in metadata.
        Shape and sampling rate are preserved.

    Raises
    ------
    ValueError
        If cutoff >= Nyquist frequency or order < 1.

    Examples
    --------
    >>> signal = wd.read_wav("audio.wav")
    >>> filtered = signal.low_pass_filter(cutoff=1000)
    >>> print(f"Filtered at {filtered.sampling_rate} Hz")
    ```
"""
```

### Analysis Functions

```python
def compute_spectrum(
    self,
    n_fft: Optional[int] = None,
    window: str = "hann"
) -> "SpectralFrame":
    """
    Compute frequency spectrum using FFT.

    Applies windowing to reduce spectral leakage, then computes
    the Fast Fourier Transform. Returns magnitude spectrum in dB.

    Parameters
    ----------
    n_fft : int, optional
        FFT size. If None, uses next power of 2 >= signal length.
        Power of 2 values optimize performance.
    window : str, default="hann"
        Window function to apply. Supported: 'hann', 'hamming',
        'blackman', 'bartlett', 'boxcar'.

    Returns
    -------
    SpectralFrame
        Frequency-domain representation with magnitude in dB.
        Contains frequencies from 0 to Nyquist.

    Raises
    ------
    ValueError
        If window type is not supported or n_fft < signal length.

    See Also
    --------
    stft : Short-time Fourier transform for time-varying analysis

    Notes
    -----
    The window function reduces spectral leakage caused by
    finite signal duration. Hann window provides good balance
    between main lobe width and side lobe attenuation.

    Examples
    --------
    >>> signal = wd.generate_sin(440, duration=1.0)
    >>> spectrum = signal.compute_spectrum()
    >>> peak_freq = spectrum.freqs[np.argmax(spectrum.data)]
    >>> print(f"Peak at {peak_freq} Hz")
    Peak at 440.0 Hz
    ```
"""
```

### I/O Functions

```python
def read_wav(
    filepath: Union[str, Path],
    sampling_rate: Optional[float] = None
) -> ChannelFrame:
    """
    Read WAV file and create ChannelFrame.

    Loads audio data from WAV file format. Supports mono and
    multi-channel audio with various bit depths.

    Parameters
    ----------
    filepath : str or Path
        Path to WAV file to read. Can be relative or absolute.
    sampling_rate : float, optional
        Expected sampling rate in Hz. If provided and doesn't match
        file, raises InvalidSamplingRateError. If None, accepts
        any sampling rate.

    Returns
    -------
    ChannelFrame
        Audio data from WAV file with metadata.

    Raises
    ------
    FileNotFoundError
        If file doesn't exist at specified path.
    InvalidSamplingRateError
        If sampling_rate is specified and doesn't match file.
    ValueError
        If file format is invalid or unsupported.

    Examples
    --------
    >>> signal = wd.read_wav("audio.wav")
    >>> print(f"{signal.duration:.2f}s at {signal.sampling_rate} Hz")

    Validate sampling rate:

    >>> signal = wd.read_wav("audio.wav", sampling_rate=44100)
    ```
"""
```

## Checklist

- [ ] Brief description (one line)
- [ ] All parameters documented with types
- [ ] Return value documented with type
- [ ] All exceptions documented with conditions
- [ ] At least one working example
- [ ] English language throughout
- [ ] Units specified where applicable
- [ ] Default values shown for optional parameters

## Testing Documentation

Ensure docstring examples work:

```python
def test_docstring_example():
    """Test that docstring example executes correctly."""
    # Copy example from docstring
    signal = wd.read_wav("test_audio.wav")
    result = signal.function_name(param1, param2)
    assert result.shape == (44100, 2)
```

Consider using `doctest`:

```bash
uv run python -m doctest wandas/module.py -v
```
