# Roughness Calculation Implementation

This document describes the implementation of roughness calculation feature in wandas using MoSQITo.

## Overview

Roughness is a psychoacoustic metric that quantifies the perceived roughness of a sound, related to rapid amplitude modulations in the 15-300 Hz range. This implementation uses the Daniel & Weber method as provided by the MoSQITo library.

## Implementation Details

### New Files Created

1. **`wandas/processing/psychoacoustic.py`**
   - New module for psychoacoustic metrics
   - Contains `RoughnessDW` class implementing the roughness calculation
   - Supports both time-domain and frequency-domain methods
   - Follows the existing `AudioOperation` pattern

2. **`tests/processing/test_psychoacoustic_operations.py`**
   - Comprehensive unit tests for `RoughnessDW` operation
   - Tests for both time and frequency domain methods
   - Parameter validation tests
   - Comparison with direct MoSQITo calculations

3. **`tests/frames/test_channel_roughness.py`**
   - Integration tests for `ChannelFrame.roughness()` method
   - Tests for mono and stereo signals
   - Validation that modulated signals have higher roughness than pure tones
   - Tests with different sampling rates

4. **`examples/roughness_example.py`**
   - Standalone example demonstrating roughness calculation
   - Shows comparison between different signal types
   - Demonstrates usage with different parameters

5. **`docs/src/tutorial/roughness.md`** (English)
   - Comprehensive user documentation
   - Usage examples and API reference
   - Explanation of roughness values and perceptual characteristics

6. **`docs/src/tutorial/roughness.ja.md`** (Japanese)
   - Japanese translation of the tutorial documentation

### Modified Files

1. **`wandas/processing/__init__.py`**
   - Added import and export of `RoughnessDW`
   - Added to `__all__` list

2. **`wandas/frames/mixins/channel_transform_mixin.py`**
   - Added `roughness()` method to ChannelTransformMixin
   - Provides easy access to roughness calculation from ChannelFrame

3. **`docs/src/api/processing.md`** and **`docs/src/api/processing.en.md`**
   - Added section for psychoacoustic metrics

## API

### Operation Class: `RoughnessDW`

```python
from wandas.processing import RoughnessDW

operation = RoughnessDW(
    sampling_rate=44100,
    method='time',  # or 'freq'
    overlap=0.0     # 0.0 to 1.0 for time method
)
```

### ChannelFrame Method: `roughness()`

```python
import wandas as wd

signal = wd.read_wav("audio.wav")
roughness = signal.roughness(method='time', overlap=0.0)
# Returns: NDArrayReal with shape (n_channels,)
```

## Key Features

1. **Two Calculation Methods**
   - Time-domain: `method='time'` using `roughness_dw_time` from MoSQITo
   - Frequency-domain: `method='freq'` using `roughness_dw_freq` from MoSQITo

2. **Multi-channel Support**
   - Calculates roughness independently for each channel
   - Returns array with one value per channel

3. **Configurable Parameters**
   - `overlap`: Overlap ratio for time-domain method (0.0 to 1.0)
   - `method`: Choice between 'time' and 'freq' domain calculation

4. **Validation**
   - Parameter validation with clear error messages
   - Input shape validation

## Test Coverage

The implementation includes comprehensive test coverage:

### Unit Tests (`test_psychoacoustic_operations.py`)
- Initialization tests
- Parameter validation
- Output shape calculation
- Both time and frequency methods
- Mono and stereo signals
- Comparison with direct MoSQITo calculations
- Edge cases (pure tone, modulated signals)

### Integration Tests (`test_channel_roughness.py`)
- ChannelFrame method availability
- Mono and stereo signal processing
- Parameter validation through the API
- Comparison between pure tones and modulated signals
- Validation against direct MoSQITo usage
- Different sampling rates

## Usage Examples

### Basic Usage

```python
import wandas as wd

# Load audio file
signal = wd.read_wav("audio.wav")

# Calculate roughness
roughness = signal.roughness(method='time')
print(f"Roughness: {roughness:.3f} asper")
```

### Comparing Signals

```python
import numpy as np
import wandas as wd

# Create pure tone (low roughness)
t = np.linspace(0, 1, 44100)
pure_tone = np.sin(2 * np.pi * 1000 * t)
signal_pure = wd.ChannelFrame(data=pure_tone, sampling_rate=44100)

# Create modulated tone (high roughness)
carrier = np.sin(2 * np.pi * 1000 * t)
modulator = 0.5 * (1 + np.sin(2 * np.pi * 70 * t))
modulated = carrier * modulator
signal_mod = wd.ChannelFrame(data=modulated, sampling_rate=44100)

# Compare roughness
print(f"Pure tone: {signal_pure.roughness():.3f} asper")
print(f"Modulated: {signal_mod.roughness():.3f} asper")
```

## Technical Details

### MoSQITo Integration

The implementation leverages MoSQITo's roughness calculation functions:
- `mosqito.sq_metrics.roughness_dw_time`: Time-domain calculation
- `mosqito.sq_metrics.roughness_dw_freq`: Frequency-domain calculation

### Processing Pipeline

1. **Time-domain method**:
   - Calls `roughness_dw_time` with signal and sampling rate
   - Extracts mean roughness value from result
   - Processes each channel independently

2. **Frequency-domain method**:
   - Computes FFT of signal
   - Calculates magnitude spectrum
   - Calls `roughness_dw_freq` with spectrum and frequencies
   - Extracts roughness value from result

### Output Format

- Single channel: Scalar value (0-dimensional or 1-element array)
- Multi-channel: 1D array with shape `(n_channels,)`
- Unit: asper (psychoacoustic unit for roughness)

## Design Principles Followed

1. **Pandas-like Interface**: The `roughness()` method integrates naturally with the existing ChannelFrame API
2. **Type Safety**: Full type hints throughout the implementation
3. **Chain Methods**: Compatible with method chaining in wandas
4. **Testability**: Comprehensive test coverage with clear assertions
5. **Documentation**: Extensive docstrings following NumPy/Google format
6. **Error Handling**: Clear error messages for invalid parameters

## Future Enhancements

Potential improvements for future versions:

1. **Time-varying Roughness**: Return time series of roughness values
2. **Additional Methods**: Support for other roughness calculation methods (e.g., Aures method)
3. **Visualization**: Built-in plotting functions for roughness analysis
4. **Batch Processing**: Efficient processing of multiple files
5. **Integration with other SQ metrics**: Combine with loudness, sharpness, etc.

## References

1. Daniel, P., & Weber, R. (1997). Psychoacoustical roughness: implementation of an optimized model. *Acustica*, 83, 113-123.

2. ECMA-418-2:2022 - Psychoacoustic metrics for ITT equipment - Part 2: Models based on human perception

3. MoSQITo Documentation: https://mosqito.readthedocs.io

4. MoSQITo GitHub: https://github.com/Eomys/MoSQITo
