# Speech Intelligibility Index (SII) Implementation Summary

## Overview
This implementation integrates MoSQITo's Speech Intelligibility Index (SII) computation into the wandas library, providing a standardized measure of speech intelligibility according to ANSI S3.5-1997.

## What is SII?
The Speech Intelligibility Index (SII) is a standardized metric that:
- Measures speech intelligibility on a scale from 0.0 (completely unintelligible) to 1.0 (perfectly intelligible)
- Follows the ANSI S3.5-1997 standard
- Takes into account the frequency-dependent nature of speech perception
- Is widely used in acoustics, hearing aid design, and audio quality assessment

## Implementation Details

### 1. Core Operation: `SpeechIntelligibilityIndex`
Located in: `wandas/processing/stats.py`

A new processing operation that wraps MoSQITo's `comp_sii` function:

```python
from wandas.processing import SpeechIntelligibilityIndex

sii_op = SpeechIntelligibilityIndex(
    sampling_rate=44100,
    method_band="octave"  # or "third octave", "critical"
)
```

Features:
- Supports three frequency band methods:
  - **octave**: Faster computation, fewer frequency bands
  - **third octave**: Balanced approach, standard for many applications
  - **critical**: Most detailed, uses psychoacoustic critical bands
- Automatically converts multi-channel signals to mono
- Validates parameters at initialization
- Integrates with wandas' operation registry

### 2. ChannelFrame Method: `sii()`
Located in: `wandas/frames/mixins/channel_processing_mixin.py`

Convenient method for direct SII calculation on any ChannelFrame:

```python
import wandas as wd

# Load or generate signal
signal = wd.read_wav("speech.wav")

# Calculate SII
sii_value, specific_sii, freq_axis = signal.sii(method_band="octave")

print(f"Speech Intelligibility Index: {sii_value:.3f}")
```

Returns:
- `sii_value` (float): Overall SII from 0.0 to 1.0
- `specific_sii` (ndarray): Frequency-specific SII values for each band
- `freq_axis` (ndarray): Corresponding frequency values in Hz

### 3. Comprehensive Test Suite

#### Operation Tests (`tests/processing/test_stats_operations.py`)
- `TestSpeechIntelligibilityIndex` class with 10 test methods:
  - Initialization with different method bands
  - Invalid parameter handling
  - Output shape validation
  - SII value range validation (0.0 to 1.0)
  - Multi-channel signal handling
  - Direct comparison with MoSQITo computation
  - Different method band options
  - Operation registry verification

#### Integration Tests (`tests/frames/test_channel_sii.py`)
- `TestChannelFrameSII` class with 13 test methods:
  - Method existence and callability
  - Return value structure and types
  - Default and custom method bands
  - Multi-channel handling
  - Direct comparison with MoSQITo
  - ANSI test case validation
  - Frequency axis ordering
  - Integration with filtering operations
  - Integration with normalization

### 4. Example Notebook
Located in: `examples/speech_intelligibility_index.ipynb`

Demonstrates:
- Basic SII calculation
- Visualization of frequency-specific SII
- Comparison of different method bands
- Effect of filtering on speech intelligibility
- Effect of noise on SII values
- Working with real audio files

## Usage Examples

### Basic Usage
```python
import wandas as wd

# Generate or load a signal
signal = wd.generate_sin(
    freqs=[250, 500, 1000, 2000],
    duration=1.0,
    sampling_rate=44100
)

# Calculate SII
sii_value, specific_sii, freq_axis = signal.sii(method_band="octave")
print(f"SII: {sii_value:.3f}")
```

### With Signal Processing
```python
# Apply filtering and check impact on intelligibility
filtered = signal.low_pass_filter(cutoff=4000)
sii_filtered, _, _ = filtered.sii()

print(f"Original SII: {signal.sii()[0]:.3f}")
print(f"Filtered SII: {sii_filtered:.3f}")
```

### Analyzing Real Audio
```python
# Load audio file
audio = wd.read_wav("recording.wav")

# Calculate SII with different method bands
for method in ["octave", "third octave", "critical"]:
    sii, _, _ = audio.sii(method_band=method)
    print(f"{method:15s}: {sii:.3f}")
```

## Technical Notes

### Multi-Channel Handling
Multi-channel signals are automatically converted to mono by averaging across channels before SII computation. This is required because SII is defined for single-channel signals.

### Computational Considerations
- **Immediate computation**: Unlike most wandas operations that use lazy evaluation, SII requires the full signal and computes immediately
- **Method band selection**: 
  - Use "octave" for faster computation and when broad frequency information suffices
  - Use "third octave" for standard analysis (common in acoustic standards)
  - Use "critical" for most detailed psychoacoustic analysis

### Integration with MoSQITo
The implementation directly uses MoSQITo's `comp_sii` function, ensuring:
- ANSI S3.5-1997 compliance
- Well-tested and validated computation
- Consistency with other MoSQITo-based operations in wandas

## Testing and Validation

All tests validate against:
1. **Value ranges**: SII must be between 0.0 and 1.0
2. **Direct MoSQITo comparison**: Results match direct MoSQITo calls
3. **Type correctness**: Return values have expected types and shapes
4. **Edge cases**: Multi-channel, different method bands, filtered signals
5. **Integration**: Works correctly with other wandas operations

To run tests:
```bash
# Run all SII-related tests
pytest tests/processing/test_stats_operations.py::TestSpeechIntelligibilityIndex -v
pytest tests/frames/test_channel_sii.py -v

# Run with coverage
pytest --cov=wandas.processing.stats --cov=wandas.frames.mixins tests/
```

## References

1. ANSI S3.5-1997: *American National Standard Methods for Calculation of the Speech Intelligibility Index*
2. MoSQITo library: https://mosqito.readthedocs.io/
3. MoSQITo SII documentation: https://mosqito.readthedocs.io/en/latest/source/reference/mosqito.sq_metrics.speech_intelligibility.sii_ansi.sii_ansi.html

## Future Enhancements

Potential improvements for future versions:
1. Support for speech-weighted noise (currently uses signal directly)
2. Custom importance function weights for specific applications
3. Batch processing for multiple signals
4. Visualization utilities for frequency-specific SII
5. Integration with other speech quality metrics

## Conclusion

This implementation provides wandas users with a robust, standards-compliant method for assessing speech intelligibility, seamlessly integrated with the library's existing signal processing capabilities.
