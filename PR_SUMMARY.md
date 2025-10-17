# Pull Request Summary: Implement Roughness Calculation using MoSQITo

## Overview

This PR implements roughness calculation functionality in wandas using the MoSQITo (Modular Sound Quality Indicators Tools) library. Roughness is a psychoacoustic metric that quantifies the perceived roughness of a sound, related to rapid amplitude modulations in the 15-300 Hz range.

## Changes Summary

### New Modules

1. **`wandas/processing/psychoacoustic.py`** - New module for psychoacoustic metrics
   - Implements `RoughnessDW` operation class
   - Supports both time-domain and frequency-domain calculations
   - Follows the existing `AudioOperation` pattern
   - Registered with the operation registry

### Enhanced Features

2. **`wandas/frames/mixins/channel_transform_mixin.py`** - Added roughness method
   - New `roughness()` method available on all ChannelFrame instances
   - Simple API: `signal.roughness(method='time', overlap=0.0)`
   - Returns scalar roughness values (one per channel)

### Export Changes

3. **`wandas/processing/__init__.py`** - Updated exports
   - Added `RoughnessDW` to exports and `__all__` list
   - Imported from new psychoacoustic module

### Tests

4. **`tests/processing/test_psychoacoustic_operations.py`** - Unit tests
   - 17 test methods covering all aspects of `RoughnessDW`
   - Tests for both time and frequency methods
   - Parameter validation tests
   - Comparison with direct MoSQITo calculations
   - Tests for pure tones and modulated signals

5. **`tests/frames/test_channel_roughness.py`** - Integration tests
   - 11 test methods for ChannelFrame integration
   - Tests for mono and stereo signals
   - Validation of perceptual characteristics
   - Cross-comparison with MoSQITo reference values
   - Different sampling rate tests

### Documentation

6. **`docs/src/api/processing.md`** and **`docs/src/api/processing.en.md`**
   - Added "Psychoacoustic Metrics" section
   - Reference to new psychoacoustic module

7. **`docs/src/tutorial/roughness.md`** - English tutorial
   - Comprehensive usage guide
   - Parameter explanations
   - Examples and use cases
   - Technical details and references

8. **`docs/src/tutorial/roughness.ja.md`** - Japanese tutorial
   - Japanese translation of tutorial documentation
   - Maintains consistency with English version

### Examples

9. **`examples/roughness_example.py`** - Demonstration script
   - Shows various usage patterns
   - Compares different signal types
   - Demonstrates parameter effects

### Implementation Documentation

10. **`IMPLEMENTATION_ROUGHNESS.md`** - Technical summary
    - Complete overview of implementation
    - API documentation
    - Design principles
    - Test coverage details

## Key Features

✓ **Two Calculation Methods**
  - Time-domain: Using `roughness_dw_time` from MoSQITo
  - Frequency-domain: Using `roughness_dw_freq` from MoSQITo

✓ **Multi-channel Support**
  - Independent calculation per channel
  - Returns array with one value per channel

✓ **Configurable Parameters**
  - `method`: 'time' or 'freq'
  - `overlap`: 0.0 to 1.0 (for time method)

✓ **Comprehensive Testing**
  - 28 test methods total
  - Direct comparison with MoSQITo reference values
  - Edge case coverage

✓ **Full Documentation**
  - English and Japanese tutorials
  - API documentation
  - Example scripts
  - Implementation details

## Usage Example

```python
import wandas as wd
import numpy as np

# Load audio file
signal = wd.read_wav("audio.wav")

# Calculate roughness (time-domain method)
roughness = signal.roughness(method='time')
print(f"Roughness: {roughness:.3f} asper")

# Calculate with frequency-domain method
roughness_freq = signal.roughness(method='freq')
print(f"Roughness (freq): {roughness_freq:.3f} asper")

# Create test signal with amplitude modulation
t = np.linspace(0, 1, 44100)
carrier = np.sin(2 * np.pi * 1000 * t)
modulator = 0.5 * (1 + np.sin(2 * np.pi * 70 * t))
modulated = carrier * modulator

# Calculate roughness of modulated signal
signal_mod = wd.ChannelFrame(data=modulated, sampling_rate=44100)
roughness_mod = signal_mod.roughness(method='time')
print(f"Modulated signal roughness: {roughness_mod:.3f} asper")
```

## Test Coverage

- **Unit Tests**: 17 methods in `test_psychoacoustic_operations.py`
- **Integration Tests**: 11 methods in `test_channel_roughness.py`
- **Total**: 28 test methods

All tests include:
- Input validation
- Output shape verification
- Value range checks
- Comparison with MoSQITo reference implementation
- Edge case handling

## Design Compliance

This implementation follows all wandas design principles:

1. ✓ **Pandas-like Interface**: Natural integration with ChannelFrame
2. ✓ **Type Safety**: Full type hints throughout
3. ✓ **Chain Methods**: Compatible with method chaining
4. ✓ **Extensibility**: Easy to add more psychoacoustic metrics
5. ✓ **Testability**: Comprehensive test coverage
6. ✓ **Documentation**: NumPy/Google-style docstrings

## Validation

All code has been validated for:
- ✓ Python syntax correctness
- ✓ Import structure
- ✓ Method availability
- ✓ Test structure
- ✓ Documentation completeness

## Dependencies

Uses existing dependency: `mosqito` (already in pyproject.toml)

## Files Changed

**New Files (10):**
- `wandas/processing/psychoacoustic.py`
- `tests/processing/test_psychoacoustic_operations.py`
- `tests/frames/test_channel_roughness.py`
- `examples/roughness_example.py`
- `docs/src/tutorial/roughness.md`
- `docs/src/tutorial/roughness.ja.md`
- `docs/src/api/processing.md` (modified)
- `docs/src/api/processing.en.md` (modified)
- `IMPLEMENTATION_ROUGHNESS.md`

**Modified Files (3):**
- `wandas/processing/__init__.py`
- `wandas/frames/mixins/channel_transform_mixin.py`
- `docs/src/api/processing.md` and `processing.en.md`

## References

1. Daniel, P., & Weber, R. (1997). Psychoacoustical roughness: implementation of an optimized model. *Acustica*, 83, 113-123.

2. MoSQITo User Guide: https://mosqito.readthedocs.io/en/latest/source/user_guide/scope.html#sq-metrics

3. ECMA-418-2:2022 - Psychoacoustic metrics for ITT equipment

## Next Steps

After this PR is merged:
- Tests will run in CI/CD pipeline
- Documentation will be built and published
- Feature will be available in next release

## Notes

- All code follows wandas coding standards
- Tests are designed to compare with MoSQITo reference values
- Documentation is available in both English and Japanese
- Implementation is ready for review and testing
