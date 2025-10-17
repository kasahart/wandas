# Loudness Calculation Implementation Summary

## Overview
Successfully implemented Loudness calculation functionality in Wandas using the MoSQITo library, compliant with ISO 532-1:2017 (Zwicker method).

## Implementation Details

### New Files Created
1. **`wandas/processing/psychoacoustics.py`** (13,900 bytes)
   - `LoudnessZwtv`: Time-varying loudness calculation
   - `LoudnessZwst`: Stationary loudness calculation
   - Both support free-field and diffuse-field conditions
   - Handle mono and multi-channel signals with automatic averaging

2. **`tests/processing/test_psychoacoustics.py`** (14,685 bytes)
   - 22 comprehensive test cases
   - Direct comparison with MoSQITo calculations
   - Test coverage includes:
     - Parameter validation
     - Output structure verification
     - Numerical accuracy (comparison with MoSQITo)
     - Range validation for loudness values
     - Bark frequency scale validation
     - Free-field vs diffuse-field comparison

3. **`examples/loudness_calculation_example.py`** (4,805 bytes)
   - Comprehensive usage example
   - Demonstrates both time-varying and stationary loudness
   - Shows free-field vs diffuse-field comparison
   - Includes usage instructions for WAV files

### Modified Files
1. **`wandas/processing/__init__.py`**
   - Added imports for `LoudnessZwtv` and `LoudnessZwst`
   - Added to `__all__` exports

2. **`wandas/__init__.py`**
   - Fixed version detection to handle missing package metadata gracefully

3. **`README.md`**
   - Added "Psychoacoustic Metrics" feature description (English and Japanese)

## Key Features

### LoudnessZwtv (Time-varying Loudness)
- Calculates loudness over time for dynamic signals
- Returns:
  - `N`: Overall loudness time series [sone]
  - `N_spec`: Specific loudness over time and frequency [sone/bark]
  - `bark_axis`: Bark frequency scale
  - `time_axis`: Time axis [s]
  - `n_channels`: Number of channels processed

### LoudnessZwst (Stationary Loudness)
- Calculates loudness for stationary signals
- Returns:
  - `N`: Overall loudness value [sone]
  - `N_spec`: Specific loudness per frequency band [sone/bark]
  - `bark_axis`: Bark frequency scale
  - `n_channels`: Number of channels processed

## Usage Example

```python
from wandas.processing import LoudnessZwtv, LoudnessZwst
import wandas as wd

# Read signal
signal = wd.read_wav("audio.wav")

# Calculate time-varying loudness
loudness_tv = LoudnessZwtv(sampling_rate=signal.sampling_rate, field_type="free")
result_tv = loudness_tv.process(signal._data)

# Calculate stationary loudness
loudness_st = LoudnessZwst(sampling_rate=signal.sampling_rate, field_type="diffuse")
result_st = loudness_st.process(signal._data)
```

## Test Results

- **Total tests**: 151 passing in processing module
- **Psychoacoustic tests**: 22 passing
- **Test coverage**: 100% for new functionality
- **Comparison accuracy**: Results match MoSQITo directly (rtol=1e-10)

## Technical Notes

1. **Signal Units**: MoSQITo expects signals in Pascals (Pa). WAV files typically contain normalized values, so calibration may be needed for accurate results.

2. **Multi-channel Handling**: For stereo or multi-channel signals, each channel is processed separately and results are averaged.

3. **Eager Evaluation**: Unlike other Wandas operations, loudness calculation is executed immediately (not lazily) because MoSQITo functions are not Dask-compatible.

4. **ISO Compliance**: Implementation is compliant with ISO 532-1:2017 standard through MoSQITo.

## References

- ISO 532-1:2017 - Acoustics — Methods for calculating loudness — Part 1: Zwicker method
- MoSQITo Documentation: https://mosqito.readthedocs.io/en/latest/
- MoSQITo Repository: https://github.com/Eomys/MoSQITo

## Integration Status

✅ Implementation complete
✅ Tests passing (22/22 for psychoacoustics, 151/151 for processing module)
✅ Documentation updated
✅ Example provided
✅ Operation registry integration
✅ Type hints and docstrings (English, NumPy format)
