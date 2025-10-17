# Implementation Plan: Add Loudness for Non-Stationary Signals using MoSQITo

## Background
Implement support for calculating loudness for non-stationary signals using the MoSQITo library's `loudness_zwtv` function, which implements the Zwicker method according to ISO 532-1:2017.

## Objective
Add the ability to calculate time-varying loudness (non-stationary) for audio signals in the wandas library, following the existing patterns for signal processing operations.

## Affected Files and Modules

### New Files
1. `wandas/processing/psychoacoustic.py` - New module for psychoacoustic metrics
2. `tests/processing/test_psychoacoustic_operations.py` - Tests for psychoacoustic operations

### Modified Files
1. `wandas/processing/__init__.py` - Export new loudness operation
2. `wandas/frames/mixins/channel_processing.py` - Add loudness method to ChannelFrame

## Implementation Details

### 1. Create Psychoacoustic Processing Module
Create `wandas/processing/psychoacoustic.py`:
- Implement `LoudnessZwtv` class inheriting from `AudioOperation`
- The operation will:
  - Accept a signal and sampling rate
  - Support `field_type` parameter ('free' or 'diffuse')
  - Return time-varying loudness values
  - Follow the existing wandas pattern for operations

### 2. MoSQITo API Integration
The MoSQITo `loudness_zwtv` function signature:
```python
def loudness_zwtv(signal, fs, field_type="free"):
    """
    Returns:
    - N: time-varying loudness (sones)
    - N_spec: specific loudness over critical band rate scale
    - bark_axis: Bark scale array
    - time_axis: time axis for loudness measurements
    """
```

### 3. Operation Implementation Strategy
- Input: 1D or 2D array (channels × samples)
- Processing: Apply `loudness_zwtv` per channel
- Output: Return loudness values with time axis
- The operation will need special handling since it returns multiple outputs (loudness, time_axis)

### 4. Testing Strategy
Based on MoSQITo test cases:
- Test with known signals (sine waves, noise)
- Compare values directly with MoSQITo's expected outputs
- Test both 'free' and 'diffuse' field types
- Verify shape preservation
- Test operation history recording
- Test with single and multi-channel data

### 5. Integration with ChannelFrame
Add a convenience method to ChannelFrame:
```python
def loudness_zwtv(self, field_type: str = "free") -> tuple[NDArrayReal, NDArrayReal]:
    """Calculate time-varying loudness using Zwicker method."""
```

## Risks and Mitigation

### Risk 1: Output Format Mismatch
The loudness operation returns multiple arrays (N, N_spec, bark_axis, time_axis) which differs from typical operations.
- **Mitigation**: Return a tuple of arrays or create a custom LoudnessFrame class to hold the results

### Risk 2: MoSQITo Dependency
MoSQITo is already a dependency, but we need to ensure compatibility.
- **Mitigation**: Test with the current MoSQITo version and document version requirements

### Risk 3: Performance
Loudness calculation may be computationally expensive for large signals.
- **Mitigation**: Use dask delayed execution where possible; document performance expectations

## Backward Compatibility
This is a new feature addition, so there are no backward compatibility concerns.

## Documentation Updates
1. Add docstrings following NumPy/Google style guide in English
2. Add examples to the method docstrings
3. Consider adding a tutorial/example notebook showing loudness calculation
4. Update API reference documentation if needed

## Implementation Steps
1. Create `wandas/processing/psychoacoustic.py` with `LoudnessZwtv` operation
2. Register the operation in `wandas/processing/__init__.py`
3. Add method to ChannelFrame for easy access
4. Write comprehensive tests comparing with MoSQITo direct usage
5. Run linters and type checkers
6. Test with real audio data
7. Update documentation

## Success Criteria
- [ ] All tests pass with 100% coverage for new code
- [ ] Type checking passes (mypy strict mode)
- [ ] Linting passes (ruff)
- [ ] Values match MoSQITo direct calculations
- [ ] Works with both single and multi-channel data
- [ ] Proper operation history tracking
- [ ] Clear documentation and examples
