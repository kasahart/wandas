# Error Message Analysis Report

## Executive Summary

This document provides a comprehensive analysis of all error messages in the Wandas codebase as of Phase 1 of the error message improvement initiative. The analysis covers 100 error instances across 19 files, categorized by error type and prioritized for improvement.

**Date**: 2024-10-22  
**Codebase Version**: 0.1.7  
**Total Errors Analyzed**: 100

## Error Distribution

### By Error Type

| Error Type | Count | Percentage | Priority |
|-----------|-------|------------|----------|
| ValueError | 60 | 60% | High |
| TypeError | 16 | 16% | High |
| NotImplementedError | 13 | 13% | Medium |
| IndexError | 5 | 5% | High |
| FileNotFoundError | 3 | 3% | High |
| KeyError | 2 | 2% | Medium |
| FileExistsError | 1 | 1% | Low |

### By Module

| Module | Count | Priority | Notes |
|--------|-------|----------|-------|
| wandas/frames/channel.py | 23 | High | Core user-facing API |
| wandas/core/base_frame.py | 13 | High | Base functionality |
| wandas/visualization/plotting.py | 7 | Medium | Visualization errors |
| wandas/utils/frame_dataset.py | 7 | Medium | Dataset operations |
| wandas/frames/roughness.py | 7 | Medium | Psychoacoustic ops |
| wandas/processing/filters.py | 5 | High | Filter operations |
| wandas/processing/base.py | 5 | Medium | Processing base |
| wandas/io/wdf_io.py | 5 | High | File I/O |
| wandas/frames/spectrogram.py | 4 | Medium | Spectrogram ops |
| wandas/io/readers.py | 4 | High | File readers |
| wandas/frames/spectral.py | 3 | Medium | Spectral analysis |
| wandas/frames/noct.py | 3 | Medium | N-octave analysis |
| wandas/frames/mixins/channel_processing_mixin.py | 3 | Medium | Mixin functionality |
| wandas/processing/psychoacoustic.py | 4 | Medium | Psychoacoustic ops |
| wandas/processing/temporal.py | 2 | Low | Temporal processing |
| wandas/frames/mixins/channel_collection_mixin.py | 2 | Low | Collection mixin |
| wandas/utils/generate_sample.py | 1 | Low | Sample generation |
| wandas/processing/spectral.py | 1 | Low | Spectral processing |
| wandas/io/wav_io.py | 1 | Medium | WAV I/O |

## Quality Assessment

### Current Quality Levels

Based on manual review of error messages:

**Level 1 (Basic)**: ~40% of errors
- Single sentence
- States what is wrong
- Missing "why" and "how" components
- Example: `raise ValueError("start must be less than end")`

**Level 2 (Intermediate)**: ~45% of errors
- Multiple pieces of information
- Includes actual values
- Missing actionable solution
- Example: `raise ValueError(f"Cutoff frequency must be between 0 Hz and {limit} Hz")`

**Level 3 (Advanced)**: ~15% of errors
- Clear problem statement
- Explanation of constraints
- Actionable guidance
- Example: Filter validation messages in `wandas/processing/filters.py`

### Common Issues

1. **Missing Context** (60% of errors)
   - No explanation of why the constraint exists
   - Missing valid range or acceptable values
   - No reference to related concepts

2. **Lack of Actionable Guidance** (70% of errors)
   - No suggestion on how to fix
   - No alternative approaches mentioned
   - No reference to documentation

3. **Inconsistent Formatting** (30% of errors)
   - Mixed use of single/multi-line messages
   - Inconsistent value formatting
   - Varying levels of detail

4. **Vague Descriptions** (25% of errors)
   - Generic messages like "Invalid input"
   - Missing specific values
   - Unclear about what was expected

## Priority Classification

### High Priority (30 errors, 30%)

**Criteria**: User-facing, frequently encountered, critical functionality

**Files**:
- `wandas/frames/channel.py` (15 errors)
  - Data dimension validation
  - Channel selection errors
  - Sampling rate mismatch
  - Type validation

- `wandas/io/readers.py` (4 errors)
  - File not found
  - File format validation
  - Channel loading errors

- `wandas/io/wdf_io.py` (3 errors)
  - File not found
  - File exists
  - Format version mismatch

- `wandas/processing/filters.py` (5 errors)
  - Cutoff frequency validation
  - Filter parameter validation

- `wandas/core/base_frame.py` (3 errors - user-facing subset)
  - Frame initialization
  - Operation validation

**Impact**: Direct user experience, common operations, critical workflows

**Recommendation**: Improve immediately in Phase 2

### Medium Priority (45 errors, 45%)

**Criteria**: Processing operations, less frequent but important

**Files**:
- `wandas/core/base_frame.py` (10 errors)
  - Internal processing validation
  - Metadata operations

- `wandas/visualization/plotting.py` (7 errors)
  - Plot parameter validation
  - Display errors

- `wandas/utils/frame_dataset.py` (7 errors)
  - Dataset operations
  - File handling

- `wandas/frames/roughness.py` (7 errors)
  - Bark band validation
  - Overlap validation

- `wandas/frames/spectrogram.py` (4 errors)
  - Spectrogram operations

- `wandas/processing/psychoacoustic.py` (4 errors)
  - Loudness calculations
  - Psychoacoustic metrics

- `wandas/frames/spectral.py` (3 errors)
- `wandas/frames/noct.py` (3 errors)

**Impact**: Processing workflows, advanced features

**Recommendation**: Improve in Phase 3

### Low Priority (25 errors, 25%)

**Criteria**: Internal errors, edge cases, rare scenarios

**Files**:
- NotImplementedError instances (13 errors)
  - Abstract methods
  - Future features
  - Format variations

- Internal validation (12 errors)
  - Debug assertions
  - Internal state checks

**Impact**: Developer-facing, rare edge cases

**Recommendation**: Improve as needed during regular development

## Specific Findings

### ValueError Analysis (60 errors)

**Common Patterns**:
1. Dimension validation (15 errors)
2. Range validation (20 errors)
3. Sampling rate mismatch (5 errors)
4. Parameter constraints (20 errors)

**Best Examples**:
```python
# Good: wandas/processing/filters.py:39
raise ValueError(f"Cutoff frequency must be between 0 Hz and {limit} Hz")
```

**Needs Improvement**:
```python
# Poor: wandas/frames/mixins/channel_processing_mixin.py:251
raise ValueError("start must be less than end")
```

**Recommended Pattern**:
```python
raise ValueError(
    f"Start time ({start}s) must be less than end time ({end}s).\n"
    f"The time range must be valid for extracting a signal segment.\n"
    f"Please ensure start < end or use start=None and end=None for the full signal."
)
```

### TypeError Analysis (16 errors)

**Common Patterns**:
1. Type mismatch in operations (8 errors)
2. Invalid parameter types (5 errors)
3. Unexpected return types (3 errors)

**Best Examples**:
```python
# Good: wandas/frames/channel.py:598
raise TypeError(
    "Addition target with SNR must be a ChannelFrame or "
    f"NumPy array: {type(other)}"
)
```

**Needs Improvement**:
```python
# Poor: wandas/frames/channel.py:624
raise TypeError("channel must be int, list, or None")
```

**Recommended Pattern**:
```python
raise TypeError(
    f"Expected int, list, or None for channel parameter, got {type(channel).__name__}.\n"
    f"Use int to select a single channel, list for multiple channels, or None for all.\n"
    f"Please convert your channel selector to one of these types."
)
```

### NotImplementedError Analysis (13 errors)

**Common Patterns**:
1. Abstract methods (5 errors)
2. Unsupported formats (4 errors)
3. Future features (4 errors)

**Recommended Pattern**:
```python
raise NotImplementedError(
    f"The '{feature_name}' feature is not yet implemented.\n"
    f"This is planned for a future release (see issue #XXX).\n"
    f"As a workaround, you can use '{alternative_approach}' or "
    f"contribute this feature at https://github.com/kasahart/wandas"
)
```

### FileNotFoundError Analysis (3 errors)

**Current State**:
- Basic file path reporting
- Missing troubleshooting guidance
- No permission check suggestions

**Recommended Improvement**:
```python
raise FileNotFoundError(
    f"File not found: {filepath}\n"
    f"The specified path does not exist or is not accessible.\n"
    f"Please check:\n"
    f"  1. Path is correct: {filepath.absolute()}\n"
    f"  2. File exists at this location\n"
    f"  3. You have read permissions\n"
    f"  4. Relative paths are resolved from: {Path.cwd()}"
)
```

### IndexError Analysis (5 errors)

**Common Patterns**:
1. Channel index out of range (3 errors)
2. Array index errors (2 errors)

**Recommended Pattern**:
```python
raise IndexError(
    f"Channel index {idx} is out of range for {n_channels} channels.\n"
    f"Valid indices are 0 to {n_channels - 1} (0-based indexing).\n"
    f"Please use an index in the range [0, {n_channels - 1}]."
)
```

## Recommendations

### Immediate Actions (Phase 2)

1. **Improve High-Priority Errors** (30 errors)
   - Focus on `wandas/frames/channel.py`
   - File I/O errors in `wandas/io/`
   - Filter validation in `wandas/processing/filters.py`

2. **Create Error Message Templates**
   - Add to `.github/copilot-instructions.md`
   - Include in development documentation

3. **Add Test Coverage**
   - Test error message content
   - Verify helpful error messages in CI/CD

### Medium-Term Actions (Phase 3)

1. **Improve Medium-Priority Errors** (45 errors)
   - Processing operations
   - Visualization errors
   - Dataset operations

2. **Standardize Format**
   - Consistent multi-line messages
   - Standard value formatting
   - Uniform guidance structure

### Long-Term Actions (Ongoing)

1. **Improve Low-Priority Errors** (25 errors)
   - NotImplementedError messages
   - Internal validation

2. **User Feedback Integration**
   - Collect feedback on error messages
   - Iterate based on actual user confusion
   - Update documentation

## Success Metrics

### Quantitative Metrics

- **Coverage**: % of errors following 3-element structure
  - Current: ~15%
  - Target (Phase 2): 50% (high-priority complete)
  - Target (Phase 3): 80% (high + medium complete)

- **Message Length**: Average characters per error message
  - Current: ~80 characters
  - Target: 200-300 characters (detailed but concise)

- **Component Completeness**: % with all 3 elements (What, Why, How)
  - Current: ~15%
  - Target: 90%

### Qualitative Metrics

- User satisfaction with error messages
- Reduced support requests related to common errors
- Faster issue resolution in GitHub issues

## Conclusion

The Wandas codebase contains 100 error messages with varying quality levels. While some messages (particularly in filter validation) are well-crafted, the majority lack actionable guidance and context. 

By focusing on the 30 high-priority errors in Phase 2, we can significantly improve the user experience for the most common error scenarios. The guidelines and templates created in this phase will ensure consistency as we improve the remaining messages in subsequent phases.

## Next Steps

1. Review and approve this analysis
2. Create implementation plan for Phase 2
3. Begin improving high-priority error messages
4. Update `.github/copilot-instructions.md` with error message guidelines
5. Track progress and measure improvements

---

**Document Version**: 1.0  
**Last Updated**: 2024-10-22  
**Status**: Complete (Phase 1)
