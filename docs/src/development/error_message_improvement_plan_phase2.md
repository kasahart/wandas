# Error Message Improvement Plan - Phase 2

## Overview

This document outlines the concrete implementation plan for Phase 2 of the error message improvement initiative. Phase 1 focused on analysis and guideline creation; Phase 2 will implement improvements for high-priority errors.

**Target**: Improve 30 high-priority error messages (30% of total)  
**Timeline**: 2 weeks  
**Success Criteria**: All high-priority errors follow the 3-element structure (What, Why, How)

## High-Priority Targets

### 1. wandas/frames/channel.py (15 errors)

**Priority**: Critical - Core user-facing API

#### Errors to Improve

| Line | Error Type | Current Message | Priority | Estimated Time |
|------|-----------|-----------------|----------|---------------|
| 69 | ValueError | Data dimension validation | High | 30min |
| 219 | ValueError | Sampling rate mismatch | High | 30min |
| 305 | ValueError | Sampling rate mismatch | High | 30min |
| 598 | ValueError | Data dimension validation | High | 30min |
| 613 | ValueError | Channel labels mismatch | High | 30min |
| 623 | ValueError | Channel units mismatch | High | 30min |
| 750 | ValueError | Channel index out of range | High | 30min |
| 758 | ValueError | Channel index out of range | High | 30min |
| 764 | TypeError | Channel type validation | High | 30min |
| 787 | ValueError | File reading error | Medium | 20min |
| 805 | ValueError | Chunk size validation | Medium | 20min |
| 720 | FileNotFoundError | File not found | High | 30min |
| 1138 | IndexError | Index out of range | High | 30min |
| 1143 | KeyError | Label not found | Medium | 20min |

**Total Time**: ~6 hours

### 2. wandas/io/readers.py (4 errors)

**Priority**: Critical - File I/O operations

#### Errors to Improve

| Line | Error Type | Current Message | Priority | Estimated Time |
|------|-----------|-----------------|----------|---------------|
| TBD | FileNotFoundError | File not found | High | 30min |
| TBD | ValueError | File format validation | High | 30min |
| TBD | ValueError | Channel loading errors | High | 30min |
| TBD | NotImplementedError | Unsupported format | Medium | 20min |

**Total Time**: ~2 hours

### 3. wandas/io/wdf_io.py (3 errors)

**Priority**: Critical - File format operations

#### Errors to Improve

| Line | Error Type | Current Message | Priority | Estimated Time |
|------|-----------|-----------------|----------|---------------|
| 172 | FileNotFoundError | File not found | High | 30min |
| TBD | FileExistsError | File already exists | Medium | 20min |
| TBD | ValueError | Format version mismatch | High | 30min |

**Total Time**: ~1.5 hours

### 4. wandas/processing/filters.py (5 errors)

**Priority**: High - Common signal processing operations

#### Errors to Improve

| Line | Error Type | Current Message | Priority | Estimated Time |
|------|-----------|-----------------|----------|---------------|
| 39 | ValueError | Cutoff frequency validation (High-pass) | High | 30min |
| TBD | ValueError | Cutoff frequency validation (Low-pass) | High | 30min |
| TBD | ValueError | Low cutoff validation (Band-pass) | High | 30min |
| TBD | ValueError | High cutoff validation (Band-pass) | High | 30min |
| TBD | ValueError | Cutoff order validation (Band-pass) | High | 30min |

**Total Time**: ~2.5 hours

### 5. wandas/core/base_frame.py (3 user-facing errors)

**Priority**: High - Base functionality

#### Errors to Improve

| Line | Error Type | Current Message | Priority | Estimated Time |
|------|-----------|-----------------|----------|---------------|
| 285 | ValueError | Frame initialization | High | 30min |
| 302 | ValueError | Empty list indexing | High | 30min |
| 428 | KeyError | Channel label not found | High | 30min |

**Total Time**: ~1.5 hours

## Implementation Strategy

### Week 1: Core User-Facing APIs

**Days 1-2**: wandas/frames/channel.py (6 hours)
- Improve all ValueError instances
- Improve TypeError instances
- Improve FileNotFoundError, IndexError, KeyError

**Days 3-4**: wandas/io/ (3.5 hours)
- Improve readers.py errors
- Improve wdf_io.py errors
- Test file I/O error scenarios

**Day 5**: wandas/core/base_frame.py (1.5 hours)
- Improve high-priority errors
- Test base frame operations

### Week 2: Signal Processing & Testing

**Days 1-2**: wandas/processing/filters.py (2.5 hours)
- Improve all filter validation errors
- Test filter edge cases

**Days 3-4**: Testing & Documentation
- Add test cases for error messages
- Update examples in documentation
- Verify all improvements

**Day 5**: Review & Refinement
- Code review
- Final testing
- Documentation updates

## Template Application

### Before (Example from channel.py:69)

```python
if data.ndim > 2:
    raise ValueError(
        f"Data must be 1-dimensional or 2-dimensional. Shape: {data.shape}"
    )
```

### After (Applying 3-Element Structure)

```python
if data.ndim > 2:
    raise ValueError(
        # What: 何が問題か
        f"Data must be 1D or 2D, got {data.ndim}D array with shape {data.shape}.\n"
        # Why: なぜダメか
        f"ChannelFrame expects 1D arrays (single channel) or 2D arrays "
        f"(shape: [n_channels, n_samples]) for multi-channel data.\n"
        # How: どうすればいいか
        f"Please reshape your array: data.reshape(n_channels, -1) "
        f"or select a 2D slice of your {data.ndim}D array."
    )
```

## Testing Strategy

### Unit Tests for Error Messages

```python
def test_channel_frame_dimension_error_message():
    """Test that dimension error provides helpful guidance."""
    invalid_data = np.random.rand(2, 3, 4)  # 3D array
    
    with pytest.raises(ValueError) as exc_info:
        ChannelFrame.from_numpy(invalid_data, sampling_rate=44100)
    
    error_msg = str(exc_info.value)
    
    # Verify 3 elements present
    assert "1D or 2D" in error_msg  # What
    assert "expects" in error_msg or "ChannelFrame" in error_msg  # Why
    assert "reshape" in error_msg or "Please" in error_msg  # How
    
    # Verify specific values
    assert "3D" in error_msg
    assert str(invalid_data.shape) in error_msg


def test_file_not_found_error_message():
    """Test that file not found provides troubleshooting steps."""
    nonexistent_file = Path("/nonexistent/audio.wav")
    
    with pytest.raises(FileNotFoundError) as exc_info:
        ChannelFrame.read(nonexistent_file)
    
    error_msg = str(exc_info.value)
    
    # Verify helpful guidance
    assert "not found" in error_msg.lower()  # What
    assert "check" in error_msg.lower()  # How
    assert str(nonexistent_file) in error_msg  # Specific path
    
    # Should provide troubleshooting steps
    assert "1." in error_msg or "2." in error_msg  # Numbered steps


def test_cutoff_frequency_error_message():
    """Test that filter cutoff validation provides valid range."""
    sr = 44100
    invalid_cutoff = 25000  # > Nyquist
    
    with pytest.raises(ValueError) as exc_info:
        HighPassFilter(sampling_rate=sr, cutoff=invalid_cutoff)
    
    error_msg = str(exc_info.value)
    
    # Verify range information
    assert str(invalid_cutoff) in error_msg  # Actual value
    assert str(sr / 2) in error_msg  # Nyquist frequency
    assert "between" in error_msg or "range" in error_msg  # Range indication
    assert "aliasing" in error_msg.lower()  # Why explanation
```

### Integration Tests

```python
def test_error_message_consistency():
    """Test that error messages follow consistent format across modules."""
    # Test various error scenarios
    test_cases = [
        (lambda: ChannelFrame.from_numpy(np.random.rand(2, 3, 4), 44100), ValueError),
        (lambda: ChannelFrame.read("/nonexistent.wav"), FileNotFoundError),
        (lambda: HighPassFilter(44100, cutoff=50000), ValueError),
    ]
    
    for test_func, expected_error in test_cases:
        with pytest.raises(expected_error) as exc_info:
            test_func()
        
        error_msg = str(exc_info.value)
        
        # Check minimum length (should be descriptive)
        assert len(error_msg) > 100, f"Error message too short: {error_msg}"
        
        # Check for newlines (multi-line structure)
        assert "\n" in error_msg, f"Error message should be multi-line: {error_msg}"
```

## Quality Metrics

### Quantitative Targets

- **Coverage**: 100% of high-priority errors (30 errors)
- **Message Length**: Average 200-300 characters
- **Component Completeness**: 100% with all 3 elements
- **Line Count**: 3-6 lines per error message

### Qualitative Targets

- Clear problem statement
- Specific values included (actual vs. expected)
- Actionable guidance provided
- Consistent formatting
- Professional tone

## Documentation Updates

### Files to Update

1. **docs/src/development/error_message_guide.md**
   - Add real examples from improved errors
   - Update statistics

2. **docs/src/development/error_message_analysis.md**
   - Update coverage metrics
   - Mark improved errors

3. **.github/copilot-instructions.md**
   - Already updated with guidelines
   - Add Phase 2 completion note

4. **CHANGELOG.md**
   - Document error message improvements
   - List affected modules

## Rollout Plan

### Step 1: Implementation (Week 1-2)
- Improve error messages according to plan
- Add unit tests for error messages
- Run existing test suite to ensure no breakage

### Step 2: Review (End of Week 2)
- Self-review all changes
- Verify 3-element structure
- Check against quality checklist

### Step 3: Testing (End of Week 2)
- Run full test suite
- Manual testing of common scenarios
- Verify error messages in practice

### Step 4: Documentation (End of Week 2)
- Update all documentation
- Create examples in docstrings
- Update changelog

### Step 5: Release (After Phase 2)
- Merge to main branch
- Tag as part of next release
- Announce improvements in release notes

## Risk Mitigation

### Potential Risks

1. **Breaking Changes**: Error message changes might break tests that check exact message content
   - **Mitigation**: Update tests to check for key components rather than exact strings

2. **Increased Message Length**: Longer messages might clutter logs
   - **Mitigation**: Keep messages concise but complete (200-300 chars)

3. **Localization**: Multi-line messages harder to localize
   - **Mitigation**: Document localization strategy for future (currently English-only)

4. **Performance**: String formatting overhead
   - **Mitigation**: Negligible - errors are exceptional cases

## Success Criteria

Phase 2 will be considered successful when:

- [ ] All 30 high-priority errors improved
- [ ] All improved errors follow 3-element structure
- [ ] All tests pass (existing + new error message tests)
- [ ] Documentation updated
- [ ] Code review completed
- [ ] Metrics show 100% coverage for high-priority errors

## Next Steps After Phase 2

- **Phase 3**: Improve medium-priority errors (45 errors)
- **Phase 4**: Improve low-priority errors (25 errors)
- **Ongoing**: Maintain error message quality for new code

---

**Document Version**: 1.0  
**Created**: 2024-10-22  
**Status**: Ready for Implementation
