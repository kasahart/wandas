# Refactor Code with Test Coverage

Refactor code while maintaining 100% test coverage.

## Process

1. **Analyze Current Implementation**
   - Understand existing functionality
   - Identify improvement opportunities
   - Review current test coverage

2. **Run Existing Tests**
   - Verify all tests pass: `uv run pytest`
   - Check coverage: `uv run pytest --cov=<module> --cov-report=term-missing`
   - Note any gaps in coverage

3. **Plan Refactoring**
   - Identify target improvements
   - Ensure changes maintain API compatibility
   - Consider performance implications

4. **Implement Incrementally**
   - Make small, focused changes
   - Run tests after each change
   - Commit working states

5. **Add/Update Tests**
   - Test new code paths
   - Update tests if API changes
   - Maintain 100% coverage

6. **Verify Quality**
   - All tests pass
   - Coverage remains 100%
   - Type checking passes
   - Linting passes

## Refactoring Targets

### Type Safety

**Before:**
```python
def process(data, rate):
    return data / rate
```

**After:**
```python
def process(data: NDArrayReal, rate: float) -> NDArrayReal:
    """
    Process data with given rate.

    Parameters
    ----------
    data : NDArrayReal
        Input data array
    rate : float
        Processing rate

    Returns
    -------
    NDArrayReal
        Processed data
    """
    return data / rate
```

### Code Clarity

**Before:**
```python
def f(x, y, z):
    return x * y + z if z > 0 else x * y
```

**After:**
```python
def apply_gain_with_offset(
    signal: NDArrayReal,
    gain: float,
    offset: float
) -> NDArrayReal:
    """
    Apply gain and optional offset to signal.

    Parameters
    ----------
    signal : NDArrayReal
        Input signal
    gain : float
        Gain factor to apply
    offset : float
        Offset to add after gain. Only applied if positive.

    Returns
    -------
    NDArrayReal
        Processed signal
    """
    amplified = signal * gain
    if offset > 0:
        amplified += offset
    return amplified
```

### DRY Principle

**Before (Repeated Code):**
```python
def low_pass_filter(self, cutoff):
    if cutoff >= self.sampling_rate / 2:
        raise ValueError(f"Cutoff {cutoff} >= Nyquist")
    # ... filter implementation

def high_pass_filter(self, cutoff):
    if cutoff >= self.sampling_rate / 2:
        raise ValueError(f"Cutoff {cutoff} >= Nyquist")
    # ... filter implementation
```

**After (Extracted Validation):**
```python
def _validate_cutoff(self, cutoff: float) -> None:
    """
    Validate cutoff frequency against Nyquist frequency.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz

    Raises
    ------
    ValueError
        If cutoff >= Nyquist frequency
    """
    nyquist = self.sampling_rate / 2
    if cutoff >= nyquist:
        raise ValueError(
            f"Cutoff frequency out of range\n"
            f"  Got: {cutoff} Hz\n"
            f"  Expected: < {nyquist} Hz (Nyquist)\n"
            f"Use cutoff < sampling_rate / 2"
        )

def low_pass_filter(self, cutoff: float) -> "ChannelFrame":
    """Apply low-pass filter."""
    self._validate_cutoff(cutoff)
    # ... filter implementation

def high_pass_filter(self, cutoff: float) -> "ChannelFrame":
    """Apply high-pass filter."""
    self._validate_cutoff(cutoff)
    # ... filter implementation
```

### Performance Optimization

**Before:**
```python
def process(self):
    result = []
    for i in range(len(self.data)):
        result.append(self.data[i] * 2)
    return np.array(result)
```

**After:**
```python
def process(self) -> NDArrayReal:
    """
    Process data using vectorized operations.

    Returns
    -------
    NDArrayReal
        Processed data

    Notes
    -----
    Uses vectorized NumPy operations for performance.
    """
    return self.data * 2
```

## Test Update Patterns

### When API Changes

**Before:**
```python
def test_old_api():
    result = obj.process(data, param1, param2)
    assert result.shape == expected_shape
```

**After API Change:**
```python
def test_new_api():
    """Test refactored API with keyword arguments."""
    result = obj.process(data, param1=value1, param2=value2)
    assert result.shape == expected_shape

def test_old_api_compatibility():
    """Test backward compatibility with positional args."""
    with pytest.warns(DeprecationWarning):
        result = obj.process(data, value1, value2)
    assert result.shape == expected_shape
```

### When Adding Internal Helpers

```python
def test_internal_validation():
    """Test extracted validation helper."""
    obj = MyClass(sampling_rate=44100)

    # Valid input
    obj._validate_cutoff(1000)  # Should not raise

    # Invalid input
    with pytest.raises(ValueError, match="out of range"):
        obj._validate_cutoff(50000)
```

## Checklist

### Before Refactoring
- [ ] All current tests pass
- [ ] Current coverage measured
- [ ] Refactoring goals identified
- [ ] API compatibility plan determined

### During Refactoring
- [ ] Changes made incrementally
- [ ] Tests run after each change
- [ ] Working states committed
- [ ] Type hints added/updated

### After Refactoring
- [ ] All tests pass
- [ ] Coverage maintained or improved
- [ ] Type checking passes: `uv run mypy`
- [ ] Linting passes: `uv run ruff check`
- [ ] Code formatted: `uv run ruff format`
- [ ] Documentation updated
- [ ] Performance verified (if applicable)

## Common Pitfalls

### Breaking Immutability

```python
# ❌ Bad: Modifies input
def normalize(self):
    self.data /= np.max(self.data)
    return self

# ✅ Good: Returns new instance
def normalize(self) -> "ChannelFrame":
    normalized = self.data / np.max(self.data)
    return self._create_new_frame(normalized, operation_name="normalize")
```

### Losing Metadata

```python
# ❌ Bad: Loses operation history
def process(self):
    return ChannelFrame(data=new_data, sampling_rate=self.sampling_rate)

# ✅ Good: Preserves metadata
def process(self) -> "ChannelFrame":
    return self._create_new_frame(
        new_data,
        operation_name="process"
    )
```

### Incomplete Type Hints

```python
# ❌ Bad: Missing return type
def process(self, data: NDArrayReal):
    return data * 2

# ✅ Good: Complete type hints
def process(self, data: NDArrayReal) -> NDArrayReal:
    return data * 2
```

## Testing Coverage

Ensure coverage for all code paths:

```bash
# Check coverage
uv run pytest --cov=wandas.module --cov-report=html

# View HTML report
open htmlcov/index.html

# Find missing coverage
uv run pytest --cov=wandas.module --cov-report=term-missing
```

## Example Workflow

```bash
# 1. Create feature branch
git checkout -b refactor-filter-module

# 2. Run tests (should pass)
uv run pytest tests/processing/test_filters.py -v

# 3. Check current coverage
uv run pytest tests/processing/test_filters.py --cov=wandas.processing.filters --cov-report=term-missing

# 4. Make incremental changes
# ... edit code ...

# 5. Run tests after each change
uv run pytest tests/processing/test_filters.py -v

# 6. Commit working state
git add wandas/processing/filters.py
git commit -m "refactor: Extract cutoff validation helper"

# 7. Final verification
uv run pytest
uv run mypy --config-file=pyproject.toml
uv run ruff check wandas tests

# 8. Push and create PR
git push origin refactor-filter-module
```
