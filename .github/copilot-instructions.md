# Wandas Development Guidelines

## About Wandas

Wandas (**W**aveform **An**alysis **Da**ta **S**tructures) is a Python library for audio signal and waveform analysis. It provides a pandas-like API for signal processing, spectral analysis, and visualization.

**Core Focus**: Type-safe, immutable operations with lazy evaluation using Dask arrays.

## Core Principles

### Critical Rules (Always Follow)

1. **Type Safety First**
   - Use mypy strict mode (`strict = true`)
   - All functions require complete type hints including return types
   - Use type aliases from `wandas.utils.types` (e.g., `NDArrayReal`, `NDArrayComplex`)

2. **Immutability**
   - Never modify input arrays or frames in-place
   - Operations always return new frames
   - Preserve operation history in metadata

3. **Pandas-like API**
   - Support method chaining
   - Return `self` or new instance for chainable operations
   - Follow pandas naming conventions where applicable

4. **Lazy Evaluation**
   - Prefer Dask arrays over NumPy for large data
   - Avoid unnecessary `.compute()` calls
   - Let users decide when to materialize results

### Design Principles

- **SOLID**: Single responsibility, open-closed, Liskov substitution, interface segregation, dependency inversion
- **YAGNI**: Implement only what's needed now
- **KISS**: Keep solutions simple and understandable
- **DRY**: Eliminate code and knowledge duplication

## Coding Standards

### Type Hints (Required)

```python
# Good: Complete type hints
def process_signal(data: NDArrayReal, sampling_rate: float) -> NDArrayComplex:
    ...

# Bad: Missing type hints
def process_signal(data, sampling_rate):
    ...
```

### Operations (Signal Processing)

All signal processing operations must:
- Inherit from `AudioOperation[InputT, OutputT]`
- Use `@register_operation` decorator
- Implement `_process_array` method
- Record operation in `operation_history`

```python
from wandas.processing.base import AudioOperation, register_operation

@register_operation
class MyFilter(AudioOperation[NDArrayReal, NDArrayReal]):
    name = "my_filter"
    
    def __init__(self, sampling_rate: float, cutoff: float) -> None:
        super().__init__(sampling_rate, cutoff=cutoff)
        
    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        # Implementation here
        ...
```

### Error Messages (3-Element Rule)

Every error must include:
1. **WHAT**: Clear problem description
2. **WHY**: Explain the constraint
3. **HOW**: Provide actionable solution

```python
# Good: Informative error
raise ValueError(
    f"Sampling rate mismatch\n"
    f"  File: {file_sr} Hz\n"
    f"  Expected: {sampling_rate} Hz\n"
    f"Use signal.resample({sampling_rate}) to convert the sampling rate."
)

# Bad: Vague error
raise ValueError("Invalid sampling rate")
```

### Documentation (Required)

- **Language**: English only
- **Style**: NumPy/Google docstring format
- **Sections**: Parameters, Returns, Raises, Examples
- **Math**: LaTeX notation for formulas

```python
def fft(self, n_fft: Optional[int] = None) -> "SpectralFrame":
    """
    Apply Fast Fourier Transform to the signal.

    Parameters
    ----------
    n_fft : int, optional
        FFT size. Must be power of 2 for optimal performance.

    Returns
    -------
    SpectralFrame
        Frequency domain representation of the signal.

    Raises
    ------
    ValueError
        If n_fft is not a positive integer.

    Examples
    --------
    >>> signal = wd.read_wav("audio.wav")
    >>> spectrum = signal.fft(n_fft=2048)
    """
    ...
```

## Testing Requirements

### Test Structure

- Use pytest with fixtures
- Test behavior, not implementation
- Name tests descriptively: `test_<action>_<expected_result>`
- Validate against theoretical values (not just non-zero checks)

```python
def test_fft_preserves_energy(sample_signal: ChannelFrame) -> None:
    """Test FFT preserves signal energy (Parseval's theorem)."""
    time_energy = np.sum(np.abs(sample_signal.data) ** 2)
    spectrum = sample_signal.fft()
    freq_energy = np.sum(np.abs(spectrum.data) ** 2) / len(sample_signal.data)
    
    # Theoretical: Both should be equal per Parseval's theorem
    np.testing.assert_allclose(time_energy, freq_energy, rtol=1e-10)
```

### Coverage

- Target: 100% coverage (minimum 90%)
- Use `# pragma: no cover` only for:
  - Platform-specific code
  - Debug code
  - Type checking blocks (`if TYPE_CHECKING:`)

## Code Change Workflow

### Before Implementation

1. **Check existing designs**: Review `docs/design/INDEX.md` for similar patterns
2. **Create plan**: Write `docs/design/working/plans/PLAN_<feature>.md` (gitignored)
3. **Write tests first**: Test-driven development, ensure tests fail initially

### During Implementation

4. **Small changes**: Make minimal, focused modifications
5. **Test frequently**: Run tests after each logical change
6. **Lint continuously**: Use ruff and mypy during development

### After Implementation

7. **Validate coverage**: Ensure tests achieve target coverage
8. **Check types**: `uv run mypy --config-file=pyproject.toml`
9. **Format code**: `uv run ruff format`
10. **Create summary**: Document in `docs/design/working/drafts/` if significant

## Tools and Commands

### Quality Checks

```bash
# Type checking
uv run mypy --config-file=pyproject.toml

# Linting and formatting
uv run ruff check wandas tests --fix
uv run ruff format wandas tests

# Testing
uv run pytest                                    # All tests
uv run pytest --cov=wandas --cov-report=html    # With coverage
uv run pytest tests/frames/test_channel_frame.py # Specific file
```

### Pre-commit

```bash
pre-commit install    # Install hooks
pre-commit run --all-files  # Manual run
```

## References

### Documentation
- **Detailed Coding Standards**: `docs/development/coding_standards.md` (to be created)
- **Error Message Guide**: `docs/development/error_message_guide.md`
- **Testing Guide**: `docs/development/testing_guide.md` (to be created)
- **Design Patterns**: `docs/design/INDEX.md`
- **Design Lifecycle**: `docs/design/DOCUMENT_LIFECYCLE.md`

### External Resources
- **NumPy Docstring Guide**: https://numpydoc.readthedocs.io/
- **Python Type Hints**: https://docs.python.org/3/library/typing.html
- **pytest Documentation**: https://docs.pytest.org/

### Project Links
- **Documentation**: https://kasahart.github.io/wandas/
- **Repository**: https://github.com/kasahart/wandas
- **Issues**: https://github.com/kasahart/wandas/issues

---

**Note**: For detailed guidance, see reference documentation. For Japanese documentation, see `docs/ja/` (when available).

---
applyTo: "**/*.py"
---

# Python-Specific Guidelines

## Array Handling

- **Prefer Dask**: Use `dask.array` for lazy evaluation
- **Explicit axes**: Always specify `axis` parameter
- **Shape validation**: Check dimensions at function entry

```python
import dask.array as da

def apply_filter(data: DaskArray, axis: int = -1) -> DaskArray:
    if data.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D array, got {data.ndim}D")
    return da.some_operation(data, axis=axis)
```

## Metadata Management

- **Operation history**: Always record operations
- **Channel metadata**: Preserve or update appropriately
- **Previous reference**: Link to previous frame for traceability

```python
def create_processed_frame(
    self,
    data: NDArrayReal,
    operation_name: str,
    **params: Any
) -> "ChannelFrame":
    new_history = self.operation_history.copy()
    new_history.append(OperationRecord(
        name=operation_name,
        params=params,
        timestamp=datetime.now()
    ))
    
    return ChannelFrame(
        data=data,
        sampling_rate=self.sampling_rate,
        channel_metadata=self.channel_metadata,
        operation_history=new_history,
        previous=self
    )
```

## Performance

- **Vectorize**: Use NumPy/Dask operations, avoid Python loops
- **Avoid compute**: Return Dask arrays without calling `.compute()`
- **Memory efficiency**: Consider chunk sizes for large arrays

---
applyTo: "tests/**/*.py"
---

# Test-Specific Guidelines

## Numerical Validation

**Critical**: Always validate against theoretical values, not just existence checks.

```python
# Good: Validates against theory
def test_normalize_produces_unit_maximum(signal: ChannelFrame) -> None:
    """Test normalization produces amplitude of 1.0."""
    normalized = signal.normalize()
    # Theory: max amplitude should be exactly 1.0
    assert np.abs(np.max(np.abs(normalized.data)) - 1.0) < 1e-10

# Bad: Only checks non-zero
def test_normalize_works(signal: ChannelFrame) -> None:
    normalized = signal.normalize()
    assert normalized.data is not None  # Not validating correctness!
```

## Test Independence

- Each test runs independently
- No shared mutable state between tests
- Use fixtures for setup, not global variables

## Metadata Validation

Always verify operation history and metadata preservation:

```python
def test_filter_records_operation_history(signal: ChannelFrame) -> None:
    """Test that filtering records operation in history."""
    filtered = signal.low_pass_filter(cutoff=1000)
    assert len(filtered.operation_history) == len(signal.operation_history) + 1
    assert filtered.operation_history[-1].name == "low_pass_filter"
    assert filtered.operation_history[-1].params["cutoff"] == 1000
```

---
applyTo: "docs/**/*.md"
---

# Documentation Guidelines

## Language and Style

- **Primary language**: English
- **Technical accuracy**: Precision over style
- **Code examples**: Must be runnable and tested
- **Cross-references**: Link to related documents

## Structure

- Use clear heading hierarchy (H1 → H2 → H3)
- Include table of contents for documents >200 lines
- Add "Last Updated" date at the top
- Reference related documents explicitly

## Code Examples

```markdown
## Example Usage

\`\`\`python
import wandas as wd

# Load and process audio
signal = wd.read_wav("audio.wav")
spectrum = signal.fft(n_fft=2048)
spectrum.plot()
\`\`\`
```

---
