# Wandas: Waveform Analysis Data Structures

## Project Overview

Wandas is a Python library for audio signal and waveform analysis, providing pandas-like APIs for signal processing, spectral analysis, and visualization. The library focuses on type safety, method chaining, and lazy evaluation for efficient processing of large audio datasets.

**Domain**: Audio signal processing, acoustic analysis, psychoacoustics
**Target Users**: Researchers, audio engineers, data scientists working with time-series audio data

## Core Design Principles

1. **Pandas-like Interface**: Familiar DataFrame-style API for signal processing operations
2. **Type Safety**: Strict mypy compliance with comprehensive type hints
3. **Method Chaining**: Fluent API enabling intuitive multi-step processing pipelines
4. **Lazy Evaluation**: Dask arrays for memory-efficient handling of large datasets
5. **Immutability**: Operations return new frames, preserving original data
6. **Traceability**: Operation history tracking in metadata for reproducibility
7. **SOLID Principles**: Single responsibility, open-closed, Liskov substitution, interface segregation, dependency inversion
8. **Simplicity**: KISS (Keep It Simple), DRY (Don't Repeat Yourself), YAGNI (You Aren't Gonna Need It)

## Technology Stack

- **Python**: 3.9+ (type hints, dataclasses)
- **Core Libraries**: NumPy (computation), Dask (lazy evaluation), pandas (inspiration)
- **Signal Processing**: scipy.signal, custom FFT implementations
- **Visualization**: matplotlib, japanize-matplotlib (Japanese text support)
- **Quality Tools**: mypy (strict mode), ruff (linting/formatting), pytest (testing)

## Key Architecture Patterns

### Frame Types
- `ChannelFrame`: Time-domain audio signals
- `SpectralFrame`: Frequency-domain representations
- `SpectrogramFrame`: Time-frequency representations
- `NOctFrame`: N-octave band analysis
- `RoughnessFrame`: Psychoacoustic roughness analysis

### Operation Pattern
Extend `AudioOperation[InputType, OutputType]` base class:
- Use `@register_operation` decorator
- Implement `_process_array()` method
- Return new frames (immutability)
- Track operations in metadata

### Error Handling
Follow 3-element pattern (WHAT/WHY/HOW):
```python
raise ValueError(
    f"Parameter out of range\n"
    f"  Got: {actual}\n"
    f"  Expected: {expected}\n"
    f"Use valid range to fix this error."
)
```

For complete error handling patterns and examples, see `.github/instructions/python-code.instructions.md`.

## Essential Coding Standards

**Type Hints**: Required on all functions (parameters + return values)

**Docstrings**: NumPy format, English, include Parameters/Returns/Raises/Examples

**Testing**:
- Target 100% coverage (minimum 90%)
- Validate against theoretical values, not just ranges
- Test normal cases, boundaries, errors, and metadata preservation
- Use appropriate tolerances for floating-point comparisons

**Immutability**: Never modify input data, always return new instances

**Metadata**: Track operation history, preserve channel metadata

## Specialized Instructions

For detailed implementation standards, refer to:
- **Python Code**: `.github/instructions/python-code.instructions.md` (applies to `wandas/**/*.py`)
- **Tests**: `.github/instructions/tests.instructions.md` (applies to `tests/**/*.py`)
- **Notebooks**: `.github/instructions/notebooks.instructions.md` (applies to `**/*.ipynb`)

Here is a list of instruction files that contain rules for modifying or creating new code.
These files are important for ensuring that the code is modified or created correctly.
Please make sure to follow the rules specified in these files when working with the codebase.
If you have not already read the file, use the `view` tool to acquire it.
Make sure to acquire the instructions before making any changes to the code.
| Pattern | File Path | Description |
| ------- | --------- | ----------- |
| tests/**/*.py | '.github/instructions/tests.instructions.md' | Test Standards for Wandas: validation approach, coverage requirements, test structure, and theoretical validation examples |
| wandas/**/*.py | '.github/instructions/python-code.instructions.md' | Python Code Standards for Wandas: type safety, signal processing patterns, error handling, docstrings, and performance guidelines |
| **/*.ipynb | '.github/instructions/notebooks.instructions.md' | Jupyter Notebook Standards for Wandas: documentation, code organization, visualization, and best practices |

## Common Workflows

**Adding New Operations**:
1. Extend AudioOperation base class
2. Validate parameters in `__init__`
3. Implement `_process_array()` logic
4. Write comprehensive tests (normal/boundary/error cases)
5. Document with examples

**Method Chaining Example**:
```python
result = (
    wd.read_wav("audio.wav")
    .normalize()
    .low_pass_filter(cutoff=1000)
    .resample(16000)
)
```

## Development Workflow

### Before Coding
1. **Check existing patterns**: Review `docs/design/INDEX.md` for similar features or past design decisions
2. **Create a plan**: Draft in `docs/design/working/plans/PLAN_<feature>.md` (gitignored, editable)
3. **Write tests first**: Follow TDD - write failing tests before implementation

### During Development
1. **Small commits**: Use meaningful, atomic commits with conventional commit messages
2. **Run tests frequently**: Execute `uv run pytest` after each significant change
3. **Update metadata**: Ensure operation history tracks all transformations
4. **Check types continuously**: Run `uv run mypy` to catch type errors early

### After Implementation
1. **Quality checks**:
   - Tests: `uv run pytest --cov=wandas` (target 100% coverage)
   - Types: `uv run mypy --config-file=pyproject.toml` (strict mode)
   - Lint: `uv run ruff check wandas tests`
2. **Documentation**: Update docstrings, API docs, and add examples if needed
3. **Design docs**: Create summary in `docs/design/guides/` for significant design decisions
4. **Final review**: Verify all tests pass, no regressions, backward compatibility maintained

## Repository Structure

- `wandas/`: Source code (frames, processing, io, utils, visualization)
- `tests/`: Comprehensive test suite (mirrors source structure)
- `docs/`: MkDocs documentation and design documents
- `examples/`: Jupyter notebooks demonstrating usage

## Quality Standards

- All PRs require: passing tests, mypy strict mode, ruff compliance
- Coverage target: 100% (minimum 90%)
- Documentation: English docstrings, Japanese UI optional
- Design decisions: Document in `docs/design/` for complex changes

## Development Commands

```bash
# Run tests with coverage
uv run pytest --cov=wandas

# Type checking
uv run mypy --config-file=pyproject.toml

# Linting and formatting
uv run ruff check wandas tests
uv run ruff format wandas tests
```

## Additional Resources

For detailed guidelines, see:
- Testing patterns: `docs/development/error_message_guide.md`
- Contributing workflow: Standard GitHub flow with feature branches
- Design documents: `docs/design/INDEX.md`
