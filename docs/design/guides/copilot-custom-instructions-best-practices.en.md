# Best Practices for GitHub Copilot Repository Custom Instructions

**Last Updated**: November 2, 2025

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Structure](#basic-structure)
3. [Content Organization Best Practices](#content-organization-best-practices)
4. [Writing Style Guide](#writing-style-guide)
5. [Providing Effective Examples](#providing-effective-examples)
6. [Maintenance](#maintenance)
7. [Common Pitfalls](#common-pitfalls)
8. [References](#references)

## Introduction

GitHub Copilot's repository custom instructions (`.github/copilot-instructions.md`) is a powerful feature for conveying project-specific coding standards, design principles, and best practices to Copilot.

### Why Custom Instructions Matter

- **Consistency**: Maintain unified coding style across the project
- **Quality Improvement**: Automatically apply project-specific best practices
- **Onboarding**: Help new developers learn project conventions
- **Efficiency**: Eliminate repetitive explanations and accelerate development

### Target Audience

This guide is intended for:

- GitHub repository maintainers
- Project leaders
- Developers establishing coding standards
- Teams looking to improve development efficiency

## Basic Structure

### File Location

```text
.github/
└── copilot-instructions.md
```

### YAML Front Matter (Optional)

Add YAML front matter at the beginning to apply instructions to specific file types:

```yaml
---
applyTo: '.py, .ipynb'
---
```

**Supported Settings**:

- `applyTo`: Comma-separated list of file extensions (e.g., `'.py, .js, .ts'`)
- Extensions must start with a dot

**Examples**:

```yaml
---
# Apply only to Python files and notebooks
applyTo: '.py, .ipynb'
---

---
# Apply only to JavaScript and TypeScript
applyTo: '.js, .ts, .jsx, .tsx'
---
```

### Recommended File Size

- **Ideal**: 500-1500 lines
- **Maximum**: Up to 2000 lines
- Beyond that, split content into multiple sections with an index

**Current wandas project**: ~970 lines (within appropriate range)

## Content Organization Best Practices

### 1. Clear Hierarchical Structure

```markdown
# Project Name Development Guidelines

## Project Overview
[Brief project description]

## Design Principles
### Domain-Specific Principles
### Universal Design Principles

## Coding Standards
### 1. Type Hints
### 2. Error Handling
### 3. Testing

## Code Change Procedures

## Tools and Workflow
```

**Recommended Points**:

- ✅ Clear heading hierarchy (H1 → H2 → H3)
- ✅ Numbered lists for procedures
- ✅ Related sections
- ❌ Excessive nesting (avoid H5+)
- ❌ Duplicate sections

### 2. Place Project Overview First

```markdown
## Project Overview
Wandas (**W**aveform **An**alysis **Da**ta **S**tructures) is a Python library
specialized for acoustic signal and waveform analysis.
It provides pandas-like API for signal processing, spectral analysis, and visualization.
```

**Purpose**:

- Help Copilot understand project purpose and domain
- Promote context-appropriate code generation
- Make it easier for new developers to understand the project

### 3. Explicitly State Design Principles

Recommend categorizing design principles into two categories:

#### Domain-Specific Principles

Project-specific design policies:

```markdown
### Domain-Specific Principles

1. **Pandas-like Interface**: Enable users to process signals with pandas-like operations
2. **Type Safety**: Comply with mypy strict mode to prevent runtime errors
3. **Method Chaining**: Allow intuitive description of multiple operations through method chaining
4. **Lazy Evaluation**: Use Dask arrays for memory-efficient processing of large data
```

#### Universal Design Principles

SOLID principles, YAGNI, KISS, DRY, etc.:

```markdown
### Universal Design Principles

#### SOLID Principles

1. **Single Responsibility Principle**
   - Each class/function has only one responsibility
   - Only one reason to change

2. **Open-Closed Principle**
   - Open for extension (can add new features)
   - Closed for modification (no need to change existing code)
```

**Why Both Are Needed**:

- Domain-specific principles: Maintain project uniqueness
- Universal principles: Adhere to software engineering fundamentals

### 4. Include Coding Standards with Examples

Include these elements for each standard:

```markdown
### 1. Type Hints and Type Safety
- **Type hints are mandatory for all functions and methods**
- Comply with mypy strict mode (`strict = true`)

\`\`\`python
# Good example
def process_signal(data: NDArrayReal, sampling_rate: float) -> NDArrayComplex:
    ...

# Bad example
def process_signal(data, sampling_rate):  # No type hints
    ...
\`\`\`
```

**Important Elements**:

- ✅ Concise explanation
- ✅ Contrast "good" and "bad" examples
- ✅ Concrete code examples
- ✅ Emphasize key points in bold
- ❌ Abstract explanations only
- ❌ No code examples

### 5. Document Implementation Steps Progressively

```markdown
## Code Change Procedures

### 0. Check Existing Design Documents
- **Always check `docs/design/INDEX.md` before making changes**

### 1. Create Change Plan
- **Create a Markdown file with the change plan**
- Filename: `docs/design/working/plans/PLAN_<feature_name>.md`

### 2. Review Change Plan
Review from these perspectives **before** implementation:

#### Design Checklist
- [ ] Aligned with design principles?
- [ ] Backward compatibility maintained?
```

**Points**:

- ✅ Number steps explicitly
- ✅ Specific actions for each step
- ✅ Provide checklist format for verification
- ✅ Emphasize important notes in bold

### 6. Document Tools and Commands

```markdown
## Tools and Workflow

### Code Quality Checks
\`\`\`bash
# Lint and format with Ruff
uv run ruff check wandas tests --fix
uv run ruff format wandas tests

# Type check with mypy
uv run mypy --config-file=pyproject.toml

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=wandas --cov-report=html --cov-report=term
\`\`\`
```

**Purpose**:

- Provide immediately usable commands
- Standardize workflow
- Unify tool usage

## Writing Style Guide

### Language Choice

**Recommendation**: Match project's primary language

- Japanese projects: Write in Japanese
- International projects: Write in English
- Bilingual: Include both (but mind maintenance costs)

**Example from wandas project**:

- Guidelines: Japanese
- Code example docstrings: English
- Comments: Japanese and English as needed

### Using Emphasis

```markdown
# Good example
- **Type hints are mandatory for all functions and methods**
- **Always check `docs/design/INDEX.md` before making changes**

# Bad example (over-emphasis)
- **All****functions****and****methods****must****have****type****hints**
```

**Recommendations**:

- ✅ Bold for important actions or principles
- ✅ Use `backticks` for file names and commands
- ❌ Excessive emphasis (reduces readability)
- ❌ Making everything bold

### Clarity of Instructions

```markdown
# Good example (clear instructions)
- **Type hints are mandatory for all functions and methods**
- **Target 100% coverage** (minimum 90%)

# Bad example (vague instructions)
- Use type hints
- Keep coverage high
```

**Recommendations**:

- ✅ Show specific criteria
- ✅ Clearly distinguish "mandatory", "recommended", "optional"
- ✅ Document exceptions
- ❌ Vague expressions

## Providing Effective Examples

### 1. Code Example Structure

```markdown
### Good vs Bad Examples

\`\`\`python
# Good: Method chaining possible
signal = (
    wd.read_wav("audio.wav")
    .normalize()
    .low_pass_filter(cutoff=1000)
    .resample(target_rate=16000)
)

# Bad: No method chaining
signal = wd.read_wav("audio.wav")
normalize(signal)  # Implemented as function
low_pass_filter(signal, cutoff=1000)
\`\`\`
```

**Points**:

- ✅ Label "good" and "bad" with comments
- ✅ Contrast both
- ✅ Explain why good/bad

### 2. Implementation Pattern Examples

```markdown
### Error Handling

\`\`\`python
from pathlib import Path

def read_wav(filepath: Union[str, Path]) -> "ChannelFrame":
    """
    Read WAV file and create ChannelFrame.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the WAV file to read.
    
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is invalid or corrupted.
    """
    filepath = Path(filepath)
    
    # Input validation
    if not filepath.exists():
        raise FileNotFoundError(
            f"WAV file not found: {filepath}\\n"
            f"Please check the file path and try again."
        )
    
    # Processing
    ...
\`\`\`
```

**Points**:

- ✅ Provide complete implementation
- ✅ Include docstrings
- ✅ Show error handling patterns
- ✅ Explain intent with comments

### 3. Test Pattern Examples

```markdown
### Concrete Test Examples

\`\`\`python
# tests/processing/test_new_feature.py

def test_new_feature_normal_case():
    """Normal case: Verify basic behavior"""
    ...

def test_new_feature_with_edge_values():
    """Edge values: Verify behavior with min/max values"""
    ...

def test_new_feature_raises_error_on_invalid_input():
    """Error case: Verify error occurs with invalid input"""
    ...

def test_new_feature_preserves_metadata():
    """Verify metadata is preserved"""
    ...
\`\`\`
```

**Points**:

- ✅ Show multiple test patterns
- ✅ Test names clearly indicate intent
- ✅ Add explanatory comments

### 4. Utilize Checklists

```markdown
### Final Verification

#### Quality Checklist
- [ ] Verify all tests pass (`uv run pytest`)
- [ ] Verify 100% coverage achieved (`uv run pytest --cov`)
- [ ] Verify type checking passes (`uv run mypy --config-file=pyproject.toml`)
- [ ] Verify linting passes (`uv run ruff check wandas tests`)
```

**Points**:

- ✅ Use markdown checklist syntax
- ✅ Clearly list items to verify
- ✅ Include commands

## Maintenance

### 1. Regular Review

**Recommended Frequency**: Every 3-6 months

**Check Items**:

```markdown
## Regular Review Checklist

### Content Validity
- [ ] Any outdated information?
- [ ] Reflecting new tools or best practices?
- [ ] Matches current project state?

### Structure Improvement
- [ ] Any hard-to-read sections?
- [ ] Any duplicate content?
- [ ] Any information to add?

### Example Updates
- [ ] Code examples match latest API?
- [ ] Command examples work correctly?
```

### 2. Version Control

Document last update date at the beginning:

```markdown
# Wandas Project Development Guidelines

**Last Updated**: November 2, 2025

## Change History

- November 2, 2025: Added numerical verification principles
- October 18, 2025: Added document lifecycle rules
- October 1, 2025: Initial version
```

### 3. Team Sharing

```markdown
## About These Guidelines

These guidelines continue to evolve. If you have improvement suggestions:
1. Start discussion in Issues
2. Create change plan (`working/plans/PLAN_update_guidelines.md`)
3. Submit Pull Request
```

**Points**:

- ✅ Clarify feedback process
- ✅ Encourage continuous improvement
- ✅ Welcome team member contributions

### 4. File Size Management

**Warning Signs**:

- File exceeds 2000 lines
- Single section exceeds 500 lines
- Much duplicate content

**Solutions**:

1. **Modularize**: Separate by related topics
2. **External Links**: Link to separate documents for details
3. **Use Summaries**: Reduce verbose explanations

```markdown
# Good (summary + link)
### Testing Strategy
- Target 100% coverage
- See [Testing Strategy Guide](docs/design/guides/testing-strategy.md) for details

# Bad (include everything)
### Testing Strategy
(500 lines of detailed explanation)
```

## Common Pitfalls

### 1. Too Much Detail

❌ **Bad Example**: Document every function implementation

```markdown
### array_sum Function Implementation
\`\`\`python
def array_sum(arr: np.ndarray) -> float:
    """Calculate array sum"""
    return np.sum(arr)
\`\`\`

### array_mean Function Implementation
\`\`\`python
def array_mean(arr: np.ndarray) -> float:
    """Calculate array mean"""
    return np.mean(arr)
\`\`\`
(List all functions...)
```

✅ **Good Example**: Show patterns

```markdown
### NumPy Function Type Hints

Add type hints to all NumPy functions:

\`\`\`python
from wandas.utils.types import NDArrayReal

def array_operation(arr: NDArrayReal) -> float:
    """
    Perform operation on array.
    
    Parameters
    ----------
    arr : NDArrayReal
        Input array.
    
    Returns
    -------
    float
        Result value.
    """
    return np.some_operation(arr)
\`\`\`
```

### 2. Vague Instructions

❌ **Bad Example**:

```markdown
- Write clean code
- Pay attention to performance
```

✅ **Good Example**:

```markdown
- **Type hints are mandatory for all functions**
- **Use Dask lazy evaluation for large data processing**
```

### 3. No Examples

❌ **Bad Example**:

```markdown
### Method Chaining
Use method chaining.
```

✅ **Good Example**:

```markdown
### Method Chaining
\`\`\`python
# Good: Method chaining enabled
signal = (
    wd.read_wav("audio.wav")
    .normalize()
    .low_pass_filter(cutoff=1000)
)
\`\`\`
```

### 4. Outdated Information

❌ **Bad Example**:

```markdown
### Run Tests
\`\`\`bash
python -m pytest  # Old command
\`\`\`
```

✅ **Good Example**:

```markdown
### Run Tests
\`\`\`bash
# Latest project setup
uv run pytest
\`\`\`

**Last Verified**: November 2, 2025
```

### 5. Inconsistent Terminology

❌ **Bad Example**:

```markdown
- Use dataframe
- Create DataFrame
- Convert to data frame
```

✅ **Good Example**:

```markdown
- Use ChannelFrame (unified to project type name)
```

## References

### Official Documentation

- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [GitHub Copilot Custom Instructions](https://docs.github.com/en/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot)

### Related Guides

- [Wandas Design Document Index](../INDEX.md)
- [Document Lifecycle Management](../DOCUMENT_LIFECYCLE.md)
- [API Improvement Patterns](./api-improvements.md)
- [Metadata Encapsulation](./metadata-encapsulation.md)

### Real Example

- [Wandas copilot-instructions.md](../../../.github/copilot-instructions.md) - Comprehensive 970-line example

### Best Practice Articles

- [Effective Documentation Practices](https://www.writethedocs.org/guide/writing/style-guides/)
- [Google Developer Documentation Style Guide](https://developers.google.com/style)

## Summary

### Key Points

1. **Clear Structure**: Hierarchical section organization
2. **Concrete Examples**: Contrast good and bad examples
3. **Practical Procedures**: Step-by-step guides
4. **Regular Updates**: Review every 3-6 months
5. **Appropriate Size**: 500-1500 lines ideal

### Checklist

Checklist for creating new custom instructions:

```markdown
- [ ] Include project overview?
- [ ] Document domain-specific design principles?
- [ ] Include universal design principles (SOLID, etc.)?
- [ ] Provide concrete code examples?
- [ ] Contrast good and bad examples?
- [ ] Document implementation steps progressively?
- [ ] Document tools and commands?
- [ ] Include last update date?
- [ ] Limit scope with YAML front matter (if needed)?
- [ ] Within 500-1500 line range?
```

### Next Steps

1. Review existing `copilot-instructions.md`
2. Add missing information
3. Set regular update schedule
4. Collect team feedback
5. Implement continuous improvement

---

**Note**: This guide is based on experience from the wandas project. Please adjust according to your project characteristics.
