# Development Documentation

This directory contains development guidelines and references for Wandas contributors.

## Error Message Guidelines

### Overview

The error message improvement initiative aims to make all error messages in Wandas user-friendly by following a 3-element structure:

1. **What** - Clearly state what went wrong
2. **Why** - Explain the constraint or condition that was violated  
3. **How** - Provide actionable guidance on resolving the issue

### Documents

- **[error_message_guide.md](error_message_guide.md)** - Comprehensive guidelines for writing error messages
  - Templates and examples for each error type
  - Quality checklist
  - Common patterns
  - Best practices

- **[error_message_analysis.md](error_message_analysis.md)** - Analysis of current error messages
  - Distribution by error type and module
  - Quality assessment
  - Priority classification
  - Specific findings and recommendations

- **[error_message_list.md](error_message_list.md)** - Complete list of all error messages
  - Organized by file and error type
  - Line numbers for easy reference
  - Useful for tracking improvements

- **[error_message_improvement_plan_phase2.md](error_message_improvement_plan_phase2.md)** - Implementation plan
  - High-priority targets (30 errors)
  - Week-by-week implementation strategy
  - Template application examples
  - Testing strategy
  - Success criteria

### Quick Start

1. **Before writing any error message**, read the [error_message_guide.md](error_message_guide.md)
2. **Use the template** from the guide to structure your error message
3. **Check the quality checklist** before committing
4. **Add tests** to verify error messages are helpful

### Example

```python
# Good: Follows 3-element structure
if cutoff <= 0 or cutoff >= nyquist:
    raise ValueError(
        f"Cutoff frequency must be between 0 Hz and {nyquist} Hz, got {cutoff} Hz.\n"
        f"The cutoff must be positive and less than Nyquist frequency to avoid aliasing.\n"
        f"Please provide a cutoff frequency in the valid range: 0 < cutoff < {nyquist}."
    )

# Bad: Missing elements
raise ValueError("Invalid cutoff")
```

## Other Development Topics

This directory will expand to include:
- Testing guidelines
- Performance optimization tips
- Documentation standards
- Code review checklists

---

For general development guidelines, see `.github/copilot-instructions.md`.
