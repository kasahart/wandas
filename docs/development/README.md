# Development Documentation

This directory contains documentation for wandas developers and contributors.

## Contents

### [Phase 1 Completion Summary](phase1_completion_summary.md)
Executive summary of Phase 1 investigation and guideline creation.

**Contents:**
- Complete analysis of 100 error messages
- Quality statistics and findings
- Tools and automation used
- Phase 2 and 3 planning
- Success metrics and recommendations

**When to Use:**
- Understanding the overall project status
- Planning Phase 2 implementation
- Reviewing what was accomplished
- Executive reporting

### [Error Message Guide](error_message_guide.md)
Comprehensive guidelines for writing user-friendly, actionable error messages.

**Topics Covered:**
- The 3-element rule (WHAT/WHY/HOW)
- Error message templates by exception type
- Examples of good vs bad error messages
- Implementation guidelines
- Testing error messages

**When to Use:**
- Writing new error messages
- Improving existing error messages
- Reviewing pull requests
- Understanding error message best practices

### [Error Improvement Examples](error_improvement_examples.md)
Practical before/after examples from actual wandas code.

**Contents:**
- 15+ real examples with improvements
- Common patterns (dimension, sampling rate, type, file, etc.)
- Implementation checklist
- Quick reference templates

**When to Use:**
- Implementing Phase 2 improvements
- Need concrete examples
- Quick pattern matching
- Copy-paste templates for similar cases

### [Error Inventory](error_inventory.md)
Complete inventory of all error messages in the codebase with quality analysis and prioritization.

**Contents:**
- 100 error messages categorized by priority
- Quality scores (0-3) for each error
- Module-level statistics
- Implementation roadmap for Phase 2

**When to Use:**
- Planning error message improvements
- Tracking progress on error message quality
- Identifying high-priority modules

## Quick Reference

### Error Message Template

```python
raise ValueError(
    f"<WHAT: Clear statement of the problem>\n"
    f"  Got: {actual_value}\n"
    f"  Expected: {expected_value}\n"
    f"<HOW: Actionable suggestion to fix>"
)
```

### Quality Checklist

- [ ] Message is in English
- [ ] Explains WHAT went wrong
- [ ] Explains WHY it's wrong (constraint/requirement)
- [ ] Provides HOW to fix it (actionable suggestion)
- [ ] Shows actual vs expected values
- [ ] Provides examples when helpful

## Related Documentation

- **[Copilot Instructions](../../.github/copilot-instructions.md)** - Main development guidelines
- **[Design Documents](../design/)** - Architecture and design decisions
- **[API Documentation](../src/api/)** - Public API documentation

## Contributing

When contributing to error message improvements:

1. Review the [Error Message Guide](error_message_guide.md)
2. Check the [Error Inventory](error_inventory.md) for priorities
3. Follow the 3-element rule (WHAT/WHY/HOW)
4. Update tests to verify error messages
5. Update this inventory after making changes

---

**Note**: This is a living documentation directory. Update these files as error handling patterns evolve.
