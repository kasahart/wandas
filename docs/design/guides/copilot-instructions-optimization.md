# GitHub Copilot Instructions Optimization

**Date**: 2025-11-05  
**Status**: Implemented  
**PR**: [Link to PR]

## Summary

Optimized GitHub Copilot instructions from 36KB monolithic Japanese file to 10KB modular English-first structure with path-specific guidance. Achieved 72% size reduction while improving clarity, international accessibility, and AI effectiveness.

## Problem

The original `.github/copilot-instructions.md` had several critical issues:

1. **Excessive length**: 36KB, 1,012 lines, ~2,560 words
2. **Language barrier**: Primarily Japanese (299 lines), limiting international contributions
3. **Anti-patterns**: Duplicated CI/CD configuration instead of referencing it
4. **Lack of modularity**: Single file for all guidance, no path-specific instructions
5. **GitHub best practices violation**: Recommended <500 chars per instruction

## Solution

### Core Restructuring

**Before** (36KB):
- Single monolithic file
- Mixed Japanese/English
- All guidance in one place
- Examples embedded inline
- CI/CD commands duplicated

**After** (10KB):
- Concise core principles
- English-first
- Path-specific sections (Python, tests, docs)
- References to comprehensive docs
- CI/CD referenced, not duplicated

### Key Improvements

1. **Size Reduction**: 36KB → 10KB (72% reduction)
2. **Language**: 0 Japanese lines (from 299)
3. **Modularity**: 3 path-specific sections using `applyTo` frontmatter
4. **References**: Points to detailed documentation instead of duplicating
5. **Clarity**: Focused on critical rules and patterns

### New File Structure

```
.github/
├── copilot-instructions.md              # 10KB: Core + path-specific
└── copilot-instructions.md.original     # Backup (gitignored)

docs/development/
├── coding_standards.md                  # NEW: Comprehensive coding guide
├── testing_guide.md                     # NEW: Testing best practices
├── error_message_guide.md               # Existing, referenced
└── README.md
```

## Implementation Details

### Core Instructions Structure

1. **About Wandas** (1 paragraph)
2. **Core Principles** (4 critical rules)
3. **Coding Standards** (type hints, operations, errors, docs)
4. **Testing Requirements** (structure, coverage)
5. **Code Change Workflow** (TDD approach)
6. **Tools and Commands** (reference only)
7. **References** (point to detailed docs)

### Path-Specific Sections

Using `applyTo` frontmatter for targeted guidance:

```markdown
---
applyTo: "**/*.py"
---
# Python-Specific Guidelines
- Array handling (Dask/NumPy)
- Metadata management
- Performance optimization
```

```markdown
---
applyTo: "tests/**/*.py"
---
# Test-Specific Guidelines
- Theoretical validation (critical)
- Metadata verification
- Independence requirements
```

```markdown
---
applyTo: "docs/**/*.md"
---
# Documentation Guidelines
- English, runnable examples
- Clear structure, cross-references
```

### Supporting Documentation

Created comprehensive reference docs:

1. **coding_standards.md** (18KB):
   - Type hints and type safety
   - Array handling (NumPy/Dask)
   - Signal processing operations
   - Metadata management
   - Performance optimization

2. **testing_guide.md** (16KB):
   - Testing philosophy
   - Numerical validation patterns
   - Fixtures and test data
   - Coverage requirements
   - Performance testing

## Design Decisions

### English-First Approach

**Decision**: Translate all instructions to English

**Rationale**:
- GitHub Copilot AI trained primarily on English
- Welcomes international contributors
- Aligns with GitHub best practices
- Technical terms clearer in English

**Trade-off**: Original Japanese developers may need adaptation period

**Mitigation**: 
- Original Japanese file preserved as backup
- Can create docs/ja/ in future if needed
- Code quality improvements outweigh transition cost

### Path-Specific vs. Separate Files

**Decision**: Use path-specific sections in single file rather than multiple files

**Rationale**:
- Single file easier to maintain
- GitHub Copilot processes frontmatter sections efficiently
- Can split later if file grows too large
- Reduces file management overhead

**Trade-off**: File still 10KB (target was <2KB)

**Mitigation**:
- Still 72% reduction from original
- Clear section separation
- References keep each section concise
- Can further optimize in future if needed

### Reference Documentation Strategy

**Decision**: Create comprehensive reference docs instead of inline examples

**Rationale**:
- Copilot instructions focus on "what" (principles)
- Reference docs explain "how" (implementation)
- Reduces duplication
- Easier to maintain and update
- Searchable documentation

**Trade-off**: Developers need to navigate to references

**Mitigation**:
- Clear links in instructions
- Quick reference sections preserved
- Common patterns still in core instructions

## Results

### Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| File Size | 36KB | 10KB | 72% reduction |
| Lines | 1,012 | 363 | 64% reduction |
| Japanese Lines | 299 | 0 | 100% removal |
| Path-Specific Sections | 0 | 3 | ∞ increase |
| Reference Docs | 1 | 3 | 200% increase |

### Qualitative Improvements

1. **Better AI Understanding**: Shorter, focused instructions improve Copilot context
2. **International Accessibility**: English-first welcomes global contributors
3. **Maintainability**: Modular structure easier to update
4. **Discoverability**: Path-specific guidance shows only when relevant
5. **Comprehensiveness**: Reference docs provide depth without bloat

## Lessons Learned

### What Worked Well

1. **Incremental approach**: Created multiple versions (moderate, minimal) before deciding
2. **Measurement**: Tracked metrics (size, lines, coverage) throughout
3. **Preservation**: Kept original file as backup
4. **Documentation**: Created comprehensive reference docs simultaneously

### What Could Be Improved

1. **Could be more concise**: 10KB still above 2KB target
2. **More prompts**: Only created improvement plan, could add .prompt files
3. **Team review**: Should get feedback before finalizing
4. **A/B testing**: Should test with actual Copilot interactions

### Future Optimizations

1. **Further reduction**: Target 5KB by removing more examples
2. **Prompt files**: Create .github/prompts/ for common tasks
3. **Japanese docs**: Create docs/ja/ for Japanese-speaking developers
4. **CI enforcement**: Add linters for instruction compliance
5. **Metrics tracking**: Monitor Copilot effectiveness over time

## Related Patterns

### Similar to API Improvements

Like the API improvements guide, this follows the principle of:
- Clear, explicit interfaces
- Reduce cognitive load
- Point to comprehensive docs for details

### Extends Error Message Guide

The error message patterns in this optimization complement the existing error_message_guide.md by:
- Providing concise reference in instructions
- Detailed patterns in guide
- Consistent 3-element rule across both

## Migration Guide

For developers familiar with the old instructions:

### Key Changes

1. **Language**: Japanese → English (original preserved in .original file)
2. **Structure**: Monolithic → Modular (path-specific sections)
3. **Examples**: Inline → Referenced (see coding_standards.md)
4. **Commands**: Duplicated → Referenced (see tools section)

### Where to Find Things

| Old Location | New Location |
|-------------|--------------|
| コーディング規約 | Coding Standards → coding_standards.md |
| テスト | Testing Requirements → testing_guide.md |
| エラーハンドリング | Error Messages → error_message_guide.md |
| ツール | Tools and Commands (brief) |
| ベストプラクティス | Coding Standards / Testing Guide |

### Adapting Your Workflow

1. **Quick reference**: Use `.github/copilot-instructions.md`
2. **Deep dive**: Refer to docs/development/ guides
3. **Language**: English for code, docs, PRs
4. **Path-specific**: Notice different guidance in different files

## References

- **Improvement Plan**: docs/design/working/drafts/COPILOT_INSTRUCTIONS_IMPROVEMENT_PLAN.md
- **Coding Standards**: docs/development/coding_standards.md
- **Testing Guide**: docs/development/testing_guide.md
- **Error Message Guide**: docs/development/error_message_guide.md
- **GitHub Copilot Docs**: https://docs.github.com/en/copilot/customizing-copilot

---

**Implementation Note**: This optimization aligns with GitHub's best practices and significantly improves maintainability and international accessibility while preserving all critical guidance.
