---
name: wandas-learning-material-authoring
description: Create, revise, or review repository-specific Wandas learning materials, especially executable marimo notebooks under learning-path/ and the summaries or examples that point to them from README, tutorial, and API documentation. Use when defining teaching goals, structuring narratives and executable examples, choosing learner-appropriate representations and public APIs, or validating pedagogical clarity and technical accuracy.
---

# Wandas Learning Material Authoring

Treat `AGENTS.md` as the repository source of truth. Use `wandas-workspace-hygiene` before editing and before reporting completion.

## Define the teaching contract

Before editing, state:

- who the learner is and what they can already assume;
- the primary capability or mental model, plus any small number of supporting outcomes;
- the observable evidence that demonstrates that outcome;
- which details are supporting context and which are outside the lesson.

Use this contract to remove content that is correct but not useful to the lesson. Do not turn setup, an example dataset, a file format, or an implementation detail into a teaching objective unless the material is specifically about it.

## Build the narrative

1. Start with the learner's problem and the value of solving it.
2. Introduce the minimum mental model needed to understand the workflow.
3. Show the smallest representative example that demonstrates the intended capability.
4. Make the result visibly prove the claim.
5. Explain the broader implication without expanding into unrelated features.

Make headings state learner outcomes rather than incidental actions. Use generic language unless real domain semantics materially improve understanding. Avoid unexplained acronyms, product-specific context, and command-runner instructions that are not part of the learning goal.

Keep supporting README, tutorial indexes, and API examples consistent with the lesson's terminology and contract. When editing those supporting references, optimize for discoverability, scope accuracy, and API-contract precision; do not force the lesson narrative onto reference material.

## Choose representations deliberately

Select each representation by what the learner needs to notice:

- Use prose for motivation, assumptions, and interpretation.
- Use code only for behavior the learner must understand or reproduce.
- Use a table when rows, fields, or comparisons are the point.
- Use a diagram, tree, or compact structure when relationships or hierarchy are the point.
- Use output when it provides evidence for a claim; omit output that merely repeats execution details.

Do not encode static explanation as executable code. Do not prescribe a particular library or visual form when another representation communicates the concept more directly.

## Design marimo cells

- Teach one idea per Python cell. Split cells when their statements support different learning claims, not merely because the operations differ.
- Add a short purpose comment, in the material's language, when a visible cell's teaching role is not already clear from its code and surrounding prose.
- Keep operations together when their combination proves one claim, such as a meaningful before/after comparison.
- Give every cross-cell variable exactly one definition. Use a descriptive unique name, or prefix disposable cell-local values with `_`.
- Run `uv run marimo check <notebook>` after adding, splitting, merging, or renaming cells. Do not rely only on export execution to detect reactive-definition conflicts.
- Make output explain what it proves. Avoid unexplained counts, dumps, and intermediate state.

## Choose examples at the right abstraction

- Demonstrate the primary public workflow a learner should normally adopt.
- Prefer an API at the same scope as the learner's task. Do not make a collection workflow look like repetitive per-item work when Wandas provides a collection-level operation.
- Avoid compatibility helpers, internal state, and lower-level materialization APIs unless they are the subject of the lesson.
- Introduce laziness, metadata propagation, history, or other execution contracts only when they explain an observable behavior or prevent a likely misunderstanding.
- Preserve frame immutability, metadata/history, and Dask laziness in every example, even when those properties are not discussed.
- Verify API behavior against current implementation, tests, and nearby documentation instead of relying on memory.

## Keep examples inspectable

- Generate the smallest self-contained sample needed by the lesson.
- Make important inputs and relationships visible using the representation appropriate to the concept.
- Use names that reveal roles without importing unnecessary domain assumptions.
- Keep sample values and expected output small enough to verify visually.
- Ensure the example feels easier than the manual approach it replaces; accidental boilerplate teaches the wrong product model.

## Validate

Run checks proportional to the touched files:

```bash
uv run marimo check learning-path/<notebook>.py
uv run marimo export html learning-path/<notebook>.py -o /tmp/<notebook>.html --no-include-code -f
uv run ruff check learning-path/<notebook>.py --config=pyproject.toml
uv run ty check learning-path/<notebook>.py
```

Also:

- Inspect exported output for the intended evidence and absence of cell errors.
- Run `git diff --check`.
- Run `uv run mkdocs build -f docs/mkdocs.yml` when MkDocs-backed documentation changes.
- Search touched files for obsolete terminology and APIs.
- Record ignored/generated artifacts for the single cleanup defined by
  `wandas-workspace-hygiene`: after merge for pull-request work, or at final
  handoff when no pull request is used.

Before finishing, ask:

- Does every section advance the teaching contract?
- Does each code cell communicate one claim?
- Is the chosen representation the simplest way to reveal that claim?
- Does the example demonstrate the intended Wandas workflow rather than incidental mechanics?
- Can the learner tell from the output why the example succeeded?

Report the changed teaching contract and validation evidence, not a cell-by-cell implementation diary.
