---
name: wandas-test-authoring
description: Use when planning, implementing, or reviewing Wandas tests, including shared test design and tests for Frames, processing, I/O, visualization, or changes that span multiple test domains.
---

# Author Wandas Tests

Treat `AGENTS.md` as the repository authority. Always read the
[grand policy](references/grand-policy.md), then read every domain reference
that matches the changed behavior:

- [Frames](references/frames.md) for Frame contracts, domain transitions, metadata, and laziness.
- [Processing](references/processing.md) for numerical kernels, acoustic validity, and authority comparisons.
- [I/O](references/io.md) for readers, writers, round-trips, normalization, metadata, and lazy loading.
- [Visualization](references/visualization.md) for dispatch, Axes behavior, parameter forwarding, and figure cleanup.

Read multiple references when a change crosses boundaries. For example, a new
Frame method backed by a processing operation requires both the Frames and
Processing references.

## Workflow

1. Identify the public behavior, touched layers, failure modes, and domain transitions.
2. Map the change onto the grand policy's four pillars and test pyramid; mark non-applicable concerns explicitly.
3. Select deterministic fixtures and independent expected values before implementing the test.
4. Add the smallest set of tests that covers the public contract, relevant errors, metadata or lineage, and Dask laziness.
5. Run focused tests first, then the relevant repository gates from `AGENTS.md`.
6. Report test layers covered, validation commands, skipped checks with reasons, and any concrete follow-up gap.

## Guardrails

- Keep detailed contracts in the references rather than duplicating them here.
- Prefer public behavior and independent authorities over private implementation details or self-referential expectations.
- Do not add unexplained tolerances, random signals where analytical signals work, or assertions that only prove no exception occurred.
- Preserve existing detailed policy unless the task explicitly changes the testing contract.
