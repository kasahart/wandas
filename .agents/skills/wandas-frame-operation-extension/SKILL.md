---
name: wandas-frame-operation-extension
description: Use when planning, implementing, or reviewing a new Wandas Frame family, AudioOperation, public Frame method, Recipe-capable operation, or the tests and documentation for those extensions.
---

# Extend Wandas Frames and Operations

Treat `AGENTS.md` as the repository authority. Read the
[Frame and Operation extension guide](../../../docs/src/contributing/frame-operation-extensions.md)
completely before changing code or tests; keep detailed contracts in that guide rather
than copying them into this skill.

## Workflow

1. Classify the smallest extension using the guide's decision table.
2. Read the [`wandas-test-authoring` Skill](../wandas-test-authoring/SKILL.md),
   including its grand policy and the Frame and Processing references.
3. Map the change across the processing kernel, public Frame boundary, Recipe
   declaration when portable, exports, docstrings, and tests.
4. Implement with TDD while preserving Frame immutability, metadata and lineage,
   Dask laziness, and caller-owned input isolation.
5. Run the focused tests and repository gates specified by the guide.

## Handoff

Report the selected extension type, files changed, public and Recipe contracts,
tests by layer, validation commands, and any intentionally deferred work.

## Guardrails

- Do not create a new Frame when an existing domain model is sufficient.
- Do not put numerical algorithms in Frame methods.
- Do not duplicate parameters across operation state, metadata, and lineage.
- Do not add operation-specific branches to Recipe central layers.
- Do not treat private implementation tests as substitutes for public contracts.
