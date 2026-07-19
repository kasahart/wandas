---
name: wandas-change-coherence
description: Use for ambiguous, high-risk, cross-cutting, or review-driven Wandas changes where contract, ownership, invariant families, scope drift, or replan decisions must stay coherent.
---

# Maintain Change Coherence

Treat `AGENTS.md` as the repository authority. Read the
[change-coherence guide](../../../docs/src/contributing/change-coherence.md)
completely when this Skill is triggered; that guide owns the detailed procedure
and review dispositions.

## Trigger

Use this procedure when a change is ambiguous, high-risk, cross-cutting, changes
public or architectural boundaries, or receives related findings that may share
an invariant. Do not create its structured record solely for a small, clear,
local change.

## Workflow

1. Triage the risk and use direct execution when the guide classifies the task as small and clear.
2. For a triggered change, record the current contract, acceptance boundary, owners, scope, and stability.
3. Classify findings by invariant family and complete the guide's bounded sibling search before patching.
4. Reassess replan signals whenever the contract, architecture, ownership, compatibility, or scope moves materially.
5. For a triggered change, adapt the concise
   [example change record](references/change-record.example.json) to the current state.
6. Before external review, run the
   [change-coherence validator](../../../scripts/validate_change_coherence.py)
   against current-head evidence and require `REVIEW_READY`.

If the validator returns `CONTRACT_REPLAN_REQUIRED`, stop implementation and
return to contract planning. `SIBLING_SEARCH_REQUIRED` and `FIX_REQUIRED` are
implementation states, not review readiness.

## Handoff

Report the outcome, contract revision, invariant families and sibling boundary
searched, scope or architecture drift, validation evidence, deferred tracking,
and residual risk. Keep GitHub operations in the selected vendor adapter.
