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
an invariant. Do not impose this procedure on a small, clear, local change.

## Workflow

1. Triage the risk and use direct execution when the guide classifies the task as small and clear.
2. For a triggered change, record the current contract, acceptance boundary, owners, scope, and stability.
3. Classify findings by invariant family and complete the guide's bounded sibling search before patching.
4. Reassess replan signals whenever the contract, architecture, ownership, compatibility, or scope moves materially.
5. Before external review, apply the guide's readiness checklist to the current
   head; capture the result in the task handoff or PR description when useful.

Return `CONTRACT_REPLAN_REQUIRED` and stop implementation when a material
boundary is unstable. Return `FIX_REQUIRED` only when the contract is stable and
the remaining work is bounded.

## Handoff

Report the outcome, contract revision, invariant families and sibling boundary
searched, scope or architecture drift, validation evidence, deferred tracking,
and residual risk. Keep GitHub operations in the selected vendor adapter.
