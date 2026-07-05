# Recipe Intent Closure Design

## Goal

Close issue #259 by turning the replay-intent work into an explicit closure sweep, not a new broad Recipe feature.

#259 asks Wandas to clarify where Recipe extraction preserves user intent and where extraction must fail instead of flattening runtime details into misleading replay steps. PR #262 already documented and tested the riskiest boundaries. This follow-up should verify the acceptance criteria, fill only small documentation or test gaps, and move any remaining larger ideas into separate follow-up issues.

## Scope

This work covers:

- mapping each #259 acceptance criterion to existing docs, tests, and implementation;
- strengthening the user-facing closure record for `fix_length(duration=...)`, channel/time selection, `add_channel(...)`, and `add_with_snr(...)`;
- adding small focused docs or tests only where the closure audit finds an actual gap;
- creating follow-up issues for work that is useful but outside #259's replay-intent cleanup;
- preparing a PR that can use `Closes #259`.

This work does not add a new Recipe model or broaden extraction semantics.

## Closure Criteria

#259 can close when the PR can show the following table with evidence.

| #259 acceptance | Closure evidence |
| --- | --- |
| Accepted cases preserve intent and replay through frame APIs. | Extraction-boundary docs describe public replay APIs. Tests show replay calls preserve behavior for supported selections, method steps, graph steps, `add_channel`, and `add_with_snr`. |
| Unsupported cases fail with `RecipeExtractionError` instead of silently changing meaning. | Tests cover callable/regex channel queries, non-literal query values, unsupported multidimensional indexing, linear `RecipeSpec` graph boundaries, and `add_channel` boundaries. |
| Any new intent representation is documented and covered by focused tests. | `IndexingStep`, `MethodStep`, `GraphRecipeSpec`, `NodeGraphRecipeSpec`, `AddChannelStep`, `AddChannelDataStep`, `BinaryFrameStep`, and `BinaryOperandStep` are covered by docs and focused tests where introduced. |
| Recipe code does not duplicate numerical logic, metadata propagation, or Dask graph construction. | Docs state that Recipe replay delegates to frame APIs. Tests should avoid expecting Recipe-side reconstruction of waveform data, channel metadata, source-time updates, or Dask graphs. |

## Follow-Up Issues To Split Out

The closure sweep should create separate issues for these known future decisions if they are not already tracked:

| Follow-up | Why it is outside #259 |
| --- | --- |
| Recipe input-name inference for graph recipes. | Current runtime lineage does not store Python variable names or stable source identity. #259 only needs the explicit-input contract. |
| True DAG identity / shared branch graph recipes. | Current graph extraction replays tree-shaped parent paths. Shared identity is a graph-model enhancement, not an intent-boundary cleanup. |
| Automatic graph recipe dispatch from `RecipeSpec.from_frame(...)`. | Existing design keeps `RecipeSpec` as single-input linear extraction and uses `GraphRecipeSpec` / `NodeGraphRecipeSpec` for multi-input graphs. |
| WDF Recipe persistence. | Already tracked by #257. |
| sklearn/joblib/skops export. | Already tracked by #258. |

If an issue already exists for a follow-up, link it instead of creating a duplicate.

## Implementation Shape

1. Work from current `origin/develop` in a dedicated worktree.
2. Audit docs and tests against the closure criteria.
3. Add a concise #259 closure checklist to `docs/src/explanation/pipeline-recipe-extraction-boundaries.md` so the evidence is easy to find from the existing Recipe contract reference.
4. Add only narrow regression tests for uncovered acceptance gaps.
5. Create follow-up GitHub issues for any large future work that is not already tracked.
6. Open a PR with `Closes #259` and explicit `Related` links to the follow-up issues.

## Error Handling

No new error category is needed. Unsupported extraction remains `RecipeExtractionError`.

The audit should distinguish:

- invalid frame operations that fail before a processed frame exists; and
- valid processed frames that cannot be safely converted into a Recipe and must fail during extraction.

Only the second category is part of the Recipe extraction contract.

## Testing

The default test target is `tests/pipeline/test_recipe.py`.

Expected validation:

- focused pytest for any new or touched Recipe tests;
- `uv run pytest tests/pipeline/test_recipe.py -q`;
- `uv run mkdocs build -f docs/mkdocs.yml`;
- `uv run ruff check`;
- `uv run ty check` or a documented diff-scope `ty` check if known unrelated repository errors remain.

## PR And Issue Closure

The PR should use `Closes #259` only after the audit table is satisfied and large deferred work is tracked separately.

After merge, check #259. If GitHub does not close it automatically, add a concise closure comment and close it manually.
