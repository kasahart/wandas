# Recipe Intent Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close #259 by adding explicit closure evidence, splitting larger deferred Recipe graph decisions into follow-up issues, and opening a PR that can use `Closes #259`.

**Architecture:** Keep this as a documentation and issue-triage sweep. The existing Recipe implementation already delegates replay to frame APIs and has focused tests for the #259 replay-intent boundaries; this plan adds a concise closure checklist and follow-up issue links instead of broadening Recipe extraction.

**Tech Stack:** Python 3.10, Wandas Recipe docs, GitHub CLI, `uv run pytest`, `uv run mkdocs`, `uv run ruff`, `uv run ty`.

---

### Task 1: Add #259 Closure Checklist To Extraction Boundaries

**Files:**
- Modify: `docs/src/explanation/pipeline-recipe-extraction-boundaries.md`

- [ ] **Step 1: Insert the closure checklist after the "Replay Intent Rule" section**

Add this section after the bullet list under `## Replay Intent Rule / 再生意図のルール`:

```markdown
## Issue #259 Closure Checklist / #259 クローズ確認

Issue #259 is satisfied by the current replay-intent contract:

| #259 acceptance | Evidence |
| --- | --- |
| Accepted cases preserve intent and replay through frame APIs. | Supported linear, method, typed-method, scalar, graph, node-graph, `add_channel`, and `add_with_snr` recipes store public replay steps and call existing frame APIs. |
| Unsupported cases fail with `RecipeExtractionError` instead of silently changing meaning. | Callable/regex channel queries, non-literal query values, unsupported multidimensional indexing, linear `RecipeSpec` graph boundaries, and single-input `add_channel` extraction are rejected. |
| New intent representations are documented and covered by focused tests. | `MethodStep`, `IndexingStep`, `TypedMethodStep`, `ScalarOperationStep`, `BinaryFrameStep`, `BinaryOperandStep`, `AddChannelStep`, `AddChannelDataStep`, `GraphRecipeSpec`, and `NodeGraphRecipeSpec` are documented below and covered in `tests/pipeline/test_recipe.py`. |
| Recipe code does not duplicate numerical logic, metadata propagation, or Dask graph construction. | Recipe replay delegates to existing frame methods and operators. Recipe extraction stores replay intent, not waveform data, materialized arrays, metadata reconstruction logic, source-time update logic, or Dask graph construction. |

The remaining larger graph-model decisions are follow-up work:

- Recipe input-name inference for graph recipes.
- True DAG identity / shared branch graph recipes.
- Automatic graph recipe dispatch from `RecipeSpec.from_frame(...)`.

WDF Recipe persistence remains tracked by #257. Recipe interoperability and export remain tracked by #258.
```

- [ ] **Step 2: Run docs build**

Run:

```bash
uv run mkdocs build -f docs/mkdocs.yml
```

Expected: `Documentation built` and exit code 0.

- [ ] **Step 3: Commit the docs checklist**

Run:

```bash
git add docs/src/explanation/pipeline-recipe-extraction-boundaries.md
git commit -m "docs: add recipe intent closure checklist"
```

Expected: one docs commit. Pre-commit may skip Python hooks because only markdown changed.

### Task 2: Create Follow-Up Issues For Deferred Graph Decisions

**Files:**
- No repository files changed.
- GitHub issues created in `kasahart/wandas`.

- [ ] **Step 1: Verify the three follow-ups do not already exist**

Run:

```bash
gh issue list --repo kasahart/wandas --state all --search "recipe input name inference" --json number,title,state,url --limit 20
gh issue list --repo kasahart/wandas --state all --search "recipe DAG identity" --json number,title,state,url --limit 20
gh issue list --repo kasahart/wandas --state all --search "RecipeSpec graph recipe" --json number,title,state,url --limit 20
```

Expected: no open issue directly covering these exact follow-ups. If an exact issue appears, record its number and do not create a duplicate.

- [ ] **Step 2: Create input-name inference issue if missing**

Run:

```bash
gh issue create --repo kasahart/wandas \
  --title "Define Recipe graph input-name inference contract" \
  --body "## Background

#259 closed the replay-intent extraction contract using explicit graph input names. Runtime lineage currently does not know Python variable names or stable source identity, so graph recipes require caller-provided names or structural defaults.

## Scope

- Decide whether Wandas should infer graph recipe input names from runtime metadata, frame labels, source identifiers, or another explicit API.
- Define when inferred names are stable enough to serialize.
- Keep explicit \`input_names=(...)\` as the reliable baseline.

## Acceptance

- The naming source is documented.
- Ambiguous or unstable names fail clearly or require explicit names.
- Tests cover inferred names, duplicate names, and fallback behavior.

## Non-goals

- WDF persistence; see #257.
- sklearn/export; see #258.
- True DAG identity; track separately.

Related #259."
```

Expected: GitHub prints the new issue URL. Save the issue number for the PR body.

- [ ] **Step 3: Create true DAG identity issue if missing**

Run:

```bash
gh issue create --repo kasahart/wandas \
  --title "Define true DAG identity for Recipe graph extraction" \
  --body "## Background

#259 closed the replay-intent boundary using the current tree-shaped operation graph. Shared branches are replayed as duplicated parent paths, which is explicit but does not preserve true shared identity.

## Scope

- Decide whether runtime lineage should preserve shared node identity.
- Define how graph recipes represent shared branches without duplicating work or changing replay meaning.
- Preserve Dask laziness and keep numerical logic in frame operations.

## Acceptance

- Shared branch behavior is documented.
- Extraction and replay tests cover shared and duplicated branches.
- Existing tree-shaped graph recipes keep working.

## Non-goals

- Recipe input-name inference.
- WDF persistence; see #257.
- sklearn/export; see #258.

Related #259."
```

Expected: GitHub prints the new issue URL. Save the issue number for the PR body.

- [ ] **Step 4: Create automatic graph dispatch issue if missing**

Run:

```bash
gh issue create --repo kasahart/wandas \
  --title "Decide automatic graph recipe dispatch from RecipeSpec.from_frame" \
  --body "## Background

#259 keeps \`RecipeSpec.from_frame(...)\` as the single-input linear extractor. Multi-input calculations use \`GraphRecipeSpec.from_frame(...)\` or \`NodeGraphRecipeSpec.from_frame(...)\` so the caller chooses the graph recipe model and provides input names when needed.

## Scope

- Decide whether \`RecipeSpec.from_frame(...)\` should ever return or delegate to graph recipe types.
- Define the public API and type contract if automatic dispatch is added.
- Keep error messages clear when users call the wrong extractor.

## Acceptance

- The chosen API is documented.
- Type expectations are covered by tests.
- Existing explicit graph recipe entry points remain supported.

## Non-goals

- Recipe input-name inference.
- True DAG identity.
- WDF persistence; see #257.
- sklearn/export; see #258.

Related #259."
```

Expected: GitHub prints the new issue URL. Save the issue number for the PR body.

### Task 3: Update Docs With Follow-Up Issue Links

**Files:**
- Modify: `docs/src/explanation/pipeline-recipe-extraction-boundaries.md`

- [ ] **Step 1: Capture the follow-up issue numbers by title**

Run:

```bash
input_name_issue=$(gh issue list --repo kasahart/wandas --state open --search "Define Recipe graph input-name inference contract in:title" --json number --jq '.[0].number')
dag_identity_issue=$(gh issue list --repo kasahart/wandas --state open --search "Define true DAG identity for Recipe graph extraction in:title" --json number --jq '.[0].number')
graph_dispatch_issue=$(gh issue list --repo kasahart/wandas --state open --search "Decide automatic graph recipe dispatch from RecipeSpec.from_frame in:title" --json number --jq '.[0].number')
printf 'input_name_issue=%s\ndag_identity_issue=%s\ngraph_dispatch_issue=%s\n' "$input_name_issue" "$dag_identity_issue" "$graph_dispatch_issue"
```

Expected: all three variables print non-empty issue numbers.

- [ ] **Step 2: Replace the unnumbered follow-up bullets**

Run:

```bash
perl -0pi -e 's/- Recipe input-name inference for graph recipes\\./- Recipe input-name inference for graph recipes is tracked by #'"$input_name_issue"'./' docs/src/explanation/pipeline-recipe-extraction-boundaries.md
perl -0pi -e 's/- True DAG identity \\/ shared branch graph recipes\\./- True DAG identity \\/ shared branch graph recipes are tracked by #'"$dag_identity_issue"'./' docs/src/explanation/pipeline-recipe-extraction-boundaries.md
perl -0pi -e 's/- Automatic graph recipe dispatch from `RecipeSpec\\.from_frame\\(\\.\\.\\.\\)`\\./- Automatic graph recipe dispatch from `RecipeSpec.from_frame(...)` is tracked by #'"$graph_dispatch_issue"'./' docs/src/explanation/pipeline-recipe-extraction-boundaries.md
```

Expected: the three follow-up bullets now contain concrete GitHub issue references using the numbers printed in Step 1.

- [ ] **Step 3: Run docs build**

Run:

```bash
uv run mkdocs build -f docs/mkdocs.yml
```

Expected: `Documentation built` and exit code 0.

- [ ] **Step 4: Commit the follow-up links**

Run:

```bash
git add docs/src/explanation/pipeline-recipe-extraction-boundaries.md
git commit -m "docs: link recipe intent follow-up issues"
```

Expected: one docs commit.

### Task 4: Verify Existing Recipe Tests Cover Closure Criteria

**Files:**
- No source changes expected.

- [ ] **Step 1: Run focused Recipe tests**

Run:

```bash
uv run pytest tests/pipeline/test_recipe.py -q
```

Expected: all tests in `tests/pipeline/test_recipe.py` pass.

- [ ] **Step 2: Check formatting and docs whitespace**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 3: Run lint and type checks**

Run:

```bash
uv run ruff check
uv run ty check
```

Expected:

- `uv run ruff check` passes.
- If full `uv run ty check` still reports unrelated diagnostics in `learning-path/07_frame_centric_recipe_ux.py`, run this diff-scope command and record both results:

```bash
uv run ty check wandas tests/pipeline/test_recipe.py
```

Expected for diff-scope: `All checks passed!`.

- [ ] **Step 4: Clean generated artifacts**

Run:

```bash
rm -rf .coverage .pytest_cache .ruff_cache coverage.xml docs/site tests/__pycache__ tests/pipeline/__pycache__ wandas/__pycache__ wandas/core/__pycache__ wandas/frames/__pycache__ wandas/frames/mixins/__pycache__ wandas/io/__pycache__ wandas/pipeline/__pycache__ wandas/processing/__pycache__ wandas/utils/__pycache__
git status --short --branch --ignored
```

Expected: no generated artifacts remain. Only intentional tracked changes should appear.

### Task 5: Open The Closure PR

**Files:**
- No additional repository changes.

- [ ] **Step 1: Push the branch**

Run:

```bash
git push -u origin codex/issue-259-closure
```

Expected: branch is pushed to origin.

- [ ] **Step 2: Capture follow-up issue numbers by title**

Run:

```bash
input_name_issue=$(gh issue list --repo kasahart/wandas --state open --search "Define Recipe graph input-name inference contract in:title" --json number --jq '.[0].number')
dag_identity_issue=$(gh issue list --repo kasahart/wandas --state open --search "Define true DAG identity for Recipe graph extraction in:title" --json number --jq '.[0].number')
graph_dispatch_issue=$(gh issue list --repo kasahart/wandas --state open --search "Decide automatic graph recipe dispatch from RecipeSpec.from_frame in:title" --json number --jq '.[0].number')
printf 'input_name_issue=%s\ndag_identity_issue=%s\ngraph_dispatch_issue=%s\n' "$input_name_issue" "$dag_identity_issue" "$graph_dispatch_issue"
```

Expected: all three variables print non-empty issue numbers.

- [ ] **Step 3: Create the PR**

Run:

```bash
gh pr create --repo kasahart/wandas \
  --base develop \
  --head codex/issue-259-closure \
  --title "[codex] Close Recipe replay intent contract" \
  --body "## Summary
- Add a #259 closure checklist to the Recipe extraction-boundaries docs.
- Split larger deferred graph-model decisions into dedicated follow-up issues.
- Keep Recipe replay intent scoped to documented frame-API delegation instead of broadening extraction behavior.

## Issue Relationship
Closes #259

Related #257
Related #258
Related #${input_name_issue}
Related #${dag_identity_issue}
Related #${graph_dispatch_issue}

## Validation
- \`uv run pytest tests/pipeline/test_recipe.py -q\`
- \`uv run mkdocs build -f docs/mkdocs.yml\`
- \`uv run ruff check\`
- \`uv run ty check\` or documented diff-scope \`uv run ty check wandas tests/pipeline/test_recipe.py\`
- \`git diff --check\`"
```

Expected: GitHub prints the new PR URL.

- [ ] **Step 4: Confirm PR readiness state**

Run:

```bash
gh pr view --repo kasahart/wandas --json number,state,isDraft,url,headRefOid,statusCheckRollup,reviewRequests,reviewDecision
git rev-parse HEAD origin/codex/issue-259-closure
```

Expected: PR is open, not draft unless intentionally created as draft, local and remote branch SHAs match, and checks are either queued or successful.
