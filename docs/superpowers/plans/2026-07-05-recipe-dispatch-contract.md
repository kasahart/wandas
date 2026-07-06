# Recipe Dispatch Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close #265 by keeping `RecipeSpec.from_frame(...)` linear-only while making graph-boundary failures actionable and documenting the future automatic-dispatch path.

**Architecture:** `RecipeSpec.from_frame(...)` continues to call `_steps_from_graph(...)` and returns only `RecipeSpec`. Graph-boundary failures in `wandas/pipeline/extraction.py` will share one helper that points users to `GraphRecipeSpec.from_frame(...)` and `NodeGraphRecipeSpec.from_frame(...)`; explicit graph extractors remain unchanged. Docs will state that automatic dispatch is intentionally deferred and should be revisited through a future higher-level factory rather than by widening `RecipeSpec.from_frame(...)`.

**Tech Stack:** Python 3.10, Wandas recipe pipeline classes, pytest, ruff, ty, MkDocs.

---

## File Structure

- Modify `wandas/pipeline/extraction.py`
  - Add a small private helper for `RecipeSpec.from_frame(...)` graph-boundary errors.
  - Reuse it where linear extraction sees multi-input graph lineage or external graph operands.
- Modify `tests/pipeline/test_recipe.py`
  - Add focused tests that lock the linear-only return type and actionable graph guidance.
  - Update existing graph-boundary message assertions from the older terse wording.
- Modify `docs/src/how-to/pipeline-recipes.md`
  - Add beginner-facing text that `RecipeSpec.from_frame(...)` does not automatically return graph recipe types.
  - Keep explicit graph entry points as the recommended path.
- Modify `docs/src/explanation/pipeline-recipe-extraction-boundaries.md`
  - Replace the #265 follow-up wording with the chosen decision.
  - Keep future automatic dispatch as a possible higher-level factory, not a `RecipeSpec.from_frame(...)` behavior change.

No new public class, wrapper type, serialization schema, or recipe dispatch factory is added.

---

### Task 1: Pin The Linear-Only Dispatch Contract With Failing Tests

**Files:**
- Modify: `tests/pipeline/test_recipe.py`

- [ ] **Step 1: Add tests for the explicit graph extractor guidance**

Add these tests near the existing `test_recipe_from_frame_reports_graph_recipe_boundary_for_multi_input_operation` block:

```python
def test_recipe_from_frame_reports_explicit_graph_extractors_for_binary_merge() -> None:
    frame = _frame()
    noise = _frame().remove_dc()
    processed = frame.normalize() + noise

    with pytest.raises(RecipeExtractionError) as exc_info:
        RecipeSpec.from_frame(processed)

    message = str(exc_info.value)
    assert "RecipeSpec.from_frame(...) cannot extract graph lineage as a linear recipe" in message
    assert "RecipeSpec.from_frame(...) only supports single-input linear recipes" in message
    assert "GraphRecipeSpec.from_frame(...)" in message
    assert "NodeGraphRecipeSpec.from_frame(...)" in message
    assert isinstance(GraphRecipeSpec.from_frame(processed), GraphRecipeSpec)


def test_recipe_from_frame_reports_node_graph_extractor_for_external_operand() -> None:
    frame = _frame()
    processed = frame + np.ones(frame.shape)

    with pytest.raises(RecipeExtractionError) as exc_info:
        RecipeSpec.from_frame(processed)

    message = str(exc_info.value)
    assert "RecipeSpec.from_frame(...) cannot extract graph lineage as a linear recipe" in message
    assert "NodeGraphRecipeSpec.from_frame(...)" in message
    assert "external operands" in message
    assert isinstance(
        NodeGraphRecipeSpec.from_frame(processed, input_names=("signal", "offset")),
        NodeGraphRecipeSpec,
    )
```

- [ ] **Step 2: Strengthen the existing linear success test**

Replace the body of `test_recipe_from_frame_empty_history_returns_empty_recipe` with:

```python
def test_recipe_from_frame_empty_history_returns_linear_recipe() -> None:
    recipe = RecipeSpec.from_frame(_frame())

    assert isinstance(recipe, RecipeSpec)
    assert recipe.steps == ()
```

- [ ] **Step 3: Update existing message fragments that should now point to graph extractors**

In `test_recipe_from_frame_reports_current_boundary_for_non_replayable_operations`, change the `"binary frame operation"` expected message to:

```python
"RecipeSpec.from_frame(...) cannot extract graph lineage as a linear recipe"
```

In the `"array operand operation"` case, change the expected message to:

```python
"NodeGraphRecipeSpec.from_frame(...)"
```

In `test_recipe_from_frame_reports_graph_recipe_boundary_for_multi_input_operation`, change the assertion to:

```python
with pytest.raises(RecipeExtractionError, match=r"GraphRecipeSpec\.from_frame"):
    RecipeSpec.from_frame(processed)
```

In `test_recipe_from_frame_rejects_add_channel_boundary`, change the assertion to:

```python
with pytest.raises(RecipeExtractionError, match=r"NodeGraphRecipeSpec\.from_frame"):
    RecipeSpec.from_frame(processed)
```

- [ ] **Step 4: Run focused tests to verify red**

Run:

```bash
uv run pytest tests/pipeline/test_recipe.py \
  -k "explicit_graph_extractors or node_graph_extractor or current_boundary_for_non_replayable_operations or graph_recipe_boundary_for_multi_input_operation or add_channel_boundary or empty_history_returns_linear_recipe" \
  -q
```

Expected: FAIL because current errors still use older wording such as `Graph operation requires graph recipe support` or `Scalar operation requires a numeric scalar operand`.

- [ ] **Step 5: Commit the red tests**

```bash
git add tests/pipeline/test_recipe.py
git commit -m "test: pin recipe dispatch boundary guidance"
```

---

### Task 2: Centralize RecipeSpec Graph-Boundary Errors

**Files:**
- Modify: `wandas/pipeline/extraction.py`
- Test: `tests/pipeline/test_recipe.py`

- [ ] **Step 1: Add a private graph-boundary error helper**

In `wandas/pipeline/extraction.py`, after the imports and before `_validate_replayable_operation(...)`, add:

```python
def _recipe_spec_graph_boundary_error(
    operation: str,
    *,
    parent_count: int | None = None,
    runtime_inputs: int | None = None,
    detail: str | None = None,
) -> RecipeExtractionError:
    lines = [
        "RecipeSpec.from_frame(...) cannot extract graph lineage as a linear recipe",
        f"  Operation: {operation}",
    ]
    if parent_count is not None:
        lines.append(f"  Parent count: {parent_count}")
    if runtime_inputs is not None:
        lines.append(f"  Runtime inputs: {runtime_inputs}")
    if detail is not None:
        lines.append(f"  Detail: {detail}")
    lines.extend(
        [
            "  RecipeSpec.from_frame(...) only supports single-input linear recipes.",
            "  Use GraphRecipeSpec.from_frame(...) for one binary frame merge.",
            "  Use NodeGraphRecipeSpec.from_frame(...) for tree-shaped graph recipes, external operands, or add_channel inputs.",
        ]
    )
    return RecipeExtractionError("\n".join(lines))
```

- [ ] **Step 2: Replace the multi-input operation validation error**

In `_validate_replayable_operation(...)`, replace the `expected_input_count != 1` raise block with:

```python
        raise _recipe_spec_graph_boundary_error(operation, runtime_inputs=expected_input_count)
```

- [ ] **Step 3: Replace the scalar external operand boundary error**

In `_scalar_operand_from_params(...)`, replace the `params.get("operand_kind") != "operand"` raise block with:

```python
        raise _recipe_spec_graph_boundary_error(
            operation,
            detail="ScalarOperationStep only handles numeric scalar operands; external operands need graph inputs.",
        )
```

- [ ] **Step 4: Replace multidimensional parent graph boundary wording**

In `_multidimensional_steps_from_graph(...)`, replace the `len(parent_inputs) > 1` raise block with:

```python
        raise _recipe_spec_graph_boundary_error(
            str(parent["operation"]),
            parent_count=len(parent_inputs),
            detail="Multidimensional indexing requires one replayable parent chain.",
        )
```

- [ ] **Step 5: Replace the add_channel linear extraction boundary**

In `_steps_from_graph(...)`, replace the `operation == "add_channel"` raise block with:

```python
        raise _recipe_spec_graph_boundary_error(
            operation,
            detail="add_channel needs a graph recipe or external data/input reference.",
        )
```

- [ ] **Step 6: Replace the general multi-parent linear extraction boundary**

In `_steps_from_graph(...)`, replace the `len(inputs) > 1` raise block with:

```python
        raise _recipe_spec_graph_boundary_error(operation, parent_count=len(inputs))
```

- [ ] **Step 7: Run focused tests to verify green**

Run:

```bash
uv run pytest tests/pipeline/test_recipe.py \
  -k "explicit_graph_extractors or node_graph_extractor or current_boundary_for_non_replayable_operations or graph_recipe_boundary_for_multi_input_operation or add_channel_boundary or empty_history_returns_linear_recipe" \
  -q
```

Expected: PASS.

- [ ] **Step 8: Run the full recipe tests**

Run:

```bash
uv run pytest tests/pipeline/test_recipe.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit implementation and tests**

```bash
git add wandas/pipeline/extraction.py tests/pipeline/test_recipe.py
git commit -m "fix: clarify recipe graph dispatch boundary"
```

---

### Task 3: Document The Explicit Dispatch Decision

**Files:**
- Modify: `docs/src/how-to/pipeline-recipes.md`
- Modify: `docs/src/explanation/pipeline-recipe-extraction-boundaries.md`
- Test: docs build

- [ ] **Step 1: Update the how-to class selection guidance**

In `docs/src/how-to/pipeline-recipes.md`, after this paragraph:

```markdown
Most users should start with `RecipeSpec.from_frame(processed)`. Use this table only when the frame
result crosses the single-input linear boundary or an integration layer requires another API.
```

add:

```markdown
`RecipeSpec.from_frame(...)` is intentionally linear-only and does not automatically return `GraphRecipeSpec` or `NodeGraphRecipeSpec`. When a result has graph lineage, choose the graph recipe class explicitly. A future higher-level factory may add automatic dispatch without changing the `RecipeSpec.from_frame(...)` return type.
```

- [ ] **Step 2: Update the how-to unsupported boundary row**

In the unsupported boundary table, replace:

```markdown
| Multi-input work through `RecipeSpec.from_frame(...)` | Use `GraphRecipeSpec.from_frame(...)` or `NodeGraphRecipeSpec.from_frame(...)`. |
```

with:

```markdown
| Multi-input work through `RecipeSpec.from_frame(...)` | `RecipeSpec.from_frame(...)` stays linear-only; use `GraphRecipeSpec.from_frame(...)` or `NodeGraphRecipeSpec.from_frame(...)` explicitly. |
```

- [ ] **Step 3: Update the extraction-boundary follow-up list**

In `docs/src/explanation/pipeline-recipe-extraction-boundaries.md`, replace:

```markdown
- Automatic graph recipe dispatch from `RecipeSpec.from_frame(...)` is tracked by #265.
```

with:

```markdown
- Automatic graph recipe dispatch is intentionally deferred. `RecipeSpec.from_frame(...)` remains linear-only; a future higher-level factory may provide explicit union-return dispatch.
```

- [ ] **Step 4: Update the not-implemented graph bullet**

In the same explanation doc, replace:

```markdown
- `RecipeSpec.from_frame(frame_a + frame_b)` が graph recipe を返すこと。`RecipeSpec` は単一入力・直列 recipe のままとし、複数入力 graph は `GraphRecipeSpec.from_frame(...)` で抽出する。
```

with:

```markdown
- `RecipeSpec.from_frame(frame_a + frame_b)` が graph recipe を返すことは現時点では行わない。`RecipeSpec` は単一入力・直列 recipe のままとし、複数入力 graph は `GraphRecipeSpec.from_frame(...)` または `NodeGraphRecipeSpec.from_frame(...)` で明示的に抽出する。
```

Replace:

```markdown
- 入力名を推定する `RecipeSpec.from_frame(signal.add(noise, snr=6.0))` の自動 graph 抽出
```

with:

```markdown
- 将来の上位 factory による自動 graph 抽出。追加する場合も `RecipeSpec.from_frame(...)` の戻り値型は変えない。
```

- [ ] **Step 5: Build docs**

Run:

```bash
uv run mkdocs build -f docs/mkdocs.yml --strict --site-dir /tmp/wandas-docs-issue-265
```

Expected: PASS.

- [ ] **Step 6: Clean generated artifacts**

Run:

```bash
rm -rf .coverage .pytest_cache .ruff_cache coverage.xml docs/site tests/__pycache__ tests/pipeline/__pycache__ wandas/__pycache__ wandas/core/__pycache__ wandas/frames/__pycache__ wandas/frames/mixins/__pycache__ wandas/io/__pycache__ wandas/pipeline/__pycache__ wandas/processing/__pycache__ wandas/utils/__pycache__
```

Expected: `git status --short --ignored=matching` shows no generated artifacts.

- [ ] **Step 7: Commit docs**

```bash
git add docs/src/how-to/pipeline-recipes.md docs/src/explanation/pipeline-recipe-extraction-boundaries.md
git commit -m "docs: clarify recipe dispatch contract"
```

---

### Task 4: Final Verification And PR

**Files:**
- Verify all changed files.
- Update PR title/body after push.

- [ ] **Step 1: Run focused recipe tests**

```bash
uv run pytest tests/pipeline/test_recipe.py \
  -k "explicit_graph_extractors or node_graph_extractor or current_boundary_for_non_replayable_operations or graph_recipe_boundary_for_multi_input_operation or add_channel_boundary or empty_history_returns_linear_recipe" \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run full recipe tests**

```bash
uv run pytest tests/pipeline/test_recipe.py -q
```

Expected: PASS.

- [ ] **Step 3: Run docs build**

```bash
uv run mkdocs build -f docs/mkdocs.yml --strict --site-dir /tmp/wandas-docs-issue-265
```

Expected: PASS.

- [ ] **Step 4: Run lint**

```bash
uv run ruff check wandas tests --config=pyproject.toml
```

Expected: `All checks passed!`

- [ ] **Step 5: Run type checks**

```bash
uv run ty check wandas tests
```

Expected: `All checks passed!`

- [ ] **Step 6: Clean generated artifacts**

```bash
rm -rf .coverage .pytest_cache .ruff_cache coverage.xml docs/site tests/__pycache__ tests/pipeline/__pycache__ wandas/__pycache__ wandas/core/__pycache__ wandas/frames/__pycache__ wandas/frames/mixins/__pycache__ wandas/io/__pycache__ wandas/pipeline/__pycache__ wandas/processing/__pycache__ wandas/utils/__pycache__
git status --short --ignored=matching
```

Expected: no output from `git status --short --ignored=matching`.

- [ ] **Step 7: Push and create PR**

```bash
git push -u origin codex/issue-265-recipe-dispatch
gh pr create \
  --repo kasahart/wandas \
  --base develop \
  --head codex/issue-265-recipe-dispatch \
  --title "[codex] Clarify Recipe dispatch boundary" \
  --body "## Summary
- Keep RecipeSpec.from_frame(...) linear-only and improve graph-boundary guidance.
- Document that automatic graph dispatch is deferred to a future higher-level factory.
- Add tests for explicit GraphRecipeSpec / NodeGraphRecipeSpec guidance.

## Validation
- uv run pytest tests/pipeline/test_recipe.py -q
- uv run mkdocs build -f docs/mkdocs.yml --strict --site-dir /tmp/wandas-docs-issue-265
- uv run ruff check wandas tests --config=pyproject.toml
- uv run ty check wandas tests

Closes #265
Related #264
Related #257
Related #258"
```

- [ ] **Step 8: Confirm PR readiness**

Run:

```bash
git rev-parse HEAD
git rev-parse origin/codex/issue-265-recipe-dispatch
gh pr view --repo kasahart/wandas --json number,url,headRefOid,baseRefName,title,body,reviewDecision,reviewRequests,mergeable,statusCheckRollup
PR_NUMBER=$(gh pr view --repo kasahart/wandas --json number --jq .number)
gh api graphql -f owner=kasahart -f name=wandas -F number="$PR_NUMBER" -f query='query($owner:String!, $name:String!, $number:Int!) { repository(owner:$owner, name:$name) { pullRequest(number:$number) { reviewThreads(first:100) { nodes { isResolved isOutdated comments(first:5) { nodes { path line body } } } } } } }'
```

Expected:

- local `HEAD`, `origin/codex/issue-265-recipe-dispatch`, and PR `headRefOid` match;
- PR body includes `Closes #265`;
- no unresolved review threads;
- GitHub checks are passing or still pending within the monitoring timebox.
