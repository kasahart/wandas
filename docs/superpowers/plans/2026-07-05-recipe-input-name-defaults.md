# Recipe Input Name Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close #263 by making omitted graph Recipe input names consistently default to `input_0`, `input_1`, ... instead of mixing `left`/`right` with `input_n`.

**Architecture:** Keep name generation local to graph Recipe extraction. `GraphRecipeSpec.from_frame(..., input_names=None)` should use the same mechanical naming style already used by `NodeGraphRecipeSpec.from_frame(...)`; explicit `input_names` remains unchanged and remains the recommended user-facing path.

**Tech Stack:** Python 3.10, Wandas `wandas.pipeline.specs`, pytest, MkDocs, `uv run ruff`, `uv run ty`.

---

## File Structure

- `wandas/pipeline/specs.py`: change the default name resolution in `GraphRecipeSpec.from_frame(...)`.
- `tests/pipeline/test_recipe.py`: update default-name tests and add one non-commutative order test.
- `docs/src/explanation/pipeline-recipe-extraction-boundaries.md`: document `input_0` / `input_1` default names and no semantic inference.
- `docs/src/how-to/pipeline-recipes.md`: update user-facing guidance for omitted and explicit `input_names`.
- `docs/superpowers/specs/2026-07-05-recipe-input-name-defaults-design.md`: already created design reference.

### Task 1: Pin GraphRecipeSpec Default Input Names

**Files:**
- Modify: `tests/pipeline/test_recipe.py`

- [ ] **Step 1: Rename and update the existing GraphRecipeSpec default-name test**

Replace the full body of `test_graph_recipe_from_frame_uses_default_structural_input_names` with:

```python
def test_graph_recipe_from_frame_uses_numbered_default_input_names() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    right_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="noise")
    processed = left_source.remove_dc() + right_source.high_pass_filter(cutoff=500.0)

    graph_recipe = GraphRecipeSpec.from_frame(processed)
    replayed = graph_recipe.apply({"input_0": left_source, "input_1": right_source})

    assert graph_recipe.input_recipes == (
        ("input_0", RecipeSpec([OperationSpec("remove_dc")])),
        ("input_1", RecipeSpec([OperationSpec("highpass_filter", {"cutoff": 500.0, "order": 4})])),
    )
    assert graph_recipe.output == BinaryFrameStep("+", "input_0", "input_1")
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history
```

- [ ] **Step 2: Update raw add-with-SNR default-name test**

Replace the full body of `test_graph_recipe_from_frame_uses_default_names_for_raw_add_with_snr` with:

```python
def test_graph_recipe_from_frame_uses_numbered_default_names_for_raw_add_with_snr() -> None:
    base = _frame()
    signal_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    noise_source = ChannelFrame.from_numpy(np.flip(base.data), sampling_rate=base.sampling_rate, label="noise")
    processed = signal_source.add(noise_source, snr=6.0)

    graph_recipe = GraphRecipeSpec.from_frame(processed)
    replayed = graph_recipe.apply({"input_0": signal_source, "input_1": noise_source})

    assert graph_recipe.input_recipes == (("input_0", RecipeSpec(())), ("input_1", RecipeSpec(())))
    assert graph_recipe.output == BinaryFrameStep("add_with_snr", "input_0", "input_1", {"snr": 6.0})
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history
```

- [ ] **Step 3: Update typed-tail default-name test**

Replace the default-name assertions and replay call inside `test_graph_recipe_from_frame_uses_default_names_with_typed_tail` so the function body becomes:

```python
def test_graph_recipe_from_frame_uses_numbered_default_names_with_typed_tail() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    right_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="noise")
    processed = (left_source + right_source).stft(
        n_fft=512,
        hop_length=128,
        win_length=512,
        window="hann",
    )

    graph_recipe = GraphRecipeSpec.from_frame(processed)
    replayed = graph_recipe.apply({"input_0": left_source, "input_1": right_source})

    assert graph_recipe.input_recipes == (("input_0", RecipeSpec(())), ("input_1", RecipeSpec(())))
    assert graph_recipe.output == BinaryFrameStep("+", "input_0", "input_1")
    assert graph_recipe.tail_recipe == RecipeSpec(
        [TypedMethodStep("stft", {"n_fft": 512, "hop_length": 128, "win_length": 512, "window": "hann"})]
    )
    np.testing.assert_allclose(replayed.data, processed.data)
    assert isinstance(replayed, SpectrogramFrame)
```

- [ ] **Step 4: Add a non-commutative order test**

Add this test after the default add-with-SNR test:

```python
def test_graph_recipe_numbered_default_names_preserve_binary_operand_order() -> None:
    base = _frame()
    signal_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    noise_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="noise")
    processed = signal_source.remove_dc() - noise_source.normalize()

    graph_recipe = GraphRecipeSpec.from_frame(processed)
    replayed = graph_recipe.apply({"input_0": signal_source, "input_1": noise_source})

    assert graph_recipe.input_recipes == (
        ("input_0", RecipeSpec([OperationSpec("remove_dc")])),
        (
            "input_1",
            RecipeSpec(
                [OperationSpec("normalize", {"axis": -1, "fill": None, "norm": float("inf"), "threshold": None})]
            ),
        ),
    )
    assert graph_recipe.output == BinaryFrameStep("-", "input_0", "input_1")
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history
```

- [ ] **Step 5: Run focused tests and verify they fail**

Run:

```bash
uv run pytest \
  tests/pipeline/test_recipe.py::test_graph_recipe_from_frame_uses_numbered_default_input_names \
  tests/pipeline/test_recipe.py::test_graph_recipe_from_frame_uses_numbered_default_names_for_raw_add_with_snr \
  tests/pipeline/test_recipe.py::test_graph_recipe_numbered_default_names_preserve_binary_operand_order \
  tests/pipeline/test_recipe.py::test_graph_recipe_from_frame_uses_numbered_default_names_with_typed_tail \
  -q
```

Expected: failures showing current default names are still `left` / `right`.

### Task 2: Change GraphRecipeSpec Default Names

**Files:**
- Modify: `wandas/pipeline/specs.py`
- Test: `tests/pipeline/test_recipe.py`

- [ ] **Step 1: Implement the default-name change**

In `GraphRecipeSpec.from_frame(...)`, replace:

```python
resolved_input_names = ("left", "right") if input_names is None and len(inputs) == 2 else input_names
```

with:

```python
resolved_input_names = tuple(f"input_{index}" for index in range(len(inputs))) if input_names is None else input_names
```

- [ ] **Step 2: Run focused tests and verify they pass**

Run:

```bash
uv run pytest \
  tests/pipeline/test_recipe.py::test_graph_recipe_from_frame_uses_numbered_default_input_names \
  tests/pipeline/test_recipe.py::test_graph_recipe_from_frame_uses_numbered_default_names_for_raw_add_with_snr \
  tests/pipeline/test_recipe.py::test_graph_recipe_numbered_default_names_preserve_binary_operand_order \
  tests/pipeline/test_recipe.py::test_graph_recipe_from_frame_uses_numbered_default_names_with_typed_tail \
  -q
```

Expected: `4 passed`.

- [ ] **Step 3: Run nearby GraphRecipeSpec tests**

Run:

```bash
uv run pytest tests/pipeline/test_recipe.py -q
```

Expected: all tests in `tests/pipeline/test_recipe.py` pass.

- [ ] **Step 4: Commit tests and implementation**

Run:

```bash
git add wandas/pipeline/specs.py tests/pipeline/test_recipe.py
git commit -m "fix: unify graph recipe default input names"
```

Expected: one commit containing only implementation and test changes.

### Task 3: Update User And Boundary Docs

**Files:**
- Modify: `docs/src/explanation/pipeline-recipe-extraction-boundaries.md`
- Modify: `docs/src/how-to/pipeline-recipes.md`

- [ ] **Step 1: Update extraction-boundaries implemented list**

In `docs/src/explanation/pipeline-recipe-extraction-boundaries.md`, replace:

```markdown
- `GraphRecipeSpec.from_frame(processed)` with default structural input names `left` and `right`
```

with:

```markdown
- `GraphRecipeSpec.from_frame(processed)` with default structural input names `input_0` and `input_1`
```

- [ ] **Step 2: Update GraphRecipeSpec explanation paragraph**

Replace the sentence:

```markdown
`input_names` を省略した場合は、Python 変数名や frame label ではなく、構造上の左右を表す `left` / `right` を使う。
```

with:

```markdown
`input_names` を省略した場合は、Python 変数名や frame label ではなく、source leaf の順番に基づく `input_0` / `input_1` を使う。二項演算では `input_0` が左 operand、`input_1` が右 operand であり、`input_0 - input_1` のような非可換演算でも順序を保つ。
```

- [ ] **Step 3: Update not-implemented naming bullet**

Replace:

```markdown
- Python 変数名や frame label に基づく入力名推定。現在の runtime lineage は source identity を保存しないため、名前推定は `left` / `right` の構造名に限定する。
```

with:

```markdown
- Python 変数名や frame label に基づく入力名推定。現在の runtime lineage は source identity を保存しないため、名前推定は行わず、省略時は `input_0` / `input_1` / ... の機械的な名前だけを使う。
```

- [ ] **Step 4: Add a default-name note to the how-to Graph Recipes section**

In `docs/src/how-to/pipeline-recipes.md`, replace:

```markdown
`input_names` are explicit because runtime lineage does not know Python variable names.
```

with:

```markdown
`input_names` are explicit because runtime lineage does not know Python variable names. If omitted, graph recipes use mechanical names such as `input_0` and `input_1`; for binary frame recipes, `input_0` is the left operand and `input_1` is the right operand.
```

- [ ] **Step 5: Update unsupported boundary table wording**

In `docs/src/how-to/pipeline-recipes.md`, replace:

```markdown
| Implicit runtime names or values | Pass explicit `input_names=(...)` and runtime inputs. |
```

with:

```markdown
| Semantic runtime names or values | Pass explicit `input_names=(...)` and runtime inputs; omitted names are mechanical `input_0`, `input_1`, ... labels. |
```

- [ ] **Step 6: Run docs build**

Run:

```bash
uv run mkdocs build -f docs/mkdocs.yml
```

Expected: documentation builds successfully.

- [ ] **Step 7: Commit docs**

Run:

```bash
git add docs/src/explanation/pipeline-recipe-extraction-boundaries.md docs/src/how-to/pipeline-recipes.md
git commit -m "docs: clarify graph recipe input name defaults"
```

Expected: one docs commit.

### Task 4: Final Verification And PR

**Files:**
- No new files expected beyond previous tasks.

- [ ] **Step 1: Run full Recipe tests**

Run:

```bash
uv run pytest tests/pipeline/test_recipe.py -q
```

Expected: all tests in `tests/pipeline/test_recipe.py` pass.

- [ ] **Step 2: Run docs build**

Run:

```bash
uv run mkdocs build -f docs/mkdocs.yml
```

Expected: documentation builds successfully.

- [ ] **Step 3: Run lint and type checks**

Run:

```bash
uv run ruff check
uv run ty check
```

Expected:

- `uv run ruff check` passes.
- If full `uv run ty check` still reports unrelated diagnostics in `learning-path/07_frame_centric_recipe_ux.py`, run and record:

```bash
uv run ty check wandas tests/pipeline/test_recipe.py
```

Expected for the diff-scope check: `All checks passed!`.

- [ ] **Step 4: Run whitespace check and clean generated artifacts**

Run:

```bash
git diff --check
rm -rf .coverage .pytest_cache .ruff_cache coverage.xml docs/site tests/__pycache__ tests/pipeline/__pycache__ wandas/__pycache__ wandas/core/__pycache__ wandas/frames/__pycache__ wandas/frames/mixins/__pycache__ wandas/io/__pycache__ wandas/pipeline/__pycache__ wandas/processing/__pycache__ wandas/utils/__pycache__
git status --short --branch --ignored
```

Expected: `git diff --check` has no output, and final status shows no generated artifacts.

- [ ] **Step 5: Push and open PR**

Run:

```bash
git push -u origin codex/issue-263-input-names
gh pr create --repo kasahart/wandas \
  --base develop \
  --head codex/issue-263-input-names \
  --title "[codex] Unify Recipe graph default input names" \
  --body "## Summary
- Change omitted GraphRecipeSpec input names from left/right to input_0/input_1.
- Keep explicit input_names behavior unchanged and keep NodeGraphRecipeSpec defaults aligned.
- Document that Wandas does not infer semantic input names from Python variables or frame labels.

## Issue Relationship
Closes #263
Related #264
Related #265
Related #257
Related #258

## Validation
- \`uv run pytest tests/pipeline/test_recipe.py -q\`
- \`uv run mkdocs build -f docs/mkdocs.yml\`
- \`uv run ruff check\`
- \`uv run ty check\` or documented diff-scope \`uv run ty check wandas tests/pipeline/test_recipe.py\`
- \`git diff --check\`"
```

Expected: PR URL is printed.

- [ ] **Step 6: Confirm PR state and SHA alignment**

Run:

```bash
gh pr view --repo kasahart/wandas --json number,state,isDraft,url,headRefOid,statusCheckRollup,reviewRequests,reviewDecision
git rev-parse HEAD origin/codex/issue-263-input-names
```

Expected: PR is open, local/remote/PR head SHAs match, and checks are queued or successful.
