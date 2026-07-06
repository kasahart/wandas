# Recipe Replay Intent Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Recipe support matrix into a tested replay-intent contract for the risky #259 boundaries.

**Architecture:** Keep Recipe replay thin: extraction converts supported lineage into public Recipe steps, and replay calls existing frame APIs. Unsupported extraction boundaries raise `RecipeExtractionError` when a processed frame exists; frame-level invalid indexing may still fail before extraction. This plan should prefer tests and docs over implementation changes unless a new regression test proves the extractor is accepting an unsupported boundary.

**Tech Stack:** Python, pytest, Dask arrays, Wandas `ChannelFrame`, `RecipeSpec`, `GraphRecipeSpec`, `NodeGraphRecipeSpec`, MkDocs.

---

## File Structure

- Modify `docs/src/explanation/pipeline-recipe-support-matrix.md`
  - Add a short "Recipe に保存するもの / 保存しないもの" contract table in beginner language.
- Modify `docs/src/explanation/pipeline-recipe-extraction-boundaries.md`
  - Align the detailed engineering reference with the same replay-intent contract.
- Modify `tests/pipeline/test_recipe.py`
  - Add focused regression tests for boundaries that are easy to misunderstand: legacy multidimensional indexing lineage, `add_channel` payload references, frame-input `add_channel` options, and `add_with_snr` length alignment.
- Modify `wandas/pipeline/extraction.py` only if a new test shows unsupported lineage is silently accepted.
  - Keep changes local to `_getitem_step_from_graph`, `_binary_frame_step_from_graph`, `_add_channel_step_from_graph`, or `_add_channel_data_step_from_graph`.
- Do not modify WDF persistence, sklearn adapters, operation numerical code, or frame metadata propagation in this PR.

---

### Task 1: Strengthen The Beginner Contract Table

**Files:**
- Modify: `docs/src/explanation/pipeline-recipe-support-matrix.md`

- [ ] **Step 1: Add the contract section after "判定の読み方"**

Insert this section after the status legend table:

```markdown
## Recipe に保存するもの

Recipe は「あとで同じ意味で実行できる手順」だけを保存します。

| 保存する | 保存しない |
| --- | --- |
| 処理名と引数 | 計算済みの波形データ |
| channel の選び方 | Python 関数そのもの |
| 外から渡す入力名 | NumPy/Dask array の中身 |
| `add_channel` の公開オプション | 内部で一時的に行われた長さ合わせ |

保存すると意味が変わりそうなものは、Recipe にせずエラーにします。
```

- [ ] **Step 2: Clarify the specific risky rows**

Update these rows in the first support table:

```markdown
| `fix_length(duration=...)` で長さをそろえる | 対応 | `RecipeSpec` | `frame.fix_length(duration=0.25)` | Recipe には秒数ではなく、実行時に決まったサンプル数を保存する |
| 2つの frame を SNR 指定で混ぜる | 対応 | `GraphRecipeSpec` | `signal.add(noise, snr=6.0)` | `snr` だけ保存する。内部の長さ合わせは Recipe に入れない |
```

Place them near the existing `fix_length`/frame-method and two-frame rows so the table stays easy to scan.

- [ ] **Step 3: Check the docs diff**

Run:

```bash
git diff -- docs/src/explanation/pipeline-recipe-support-matrix.md
```

Expected: the new section uses beginner language and does not introduce WDF persistence, sklearn export, or implementation-only terms such as `operation_graph`.

---

### Task 2: Align The Detailed Boundary Reference

**Files:**
- Modify: `docs/src/explanation/pipeline-recipe-extraction-boundaries.md`

- [ ] **Step 1: Add an explicit replay-intent rule near the top**

Insert this after the "Current Extraction Entry Point" paragraph:

```markdown
## Replay Intent Rule / 再生意図のルール

Recipe extraction preserves what the user meant to replay, not every helper step that happened at runtime.

- If a public frame API can replay the same meaning, Recipe stores that public API call.
- If only a runtime helper detail is available, Recipe either converts it to a stable public argument or rejects extraction.
- Recipe replay delegates to existing frame APIs. Recipe code does not duplicate numerical processing, metadata propagation, source-time updates, or Dask graph construction.
```

- [ ] **Step 2: Tighten the existing `fix_length(duration=...)` wording**

Replace the current table row for `frame.fix_length(duration=0.25)` with:

```markdown
| `frame.fix_length(duration=0.25)` | `MethodStep("fix_length", {"length": int(0.25 * sampling_rate)})` | Recipe stores the resolved sample length. It does not promise to replay the original duration argument on a different sampling rate. |
```

- [ ] **Step 3: Add the `add_with_snr` helper-boundary note**

In the Stage 4 section, after the paragraph about `GraphRecipeSpec.from_frame(...)` for binary merges, add:

```markdown
For `add_with_snr`, Recipe stores the two frame inputs and the public `snr` value. If the runtime method internally adjusts the noise length, that helper operation is not extracted as a separate `fix_length` step. Replay calls `frame.add(other, snr=...)`, so the current inputs decide any length alignment.
```

- [ ] **Step 4: Check the docs diff**

Run:

```bash
git diff -- docs/src/explanation/pipeline-recipe-extraction-boundaries.md
```

Expected: the detailed page now states the same contract as the support matrix without changing WDF or sklearn/export scope.

---

### Task 3: Add Focused Regression Tests

**Files:**
- Modify: `tests/pipeline/test_recipe.py`

- [ ] **Step 1: Add a test for unsupported multidimensional lineage**

Add this near the existing multidimensional indexing tests:

```python
def test_recipe_from_frame_rejects_non_slice_multidimensional_indexing_lineage() -> None:
    frame = _two_channel_frame_with_refs()
    processed = frame[:, 100:400]
    assert processed.lineage is not None
    processed._lineage = frame._lineage_with_method("__getitem__", {"indexing": "multidimensional"})

    with pytest.raises(RecipeExtractionError, match="Indexing recipe extraction only supports"):
        RecipeSpec.from_frame(processed)
```

- [ ] **Step 2: Add a test that raw add-channel data is referenced, not serialized**

Add this near `test_node_graph_recipe_from_frame_extracts_add_channel_numpy_data_input`:

```python
def test_node_graph_recipe_add_channel_data_serializes_input_name_not_array_values() -> None:
    frame = _frame()
    raw = np.arange(frame.n_samples, dtype=float)
    processed = frame.add_channel(raw, label="raw", source_time_offset=1.25)

    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("signal", "raw"))
    serialized = recipe.to_dict()

    assert serialized["inputs"] == ["signal", "raw"]
    assert serialized["nodes"] == [
        {
            "id": "n0",
            "step": {
                "add_channel_data": {
                    "base": "signal",
                    "data": "raw",
                    "params": {
                        "align": "strict",
                        "label": "raw",
                        "suffix_on_dup": None,
                        "source_time_offset": 1.25,
                    },
                }
            },
            "inputs": ["signal", "raw"],
        }
    ]
    assert "array" not in repr(serialized).lower()
    assert "arange" not in repr(serialized).lower()
```

- [ ] **Step 3: Add a test that frame-input add-channel does not accept raw source-time override**

Add this near `test_node_graph_recipe_from_frame_extracts_add_channel_frame_inputs`:

```python
def test_node_graph_recipe_add_channel_frame_input_omits_raw_source_time_offset_option() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, ch_labels=["left"])
    right_source = ChannelFrame.from_numpy(
        base.data * 0.5,
        sampling_rate=base.sampling_rate,
        ch_labels=["right"],
        source_time_offset=2.5,
    )
    processed = left_source.add_channel(right_source, label="ref")

    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("left", "right"))
    serialized = recipe.to_dict()

    assert serialized["nodes"][0]["step"] == {
        "add_channel": {
            "base": "left",
            "added": "right",
            "params": {"align": "strict", "label": "ref", "suffix_on_dup": None},
        }
    }
    assert "source_time_offset" not in serialized["nodes"][0]["step"]["add_channel"]["params"]
```

- [ ] **Step 4: Add a test that `add_with_snr` serialization omits implicit length alignment**

Add this near `test_graph_recipe_from_frame_does_not_bake_source_length_into_add_with_snr`:

```python
def test_graph_recipe_add_with_snr_serializes_snr_without_implicit_fix_length_step() -> None:
    base = _frame()
    data = base.data.reshape(1, -1)
    signal_source = ChannelFrame.from_numpy(data[:, :100], sampling_rate=base.sampling_rate, label="signal")
    noise_source = ChannelFrame.from_numpy(data[:, :200], sampling_rate=base.sampling_rate, label="noise")
    processed = signal_source.add(noise_source, snr=6.0)

    graph_recipe = GraphRecipeSpec.from_frame(processed, input_names=("signal", "noise"))

    assert graph_recipe.to_dict() == {
        "inputs": {
            "signal": {"steps": []},
            "noise": {"steps": []},
        },
        "output": {
            "binary_frame": {
                "operation": "add_with_snr",
                "left": "signal",
                "right": "noise",
                "params": {"snr": 6.0},
            }
        },
    }
```

- [ ] **Step 5: Run the new tests**

Run:

```bash
uv run pytest \
  tests/pipeline/test_recipe.py::test_recipe_from_frame_rejects_non_slice_multidimensional_indexing_lineage \
  tests/pipeline/test_recipe.py::test_node_graph_recipe_add_channel_data_serializes_input_name_not_array_values \
  tests/pipeline/test_recipe.py::test_node_graph_recipe_add_channel_frame_input_omits_raw_source_time_offset_option \
  tests/pipeline/test_recipe.py::test_graph_recipe_add_with_snr_serializes_snr_without_implicit_fix_length_step \
  -q
```

Expected: all selected tests pass if current implementation already follows the contract. If a test fails because serialization or extraction accepts unsupported data, proceed to Task 4.

---

### Task 4: Apply Minimal Extractor Fixes Only If Needed

**Files:**
- Modify only if tests fail: `wandas/pipeline/extraction.py`
- Test: `tests/pipeline/test_recipe.py`

- [ ] **Step 1: If non-slice multidimensional lineage is accepted, tighten `_getitem_step_from_graph`**

If `test_recipe_from_frame_rejects_non_slice_multidimensional_indexing_lineage` fails, ensure `_getitem_step_from_graph` rejects unknown indexing kinds with the existing `RecipeExtractionError` path:

```python
def _getitem_step_from_graph(params: Mapping[str, Any]) -> IndexingStep:
    indexing = params.get("indexing")
    if indexing not in _REPLAYABLE_GETITEM_INDEXING:
        raise RecipeExtractionError(
            "Indexing recipe extraction only supports channel-only label, slice, label list, "
            "and multidimensional slice selection\n"
            f"  Indexing kind: {indexing!r}\n"
            "  Multidimensional, callable, regex, dict, and array indexing need a selection recipe model "
            "that can preserve full indexing intent."
        )
```

- [ ] **Step 2: If `add_channel` serializes raw values, keep only input references**

If either add-channel serialization test fails, keep `_add_channel_step_from_graph` and `_add_channel_data_step_from_graph` limited to these params:

```python
return AddChannelStep(
    base,
    added,
    {
        "align": params.get("align", "strict"),
        "label": params.get("label"),
        "suffix_on_dup": params.get("suffix_on_dup"),
    },
)
```

```python
return AddChannelDataStep(
    base,
    data,
    {
        "align": params.get("align", "strict"),
        "label": params.get("label"),
        "suffix_on_dup": params.get("suffix_on_dup"),
        "source_time_offset": params.get("source_time_offset"),
    },
)
```

- [ ] **Step 3: If `add_with_snr` serializes helper steps, keep only `snr`**

If the `add_with_snr` serialization test fails, keep `_binary_frame_step_from_graph` limited to public `snr`:

```python
def _binary_frame_step_from_graph(operation: str, params: Mapping[str, Any], left: str, right: str) -> BinaryFrameStep:
    if operation == "add_with_snr":
        return BinaryFrameStep("add_with_snr", left, right, {"snr": params["snr"]})
```

- [ ] **Step 4: Re-run the focused tests**

Run the same command from Task 3 Step 5.

Expected: all selected tests pass.

---

### Task 5: Run Contract-Level Verification

**Files:**
- Test: `tests/pipeline/test_recipe.py`
- Docs: `docs/src/explanation/pipeline-recipe-support-matrix.md`, `docs/src/explanation/pipeline-recipe-extraction-boundaries.md`

- [ ] **Step 1: Run the Recipe test file**

Run:

```bash
uv run pytest tests/pipeline/test_recipe.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run docs build**

Run:

```bash
uv run mkdocs build -f docs/mkdocs.yml
```

Expected: documentation builds successfully.

- [ ] **Step 3: Remove generated docs artifacts**

Run:

```bash
rm -rf docs/site
```

Expected: `docs/site/` is gone and not staged.

- [ ] **Step 4: Check formatting and generated artifacts**

Run:

```bash
git diff --check
git status --short --ignored
```

Expected: `git diff --check` exits 0. `git status --short --ignored` shows only intended tracked changes and no unintended `.pytest_cache/`, `.ruff_cache/`, `docs/site/`, or `__pycache__/` artifacts.

---

### Task 6: Commit And Update PR Metadata

**Files:**
- Commit all modified docs/tests/code files from this plan.
- Update PR #262 body after push.

- [ ] **Step 1: Commit the implementation**

Run:

```bash
git add \
  docs/src/explanation/pipeline-recipe-support-matrix.md \
  docs/src/explanation/pipeline-recipe-extraction-boundaries.md \
  tests/pipeline/test_recipe.py \
  wandas/pipeline/extraction.py
git commit -m "test: pin recipe replay intent boundaries"
```

If `wandas/pipeline/extraction.py` was not modified, omit it from `git add`.

- [ ] **Step 2: Push the branch**

Run:

```bash
git push
```

Expected: branch `codex/issue-259-recipe-support-matrix` updates PR #262.

- [ ] **Step 3: Update PR body**

Use `gh pr edit 262 --repo kasahart/wandas --body-file <file>` with a body that includes:

```markdown
## Summary
- Add a beginner-friendly Recipe support matrix.
- Define and document replay-intent contracts for `fix_length`, channel/time selection, `add_channel`, and `add_with_snr`.
- Add focused tests that pin supported and unsupported Recipe extraction boundaries.

## Issue Relationship
Related #259

This does not close #259 unless the final implementation fully satisfies every acceptance criterion.

## Validation
- `uv run pytest tests/pipeline/test_recipe.py -q`
- `uv run mkdocs build -f docs/mkdocs.yml`
- `git diff --check`
```

- [ ] **Step 4: Verify SHA alignment and checks**

Run:

```bash
git rev-parse HEAD origin/codex/issue-259-recipe-support-matrix
gh pr view 262 --repo kasahart/wandas --json headRefOid,statusCheckRollup,reviewRequests
```

Expected: local HEAD, remote branch, and PR head SHA match. Checks are passing or in progress with no unexpected failures.
