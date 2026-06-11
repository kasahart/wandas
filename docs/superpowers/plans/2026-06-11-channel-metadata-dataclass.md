# ChannelMetadata Dataclass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `ChannelMetadata`'s Pydantic model with a standard-library dataclass and remove the runtime `pydantic` dependency while preserving channel metadata behavior.

**Architecture:** Keep `_channel_metadata` as a Wandas-owned list of mutable `ChannelMetadata` objects. Only the implementation of `ChannelMetadata` changes: Pydantic APIs are removed and internal callers use standard-library `copy.deepcopy()`, class field sets, and `json`. xarray coords/attrs, operation execution, and storage layout are unchanged.

**Tech Stack:** Python dataclasses, standard `json`, existing `pytest`, `ruff`, `ty`, `uv`.

---

## File Structure

- Modify: `wandas/core/metadata.py`
  - Replace Pydantic `BaseModel` with a `@dataclass` implementation.
  - Keep `ChannelMetadata` as the only production type in the file.
- Modify: `wandas/core/base_frame.py`
  - Remove `pydantic.ValidationError` import.
  - Convert dict-to-`ChannelMetadata` construction errors to clear `ValueError`.
  - Replace `ChannelMetadata.model_fields` with `ChannelMetadata._MODEL_FIELDS`.
  - Replace `model_copy(deep=True)` with `copy.deepcopy(...)`.
- Modify: `wandas/frames/channel.py`
  - Replace remaining `model_copy(deep=True)` calls with `copy.deepcopy(...)`.
- Modify: `tests/core/test_metadata.py`
  - Update Pydantic-specific tests to dataclass behavior.
  - Add tests for non-Pydantic status, JSON validation, and `copy.deepcopy()` isolation.
- Modify: `tests/core/test_base_frame.py`
  - Add or adjust invalid dict metadata assertion to check the new `ValueError` message.
  - Add query validation coverage for dataclass field set if not already covered.
- Modify: `pyproject.toml`
  - Remove `pydantic>=2.11.0` from runtime dependencies.
- Modify: `uv.lock`
  - Regenerate lockfile with `uv lock` after removing the dependency.

## Baseline

Current baseline from the Phase 4 worktree:

```bash
uv run pytest
```

Expected: `1459 passed, 3 skipped`.

---

### Task 1: Add Dataclass-Oriented ChannelMetadata Tests

**Files:**
- Modify: `tests/core/test_metadata.py`

- [ ] **Step 1: Add standard-library copy import**

At the top of `tests/core/test_metadata.py`, change the imports from:

```python
import json
from typing import Any
```

to:

```python
import copy
import json
from dataclasses import is_dataclass
from typing import Any
```

- [ ] **Step 2: Add a test proving ChannelMetadata is a dataclass, not a Pydantic model**

Add this test near the top of `TestChannelMetadata`:

```python
    def test_channel_metadata_is_dataclass_not_pydantic_model(self) -> None:
        """ChannelMetadata uses stdlib dataclass semantics, not Pydantic APIs."""
        metadata = ChannelMetadata()

        assert is_dataclass(metadata)
        assert not hasattr(metadata, "model_copy")
        assert not hasattr(metadata, "model_fields")
        assert not hasattr(metadata, "model_dump_json")
        assert not hasattr(metadata, "model_validate_json")
```

- [ ] **Step 3: Replace the Pydantic model_copy test with deepcopy behavior**

Replace `test_copy_deep_independent_from_original` with:

```python
    def test_deepcopy_independent_from_original(self) -> None:
        """Standard deepcopy is independent from original."""
        metadata = ChannelMetadata(
            label="test_label",
            unit="Hz",
            extra={"source": "microphone", "nested": {"gain": 10}},
        )
        copied = copy.deepcopy(metadata)

        assert copied.label == metadata.label
        assert copied.unit == metadata.unit
        assert copied.extra == metadata.extra

        metadata.label = "modified_label"
        metadata.extra["new_key"] = "new_value"
        metadata.extra["nested"]["gain"] = 99

        assert copied.label == "test_label"
        assert "new_key" not in copied.extra
        assert copied.extra["nested"]["gain"] == 10
```

- [ ] **Step 4: Add explicit non-object JSON validation test**

Add this test after the existing JSON round-trip tests:

```python
    def test_from_json_rejects_non_object_json(self) -> None:
        """ChannelMetadata JSON must decode to an object."""
        with pytest.raises(ValueError, match="ChannelMetadata JSON must decode to an object"):
            ChannelMetadata.from_json('["not", "an", "object"]')
```

If `pytest` is not imported in `tests/core/test_metadata.py`, add:

```python
import pytest
```

near the other imports.

- [ ] **Step 5: Run metadata tests and verify expected failures**

Run:

```bash
uv run pytest tests/core/test_metadata.py -q
```

Expected: failures because `ChannelMetadata` is still a Pydantic model, `model_copy` still exists, and `from_json()` does not yet reject non-object JSON with the new message.

- [ ] **Step 6: Commit failing tests**

```bash
git add tests/core/test_metadata.py
git commit -m "test: define dataclass channel metadata behavior"
```

---

### Task 2: Replace ChannelMetadata Pydantic Model With Dataclass

**Files:**
- Modify: `wandas/core/metadata.py`
- Test: `tests/core/test_metadata.py`

- [ ] **Step 1: Replace metadata.py with dataclass implementation**

Replace the contents of `wandas/core/metadata.py` with:

```python
import json
from dataclasses import dataclass, field
from typing import Any

from wandas.utils.util import unit_to_ref


@dataclass
class ChannelMetadata:
    """Metadata for a single channel."""

    label: str = ""
    unit: str = ""
    ref: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)
    _initialized: bool = field(default=False, init=False, repr=False)

    _MODEL_FIELDS = frozenset({"label", "unit", "ref", "extra"})

    def __post_init__(self) -> None:
        if not isinstance(self.extra, dict):
            raise TypeError("ChannelMetadata extra must be a dictionary")
        if self.unit and self.ref == 1.0:
            self.ref = unit_to_ref(self.unit)
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name == "unit" and getattr(self, "_initialized", False) and value and isinstance(value, str):
            object.__setattr__(self, "ref", unit_to_ref(value))

    def __getitem__(self, key: str) -> Any:
        """Provide dictionary-like behavior."""
        if key in self._MODEL_FIELDS:
            return getattr(self, key)
        return self.extra.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Provide dictionary-like behavior."""
        if key in ("label", "ref"):
            setattr(self, key, value)
        elif key == "unit":
            self.unit = value
        else:
            self.extra[key] = value

    def matches_query(self, query: dict[str, Any]) -> bool:
        """Check whether this channel matches all key-value pairs in query."""
        for key, expected in query.items():
            actual = getattr(self, key, None)
            if actual is None:
                actual = self.extra.get(key)
                if actual is None:
                    return False

            if hasattr(expected, "search") and callable(expected.search):
                if not (isinstance(actual, str) and expected.search(actual)):
                    return False
            elif actual != expected:
                return False
        return True

    def to_json(self) -> str:
        """Convert to JSON format."""
        return json.dumps(
            {
                "label": self.label,
                "unit": self.unit,
                "ref": self.ref,
                "extra": self.extra,
            },
            indent=4,
        )

    @classmethod
    def from_json(cls, json_data: str) -> "ChannelMetadata":
        """Convert from JSON format."""
        data = json.loads(json_data)
        if not isinstance(data, dict):
            raise ValueError("ChannelMetadata JSON must decode to an object")
        try:
            return cls(**data)
        except TypeError as e:
            raise ValueError(f"Invalid ChannelMetadata JSON object: {e}") from e
```

- [ ] **Step 2: Run metadata tests**

Run:

```bash
uv run pytest tests/core/test_metadata.py -q
```

Expected: metadata tests pass, or fail only where tests still assert exact Pydantic JSON formatting.

- [ ] **Step 3: If JSON formatting tests fail, assert parsed JSON instead of raw string**

For tests that inspect `to_json()`, use:

```python
parsed = json.loads(metadata.to_json())
assert parsed == {
    "label": metadata.label,
    "unit": metadata.unit,
    "ref": metadata.ref,
    "extra": metadata.extra,
}
```

Do not assert exact whitespace or key ordering.

- [ ] **Step 4: Run metadata tests again**

Run:

```bash
uv run pytest tests/core/test_metadata.py -q
```

Expected: all tests in `tests/core/test_metadata.py` pass.

- [ ] **Step 5: Commit dataclass implementation**

```bash
git add wandas/core/metadata.py tests/core/test_metadata.py
git commit -m "refactor: make channel metadata a dataclass"
```

---

### Task 3: Remove Pydantic-Specific BaseFrame Usage

**Files:**
- Modify: `wandas/core/base_frame.py`
- Test: `tests/core/test_base_frame.py`

- [ ] **Step 1: Remove ValidationError import**

In `wandas/core/base_frame.py`, delete:

```python
from pydantic import ValidationError
```

- [ ] **Step 2: Replace dict conversion error handling**

In `BaseFrame.__init__`, replace:

```python
                    except ValidationError as e:
                        raise ValueError(
                            f"Invalid channel_metadata at index {index}\n"
                            f"  Got: {ch}\n"
                            f"  Validation error: {e}\n"
                            f"Ensure all dict keys match ChannelMetadata fields "
                            f"(label, unit, ref, extra) and have correct types."
                        ) from e
```

with:

```python
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"Invalid channel_metadata at index {index}\n"
                            f"  Got: {ch}\n"
                            f"  Error: {e}\n"
                            f"Ensure all dict keys match ChannelMetadata fields "
                            f"(label, unit, ref, extra) and have correct types."
                        ) from e
```

- [ ] **Step 3: Replace query key validation field access**

Replace:

```python
                    model_keys = set(ChannelMetadata.model_fields.keys())
```

with:

```python
                    model_keys = set(ChannelMetadata._MODEL_FIELDS)
```

- [ ] **Step 4: Replace BaseFrame model_copy call in binary operations**

Replace:

```python
            ch = self_ch.model_copy(deep=True)
```

with:

```python
            ch = copy.deepcopy(self_ch)
```

- [ ] **Step 5: Replace BaseFrame model_copy call in channel metadata helper**

Replace:

```python
            new_ch = ch.model_copy(deep=True)
```

with:

```python
            new_ch = copy.deepcopy(ch)
```

- [ ] **Step 6: Add invalid dict error assertion if missing**

In `tests/core/test_base_frame.py`, locate `TestBaseFrameChannelMetadata` and the invalid channel metadata test. Ensure it includes this assertion shape:

```python
        with pytest.raises(ValueError, match="Invalid channel_metadata at index 0"):
            ChannelFrame.from_numpy(
                np.array([[1.0, 2.0, 3.0]]),
                sampling_rate=3.0,
                channel_metadata=[{"label": "bad", "unknown": "field"}],
            )
```

If there is already a similar test for invalid dicts, update only the expected message to include `Invalid channel_metadata at index 0` and `unknown`.

- [ ] **Step 7: Run focused BaseFrame tests**

Run:

```bash
uv run pytest tests/core/test_base_frame.py::TestBaseFrameChannelMetadata -q
```

Expected: tests pass.

- [ ] **Step 8: Commit BaseFrame cleanup**

```bash
git add wandas/core/base_frame.py tests/core/test_base_frame.py
git commit -m "refactor: remove pydantic usage from base frame"
```

---

### Task 4: Replace Remaining model_copy Calls in Frame Code

**Files:**
- Modify: `wandas/frames/channel.py`
- Test: `tests/frames/test_channel_label_updates.py`
- Test: `tests/frames/test_channel_collection.py`

- [ ] **Step 1: Ensure channel.py imports copy**

At the top of `wandas/frames/channel.py`, add this import if it is not already present:

```python
import copy
```

- [ ] **Step 2: Replace model_copy in add-channel metadata copy**

Replace:

```python
                new_ch_meta = chmeta.model_copy(deep=True)
```

with:

```python
                new_ch_meta = copy.deepcopy(chmeta)
```

- [ ] **Step 3: Replace model_copy in rename metadata copy**

Replace:

```python
            new_ch_meta = ch_meta.model_copy(deep=True)
```

with:

```python
            new_ch_meta = copy.deepcopy(ch_meta)
```

- [ ] **Step 4: Run focused frame tests**

Run:

```bash
uv run pytest tests/frames/test_channel_label_updates.py tests/frames/test_channel_collection.py -q
```

Expected: tests pass.

- [ ] **Step 5: Search for remaining model_copy usage**

Run:

```bash
rg -n "model_copy" wandas tests
```

Expected: no output.

- [ ] **Step 6: Commit frame cleanup**

```bash
git add wandas/frames/channel.py
git commit -m "refactor: use deepcopy for channel metadata copies"
```

---

### Task 5: Remove Pydantic Dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Verify no code imports Pydantic before dependency removal**

Run:

```bash
rg -n "pydantic|ValidationError|model_fields|model_copy|model_dump|model_validate" wandas tests pyproject.toml
```

Expected: only `pyproject.toml` still contains `pydantic>=2.11.0`.

- [ ] **Step 2: Remove pydantic from runtime dependencies**

In `pyproject.toml`, delete this line from the dependency list:

```toml
    "pydantic>=2.11.0",
```

- [ ] **Step 3: Regenerate lockfile**

Run:

```bash
uv lock
```

Expected: lockfile updates successfully and removes the direct `wandas` dependency entry for `pydantic`.

- [ ] **Step 4: Verify pydantic is absent from direct dependency declarations**

Run:

```bash
rg -n "pydantic" pyproject.toml
```

Expected: no output.

Run:

```bash
rg -n "pydantic" uv.lock
```

Expected: no output if no transitive dependency requires it. If output remains, inspect the package section to confirm it is transitive and not listed under the `wandas` package dependencies.

- [ ] **Step 5: Commit dependency cleanup**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: remove pydantic dependency"
```

---

### Task 6: Final Verification and PR Prep

**Files:**
- No intended production changes.
- May update PR body if creating a PR in this task.

- [ ] **Step 1: Run the full Pydantic-specific search**

Run:

```bash
rg -n "pydantic|ValidationError|model_fields|model_copy|model_dump|model_validate" wandas tests pyproject.toml
```

Expected: no output.

- [ ] **Step 2: Run lint**

Run:

```bash
uv run ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 3: Run format check**

Run:

```bash
uv run ruff format --check .
```

Expected: all files already formatted.

- [ ] **Step 4: Run type check**

Run:

```bash
uv run ty check wandas tests
```

Expected: all checks pass.

- [ ] **Step 5: Run full tests**

Run:

```bash
uv run pytest
```

Expected: all tests pass. The baseline before Phase 4 was `1459 passed, 3 skipped`; the exact count may increase by the new tests.

- [ ] **Step 6: Review diff for out-of-scope changes**

Run:

```bash
git diff --stat origin/develop...HEAD
git diff origin/develop...HEAD -- wandas/core/metadata.py wandas/core/base_frame.py wandas/frames/channel.py pyproject.toml tests/core/test_metadata.py tests/core/test_base_frame.py
```

Expected: changes are limited to dataclass conversion, Pydantic dependency removal, related tests, and the Phase 4 spec/plan docs. There should be no xarray coords/attrs storage migration and no operation execution changes.

- [ ] **Step 7: Commit any verification-only documentation update if needed**

If the PR body or local plan needs a verification count update, commit only documentation changes:

```bash
git add docs/superpowers/plans/2026-06-11-channel-metadata-dataclass.md
git commit -m "docs: finalize channel metadata dataclass plan"
```

Expected: skip this step if no files changed.

- [ ] **Step 8: Prepare PR summary**

Use this PR summary:

```markdown
## Summary
- Replace `ChannelMetadata` Pydantic model with a standard-library dataclass.
- Replace internal Pydantic APIs (`model_copy`, `model_fields`, JSON helpers, `ValidationError`) with stdlib equivalents.
- Remove the runtime `pydantic` dependency while preserving existing channel metadata behavior.

## Non-goals
- No `_channel_metadata` storage migration to xarray coords/attrs.
- No operation execution changes.
- No `from_xarray`, NetCDF, or Zarr changes.

## Verification
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run ty check wandas tests`
- `uv run pytest`
- `rg -n "pydantic|ValidationError|model_fields|model_copy|model_dump|model_validate" wandas tests pyproject.toml` -> no matches
```
