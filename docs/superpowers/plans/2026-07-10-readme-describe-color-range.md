# README Describe Color Range Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix both bilingual README `describe()` examples and their published sample figures to a spectrogram range of `-80` through `-20` dB.

**Architecture:** Keep `ChannelFrame.describe()` defaults unchanged and make the range explicit only in executable README code. Treat the README test as the public contract, then regenerate the two committed sample figures from the same arguments.

**Tech Stack:** Markdown, Python 3.10+, pytest, Wandas, Matplotlib, `uv`, Ruff, ty

## Global Constraints

- Use `vmin=-80, vmax=-20`; `vmin` must remain lower than `vmax`.
- Use `start=0, end=15` in both recording examples and in the regenerated sample figures.
- Update both `README.md` and `README.ja.md` with identical Python blocks.
- Regenerate only `images/readme_sample_audio_describe_0.png` and `images/readme_sample_audio_describe_1.png`.
- Do not change the `ChannelFrame.describe()` API or defaults.
- Use `uv` for Python commands.

---

### Task 1: Lock and implement the README color-range contract

**Files:**
- Modify: `tests/docs/test_readme_examples.py`
- Modify: `README.md`
- Modify: `README.ja.md`

**Interfaces:**
- Consumes: `_python_block_containing(path, marker)` and the two existing README `describe()` examples.
- Produces: Identical bilingual Python blocks with `vmin=-80, vmax=-20` in both `describe()` calls.

- [ ] **Step 1: Change the focused test before the READMEs**

Replace the existing frequency-range test with this complete contract test:

```python
def test_readme_describe_examples_keep_committed_color_range() -> None:
    """README describe examples should use the color range shown by the figures."""
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (REPO_ROOT / "README.ja.md").read_text(encoding="utf-8")
    english_sample = _python_block_containing(REPO_ROOT / "README.md", "sample_source")
    japanese_sample = _python_block_containing(REPO_ROOT / "README.ja.md", "sample_source")

    sample_call = (
        'recording.describe(fmin=20, fmax=8_000, vmin=-80, vmax=-20, '
        'image_save="readme_sample_audio_describe.png")'
    )
    own_data_call = (
        'clean.describe(fmin=20, fmax=fmax, vmin=-80, vmax=-20, '
        'image_save="recording_overview.png")'
    )

    assert sample_call in english_sample
    assert sample_call in japanese_sample
    assert own_data_call in english
    assert own_data_call in japanese
```

- [ ] **Step 2: Run the focused test and confirm RED**

```bash
uv run pytest tests/docs/test_readme_examples.py::test_readme_describe_examples_keep_committed_color_range -q
```

Expected: FAIL because the README calls do not yet contain `vmin=-80, vmax=-20`.

- [ ] **Step 3: Update both README calls**

Use these exact calls in both languages:

```python
recording.describe(fmin=20, fmax=8_000, vmin=-80, vmax=-20, image_save="readme_sample_audio_describe.png")
```

```python
clean.describe(fmin=20, fmax=fmax, vmin=-80, vmax=-20, image_save="recording_overview.png")
```

- [ ] **Step 4: Run the focused test and confirm GREEN**

```bash
uv run pytest tests/docs/test_readme_examples.py::test_readme_describe_examples_keep_committed_color_range -q
```

Expected: PASS.

---

### Task 2: Regenerate figures and verify the documentation

**Files:**
- Modify: `images/readme_sample_audio_describe_0.png`
- Modify: `images/readme_sample_audio_describe_1.png`
- Verify: `README.md`
- Verify: `README.ja.md`
- Verify: `tests/docs/test_readme_examples.py`

**Interfaces:**
- Consumes: The sample audio at `learning-path/sample_audio.wav` and the exact Task 1 color range.
- Produces: Two committed PNGs matching the documented sample call.

- [ ] **Step 1: Regenerate the sample figures**

```bash
uv run python -c 'import wandas as wd; recording = wd.read("learning-path/sample_audio.wav", start=0, end=15, normalize=True); recording.describe(fmin=20, fmax=8_000, vmin=-80, vmax=-20, image_save="images/readme_sample_audio_describe.png")'
```

Expected: `images/readme_sample_audio_describe_0.png` and `images/readme_sample_audio_describe_1.png` are rewritten.

- [ ] **Step 2: Run documentation and static checks**

```bash
uv run pytest tests/docs
uv run ruff check wandas tests
uv run --extra marimo --extra psychoacoustic ty check wandas tests
git diff --check
```

Expected: all commands exit successfully.

- [ ] **Step 3: Check generated artifacts and commit**

```bash
git status --short --ignored
git add README.md README.ja.md tests/docs/test_readme_examples.py \
  images/readme_sample_audio_describe_0.png \
  images/readme_sample_audio_describe_1.png \
  docs/superpowers/plans/2026-07-10-readme-describe-color-range.md
git commit -m "docs: fix README describe color range"
```

Expected: one implementation commit containing the bilingual examples, test contract, regenerated figures, and plan.
