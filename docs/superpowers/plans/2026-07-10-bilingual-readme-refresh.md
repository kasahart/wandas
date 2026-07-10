# Bilingual README Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish accurate, equivalent English and Japanese README onboarding flows that work from both a repository checkout and a PyPI installation.

**Architecture:** Treat the two README files as one bilingual public contract and keep executable behavior in `tests/docs/test_readme_examples.py`. Preserve the deterministic known-signal proof and committed figures, while replacing repository-only sample loading with a local-path/public-URL source expression.

**Tech Stack:** Markdown, Python 3.10+, pytest, Wandas public API, NumPy, Matplotlib, `uv`, Ruff, ty

## Global Constraints

- Use the user-provided Japanese README as the content baseline and write a natural, semantically equivalent English version.
- Do not change Wandas runtime APIs or regenerate committed figures unless verification proves the documentation cannot otherwise be accurate.
- Use `wd.read()` for registered external sources and `wd.load()` for WDF.
- Limit `normalize=True` to listening and shape inspection; calibrated acoustic analysis must start with pressure-valued data.
- Preserve frame immutability, metadata/history, and Dask laziness claims without implying NumPy or tensor conversion remains lazy.
- Use `uv` for Python commands.

---

### Task 1: Lock the bilingual README contract in tests

**Files:**
- Modify: `tests/docs/test_readme_examples.py`
- Test: `tests/docs/test_readme_examples.py`

**Interfaces:**
- Consumes: Markdown files in `README_PATHS` and existing `_python_blocks()` / `_python_block_containing()` helpers.
- Produces: Tests that require local/public sample selection, bilingual section parity, executable examples, calibrated-analysis cautions, and the existing known-signal semantics.

- [ ] **Step 1: Add a failing sample-source contract test**

Add a test that checks both README sample blocks contain the local path, the public fallback, and the same `sample_source` call:

```python
def test_readme_sample_audio_supports_checkout_and_installed_users() -> None:
    expected_url = (
        "https://raw.githubusercontent.com/kasahart/wandas/main/"
        "learning-path/sample_audio.wav"
    )
    for path in README_PATHS:
        block = _python_block_containing(path, "sample_source")
        assert 'Path("learning-path/sample_audio.wav")' in block
        assert expected_url in block
        assert "recording = wd.read(sample_source" in block
```

- [ ] **Step 2: Add failing content-contract assertions**

Require both languages to document `previous`, `operation_history`, `wd.load()`, calibrated pressure, Dask-backed folder datasets, and tensor conversion. Keep assertions semantic and avoid requiring whole paragraphs verbatim.

```python
def test_readme_documents_frame_context_and_boundaries() -> None:
    english = README_PATHS[0].read_text(encoding="utf-8")
    japanese = README_PATHS[1].read_text(encoding="utf-8")
    for text in (english, japanese):
        assert "`previous`" in text
        assert "`operation_history`" in text
        assert "`wd.load()`" in text
        assert "`wandas[ml]`" in text
        assert 'lazy_loading=True' in text
    assert "calibrated" in english
    assert "校正" in japanese
```

- [ ] **Step 3: Run the new tests and confirm failure**

Run:

```bash
uv run pytest tests/docs/test_readme_examples.py -q
```

Expected: failures identifying missing `sample_source` fallback and missing newly required README contract text.

- [ ] **Step 4: Keep existing high-value tests intact**

Retain execution of all Python blocks, repository-link checks, known-signal numeric peaks, plot semantics, PNG signatures, install-before-example ordering, and the maximum of three Python blocks per language. Adjust only heading and prose markers that intentionally changed.

---

### Task 2: Rewrite the Japanese README from the approved baseline

**Files:**
- Modify: `README.ja.md`
- Test: `tests/docs/test_readme_examples.py`

**Interfaces:**
- Consumes: The approved design, the user's Japanese baseline, existing figure paths, and the public Wandas API.
- Produces: The canonical Japanese onboarding narrative used to align the English version.

- [ ] **Step 1: Replace the sample-audio block with an accessible source expression**

Use this exact source-selection contract:

```python
from pathlib import Path

import wandas as wd

local_sample = Path("learning-path/sample_audio.wav")
sample_source = (
    local_sample
    if local_sample.exists()
    else "https://raw.githubusercontent.com/kasahart/wandas/main/learning-path/sample_audio.wav"
)

recording = wd.read(sample_source, start=0, end=10, normalize=True)
recording.describe(fmin=20, fmax=8_000, image_save="readme_sample_audio_describe.png")
```

- [ ] **Step 2: Apply the approved Japanese structure and wording**

Use these section headings in this order:

```text
## なぜ Wandas か
## インストール
## サンプル音声を確認する
## 既知信号で確認する
## 手元のデータで使う
## 小さな top-level API
## 主なオブジェクト
## 向いている用途
## 次に読む
## プロジェクトの状態
## 貢献
## ライセンス
```

Include the immutable `remove_dc()` result, `clean.previous`, `clean.operation_history`, the WDF `wd.load()` distinction, the calibrated-pressure warning, folder dataset chain, and `wandas[ml]` requirement exactly once in their most relevant sections.

- [ ] **Step 3: Run focused Japanese contract checks**

Run:

```bash
uv run pytest tests/docs/test_readme_examples.py -q
```

Expected: Japanese assertions and executable blocks pass; English parity assertions may still fail until Task 3.

---

### Task 3: Produce the equivalent English README

**Files:**
- Modify: `README.md`
- Test: `tests/docs/test_readme_examples.py`

**Interfaces:**
- Consumes: The completed Japanese structure from Task 2.
- Produces: A natural English README with identical examples, feature boundaries, warnings, and section order.

- [ ] **Step 1: Mirror the Japanese information architecture in English**

Use these section headings in this order:

```text
## Why Wandas
## Installation
## Inspect the Sample Audio
## Validate with a Known Signal
## Use Your Own Data
## Small top-level API
## Core Objects
## Good Fits
## Learn More
## Project Status
## Contributing
## License
```

Translate for idiomatic technical English rather than sentence-by-sentence literal equivalence. Keep all code blocks equivalent except prose labels and surrounding explanation.

- [ ] **Step 2: Use the same sample-source code and known-signal code**

Copy the Task 2 sample-source block unchanged. Preserve the existing deterministic signal generation, `remove_dc()`, `welch(n_fft=4096)`, waveform plot, and spectrum plot code so the committed figures and numerical tests remain valid.

- [ ] **Step 3: State the analysis boundaries explicitly**

Include all of the following claims in natural English:

```text
normalize=True is for listening and shape inspection.
Use pressure-calibrated data for SPL and psychoacoustic metrics.
Read WDF with wd.load(), not wd.read().
Tensor conversion requires wandas[ml] and materializes the data.
Folder datasets default to lazy loading and can chain preprocessing before STFT.
```

- [ ] **Step 4: Run the complete README test module**

Run:

```bash
uv run pytest tests/docs/test_readme_examples.py -q
```

Expected: all tests pass.

---

### Task 4: Validate accuracy and repository quality

**Files:**
- Verify: `README.md`
- Verify: `README.ja.md`
- Verify: `tests/docs/test_readme_examples.py`

**Interfaces:**
- Consumes: Completed bilingual READMEs and tests.
- Produces: Evidence that the PR is accurate and ready for a completion review.

- [ ] **Step 1: Run all documentation tests**

```bash
uv run pytest tests/docs
```

Expected: all tests pass.

- [ ] **Step 2: Run lint and type checks required by repository guidance**

```bash
uv run ruff check wandas tests
uv run ty check wandas tests
```

Expected: both commands exit successfully.

- [ ] **Step 3: Check Markdown and workspace integrity**

```bash
git diff --check
git status --short --ignored
```

Expected: no whitespace errors; only intended README, test, design, and plan changes plus pre-existing ignored artifacts.

- [ ] **Step 4: Recheck the unresolved review requirement**

Confirm both README code blocks contain the raw GitHub fallback and that PR #281 still has no additional unresolved actionable thread. Do not reply to or resolve GitHub threads without explicit authorization.

- [ ] **Step 5: Commit the implementation separately from planning documentation**

```bash
git add README.md README.ja.md tests/docs/test_readme_examples.py docs/superpowers/plans/2026-07-10-bilingual-readme-refresh.md
git commit -m "docs: complete bilingual README onboarding"
```

Expected: the implementation commit contains only the two READMEs, README tests, and this plan.
