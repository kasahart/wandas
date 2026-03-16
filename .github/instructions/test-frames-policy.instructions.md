---
description: "Frame test patterns: immutability, metadata propagation, lazy evaluation, chaining, and domain transitions"
applyTo: "tests/frames/**"
---
# Wandas Test Policy: Frames (`tests/frames/`)

Frame テストは「データ構造としての正しさ」と「ドメイン横断時のメタデータ整合性」を検証する層です。
Frames は Wandas の public API の中核であり、ユーザーが直接操作するオブジェクトです。

**前提**: このファイルは [test-grand-policy.instructions.md](test-grand-policy.instructions.md) と同時に適用されます。
Grand Policy の 4 つの柱を遵守したうえで、以下の追加ガイドラインに従ってください。

---

## Common Fixtures for Frame Tests

```python
import numpy as np
import pytest
from wandas.frames.channel import ChannelFrame


@pytest.fixture
def channel_frame():
    """Standard 2-channel frame using deterministic sine waves (not random)."""
    sr = 16000
    t = np.arange(sr) / sr
    ch0 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    ch1 = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    return ChannelFrame.from_numpy(np.stack([ch0, ch1]), sampling_rate=sr, ch_labels=["ch0", "ch1"])


@pytest.fixture
def mono_frame():
    """Single-channel frame."""
    sr = 16000
    data = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    return ChannelFrame.from_numpy(data, sampling_rate=sr, label="mono_440Hz")
```

> **Note**: Use deterministic signals (sine/impulse) rather than random data. For structural/immutability tests where content doesn't matter, seeded RNG (`np.random.default_rng(42)`) is acceptable, but prefer sines for consistency with the grand policy.

---

## Frame Types & Their Test Concerns

### ChannelFrame (時間領域)
- **不変性**: 全演算で `assert result is not channel_frame`
- **チャンネル操作**: `add_channel`, `remove_channel`, `rename_channels` 後のラベル整合性
- **スライシング**: int / slice / bool mask / label-based インデックスで正しいチャンネルが返ること
- **Dask laziness**: 演算設定後に `.compute()` が呼ばれていないこと

### SpectralFrame (周波数領域)
- **Complex data 型**: `.data` が複素数配列であること
- **Derived properties**: `magnitude`, `phase`, `power`, `dB`, `dBA` が数学的に正しいこと
- **周波数軸**: `frequency` 配列が Nyquist 周波数まで正しく構成されていること

### SpectrogramFrame (時間-周波数領域)
- **3 次元性**: shape が `(n_channels, n_freq_bins, n_time_frames)` であること
- **ISTFT round-trip**: 逆変換で元の時間領域信号を再現できること
- **時間軸・周波数軸**: 両軸のメタデータが正確であること

### RoughnessFrame, NOctFrame (特殊解析)
- **MoSQITo 等価性**: 外部ライブラリとの結果一致を検証
- **物理量メタデータ**: 単位（sone, asper 等）が正しく設定されていること

---

## Required Test Categories per Frame Operation

新しい Frame メソッドを追加する場合、以下の 4 カテゴリを必ず含めること:

**1. Immutability** — 元のフレームが変更されないこと:
```python
original_data = channel_frame.data.copy()
result = channel_frame.{method}(...)
assert result is not channel_frame
np.testing.assert_array_equal(channel_frame.data, original_data)
```

**2. Metadata Propagation** — サンプリングレートと history が正しく更新されること:
```python
result = channel_frame.{method}(...)
assert result.sampling_rate == channel_frame.sampling_rate
assert len(result.operation_history) == len(channel_frame.operation_history) + 1
```

**3. Lazy Evaluation** — Dask 配列が保持されること:
```python
from dask.array.core import Array as DaArray
result = channel_frame.{method}(...)
assert isinstance(result._data, DaArray)
```

**4. Chaining** — チェーン呼び出しに対応していること:
```python
result = channel_frame.{method}(...).normalize()
assert isinstance(result, ChannelFrame)
```

---

## Domain Transition Test Patterns

ドメインが変わる処理（`fft()`, `stft()`, `loudness()` 等）では追加の検証が必要:

```python
def test_fft_domain_transition(channel_frame):
    spectral = channel_frame.fft()
    assert isinstance(spectral, SpectralFrame)
    assert spectral.n_samples == channel_frame.n_samples // 2 + 1  # N/2 + 1 bins
    assert spectral.sampling_rate == channel_frame.sampling_rate
```

---

## Channel Collection Test Patterns

マルチチャンネル操作のテストで特に注意すべき点:

- `add_channel` 後のラベルと data の順序が一致すること
- 重複ラベルは `ValueError` になること
- 長さが違うデータを追加した場合は `ValueError` になること

```python
def test_add_channel_label_alignment(channel_frame):
    result = channel_frame.add_channel(np.ones((1, channel_frame.n_samples)), label="c")
    assert result.n_channels == channel_frame.n_channels + 1
    assert result.labels[-1] == "c"
```

---

## Indexing Test Matrix

ChannelFrame のインデックスアクセスはテストカバレッジが重要:

| Index Type | Example | Expected Behavior |
|-----------|---------|-------------------|
| `int` | `cf[0]` | Single channel ChannelFrame |
| Negative `int` | `cf[-1]` | Last channel |
| `slice` | `cf[0:2]` | Multi-channel subset |
| `bool mask` | `cf[cf.rms > 0.5]` | Channels matching condition |
| `str` label | `cf["ch0"]` | Channel by label |
| `list[str]` | `cf[["ch0", "ch1"]]` | Multiple channels by label |
| `list[int]` | `cf[[0, 2]]` | Multiple channels by index |
| Out of range | `cf[999]` | IndexError |

---

## Cross-References
- [test-grand-policy.instructions.md](test-grand-policy.instructions.md) — 4 pillars and test pyramid
- [frames-design.instructions.md](frames-design.instructions.md) — Frame architecture and immutability rules
