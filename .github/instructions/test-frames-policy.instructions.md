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
    """Standard 2-channel frame for general testing."""
    sr = 16000
    n_samples = 16000  # 1 second
    data = np.random.default_rng(42).standard_normal((2, n_samples)).astype(np.float32)
    return ChannelFrame.from_numpy(data, sampling_rate=sr, ch_labels=["ch0", "ch1"])


@pytest.fixture
def mono_frame():
    """Single-channel frame."""
    sr = 16000
    data = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    return ChannelFrame.from_numpy(data, sampling_rate=sr, label="mono_440Hz")


@pytest.fixture
def stereo_frame():
    """2-channel frame with different frequencies per channel."""
    sr = 16000
    t = np.arange(sr) / sr
    ch0 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    ch1 = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    data = np.stack([ch0, ch1])
    return ChannelFrame.from_numpy(data, sampling_rate=sr, ch_labels=["440Hz", "880Hz"])
```

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

新しい Frame メソッドを追加する場合、以下のテストカテゴリを必ず含めること:

### 1. Immutability Test
```python
def test_{method}_immutability(channel_frame):
    """Operation must not mutate the original frame."""
    original_data = channel_frame.data.copy()
    original_history_len = len(channel_frame.operation_history)

    result = channel_frame.{method}(...)

    assert result is not channel_frame
    np.testing.assert_array_equal(channel_frame.data, original_data)
    assert len(channel_frame.operation_history) == original_history_len
```

### 2. Metadata Propagation Test
```python
def test_{method}_metadata(channel_frame):
    """Operation must correctly update metadata and operation_history."""
    result = channel_frame.{method}(...)

    # Sampling rate rule: preserved unless resampling
    assert result.sampling_rate == channel_frame.sampling_rate
    # Channel count rule: preserved unless explicitly adding/removing
    assert result.n_channels == channel_frame.n_channels
    # History rule: exactly one new entry
    assert len(result.operation_history) == len(channel_frame.operation_history) + 1
    assert result.operation_history[-1]["operation"] == "{method}"
```

### 3. Lazy Evaluation Test
```python
def test_{method}_lazy(channel_frame):
    """Operation must preserve Dask lazy evaluation."""
    from dask.array.core import Array as DaArray
    result = channel_frame.{method}(...)
    assert isinstance(result._data, DaArray)
```

### 4. Chaining Test
```python
def test_{method}_chainable(channel_frame):
    """Operation result must support further method chaining."""
    result = channel_frame.{method}(...).normalize()
    assert isinstance(result, ChannelFrame)
    assert len(result.operation_history) >= 2
```

---

## Domain Transition Test Patterns

ドメインが変わる処理（`fft()`, `stft()`, `loudness()` 等）では追加の検証が必要:

```python
def test_fft_domain_transition(channel_frame):
    """Time -> Frequency domain transition."""
    spectral = channel_frame.fft()

    # Return type must change
    assert isinstance(spectral, SpectralFrame)
    # Frequency bins = N/2 + 1
    assert spectral.n_samples == channel_frame.n_samples // 2 + 1
    # Sampling rate carried forward
    assert spectral.sampling_rate == channel_frame.sampling_rate
    # Operation history tracks the transition
    assert spectral.operation_history[-1]["operation"] == "fft"
```

---

## Channel Collection Test Patterns

マルチチャンネル操作のテストで特に注意すべき点:

```python
def test_add_channel_label_alignment(channel_frame):
    """Channel labels must stay aligned with data after add_channel."""
    new_data = np.ones((1, channel_frame.n_samples))
    result = channel_frame.add_channel(new_data, label="c")

    assert result.n_channels == channel_frame.n_channels + 1
    assert result.labels[-1] == "c"
    np.testing.assert_array_equal(result.data[-1], np.ones(channel_frame.n_samples))

def test_add_channel_duplicate_label_raises(channel_frame):
    """Duplicate labels must be rejected unless suffix_on_dup is set."""
    with pytest.raises(ValueError, match=r"Duplicate channel label"):
        channel_frame.add_channel(np.zeros(channel_frame.n_samples), label=channel_frame.labels[0])

def test_add_channel_length_mismatch_strict(channel_frame):
    """Length mismatch with align='strict' must raise."""
    with pytest.raises(ValueError, match=r"Data length mismatch"):
        channel_frame.add_channel(np.zeros(50))  # Different length
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
- [frames-design.prompt.md](frames-design.prompt.md) — Frame architecture and immutability rules
