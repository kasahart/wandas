---
description: "Test grand policy: 4 pillars (immutability, metadata sync, math consistency, reference verification) and signal processing test pyramid"
applyTo: "tests/**"
---
# Wandas Test Grand Policy: AI-Driven Acoustic Testing

Wandas のテストコードは単なるバグ検出器ではなく、**「音響物理学的な正しさ」と「計算グラフの整合性」**を AI エージェントに担保させるためのガードレールです。
AI によるハルシネーション（もっともらしい計算ミスの生成）を防ぐため、以下の **4 つの柱** と **検証のピラミッド** を遵守してください。

このポリシーは `tests/**` 内のすべてのテストファイルに自動適用されます（`.instructions.md` + `applyTo` による自動注入）。

---

## Common Fixtures — Standard Test Signals

テスト用信号は解析的に予測可能なものを使用する。各コンポーネントポリシーは以下の基準 fixture を拡張する形で定義すること。

### Fixture Naming Convention

| Suffix | Return Type | Usage Layer |
|--------|------------|-------------|
| (なし) | `ChannelFrame` | Frame / Visualization テスト |
| `_dask` | `(DaskArray, int)` | Processing テスト（frame を経由しない数値検証） |

### Standard Fixtures

```python
import numpy as np
import pytest
from wandas.frames.channel import ChannelFrame
from wandas.utils.dask_helpers import da_from_array as _da_from_array


@pytest.fixture
def pure_sine_1khz():
    """1 kHz pure sine wave — FFT peak is analytically predictable."""
    sr = 16000
    t = np.arange(sr) / sr  # 1 second
    data = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    return ChannelFrame.from_numpy(data, sampling_rate=sr, label="sine_1kHz")


@pytest.fixture
def dual_tone():
    """50 Hz + 1000 Hz — useful for filter attenuation tests."""
    sr = 16000
    t = np.arange(sr) / sr
    data = (np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    return ChannelFrame.from_numpy(data, sampling_rate=sr, label="dual_tone")


@pytest.fixture
def calibrated_sine_1khz_70dB():
    """1 kHz sine at 70 dB SPL — standard IEC psychoacoustic test signal."""
    sr = 48000
    p_ref = 2e-5  # Pa
    level_db = 70
    amplitude = p_ref * 10 ** (level_db / 20) * np.sqrt(2)
    t = np.arange(sr) / sr
    data = (amplitude * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    return ChannelFrame.from_numpy(data, sampling_rate=sr, label="calibrated_1kHz_70dB")


@pytest.fixture
def impulse():
    """Unit impulse — useful for filter impulse response testing."""
    sr = 16000
    data = np.zeros(sr, dtype=np.float32)
    data[0] = 1.0
    return ChannelFrame.from_numpy(data, sampling_rate=sr, label="impulse")
```

---

## Pillar 1: Data Immutability & Lazy Evaluation Integrity

### Rule: Side-Effect-Free Operations
いかなる演算を行っても、元のインスタンスの内部データやメタデータが変更されてはならない。

```python
def test_operation_preserves_original(channel_frame):
    original_data = channel_frame.data.copy()
    result = channel_frame.normalize()
    np.testing.assert_array_equal(channel_frame.data, original_data)  # Original unchanged
    assert result is not channel_frame  # New instance returned
```

### Rule: Dask Graph Protection
Dask による遅延評価を維持するため、演算後の `result._data` が `DaskArray` であることを確認する。

```python
from dask.array.core import Array as DaArray

def test_operation_stays_lazy(channel_frame):
    result = channel_frame.low_pass_filter(cutoff=1000)
    assert isinstance(result._data, DaArray)  # Must remain lazy
```

### Checklist
- [ ] Operation returns a new instance (`assert result is not original`)
- [ ] Original data unchanged after operation (`assert_array_equal`)
- [ ] Original metadata unchanged (sampling_rate, labels, operation_history)
- [ ] Dask graph preserved (no premature `.compute()`)
- [ ] `result._data` is a `DaskArray` instance after operation

---

## Pillar 2: Physical Domain Metadata Synchronization

### Rule: Metadata Auto-Tracking
演算後もサンプリングレート・チャンネル数・ラベルが正確に継承または変換されていること。

```python
def test_filter_preserves_metadata(channel_frame):
    result = channel_frame.low_pass_filter(cutoff=1000)
    assert result.sampling_rate == channel_frame.sampling_rate
    assert result.n_channels == channel_frame.n_channels
```

### Rule: Operation History Traceability
`operation_history` に処理内容とパラメータが記録されていること。
注意: `operation_history` に記録されるのはメソッド名ではなく **レジストリキー**（例: `"lowpass_filter"`）。

```python
def test_operation_history_recorded(channel_frame):
    result = channel_frame.low_pass_filter(cutoff=1000)
    assert len(result.operation_history) == len(channel_frame.operation_history) + 1
    last_op = result.operation_history[-1]
    assert last_op["operation"] == "lowpass_filter"  # registry name, not method name
    assert last_op["params"]["cutoff"] == 1000
```

### Rule: Domain Transition Metadata
ドメインが変わる処理では、メタデータの変換が正しいことを検証する。

```python
def test_fft_metadata_transition(channel_frame):
    spectral = channel_frame.fft()
    assert spectral.n_samples == channel_frame.n_samples // 2 + 1  # N/2 + 1 bins
    assert spectral.sampling_rate == channel_frame.sampling_rate
```

### Checklist
- [ ] Sampling rate preserved (or correctly updated for resampling)
- [ ] Channel labels propagated with operation annotation
- [ ] `operation_history` grows by exactly 1 entry per operation
- [ ] History entry contains correct operation name and params
- [ ] Domain transition produces correct axis dimensions

---

## Pillar 3: Mathematical Consistency & Transform Reversibility

### Rule: Domain Round-Trip Fidelity

```python
def test_stft_istft_roundtrip(channel_frame):
    reconstructed = channel_frame.stft(n_fft=1024, hop_length=512).istft()
    np.testing.assert_allclose(
        reconstructed.data[:, :channel_frame.n_samples], channel_frame.data, atol=1e-6
    )
```

### Rule: Shape & Type Consistency

```python
def test_stft_output_shape(channel_frame):
    result = channel_frame.stft(n_fft=1024, hop_length=512)
    assert result.data.shape[-2] == 1024 // 2 + 1  # Frequency bins
```

### Rule: Sampling Rate Constraint Enforcement

```python
def test_mismatched_sampling_rate_raises(channel_frame):
    other = ChannelFrame.from_numpy(np.zeros((1, 100)), sampling_rate=22050)
    with pytest.raises(ValueError, match=r"Sampling rate mismatch"):
        channel_frame + other
```

### Checklist
- [ ] Round-trip transforms recover original within tolerance
- [ ] Output shapes match theoretical values
- [ ] Data types are correct (real for time, complex for spectral)
- [ ] Invalid cross-rate operations raise clear errors

---

## Pillar 4: Numerical Validity — Reference-Based Verification

AI エージェントは複雑な信号処理の数式において、論理的に正しく見える「誤った計算コード」を生成するリスクがある。
これを防ぐため、以下の検証パラダイムを必須とする。

### Rule: Wrapper Equivalence Testing
Wandas が SciPy 等のラッパーである場合、外部ライブラリを直接呼び出した結果と比較する。
例: `LowPassFilter` は `scipy.signal.butter(b/a) + filtfilt` を使用しているため:

```python
def test_low_pass_matches_scipy(channel_frame):
    cutoff = 1000
    result = channel_frame.low_pass_filter(cutoff=cutoff)

    # Mirror the Wandas implementation: butter(b/a) + filtfilt
    nyquist = channel_frame.sampling_rate / 2
    b, a = scipy.signal.butter(4, cutoff / nyquist, btype="low")
    expected = scipy.signal.filtfilt(b, a, channel_frame.data)

    np.testing.assert_allclose(result.data, expected)  # Same algorithm: exact match
```

### Rule: Theoretical Value Verification
自前でアルゴリズムを実装する場合、解析的に予測可能な信号を使用して理論値と比較する。

```python
def test_fft_of_known_sine(pure_sine_1khz):
    spectral = pure_sine_1khz.fft()
    magnitudes = np.abs(spectral.data[0])
    peak_bin = np.argmax(magnitudes)
    freq_resolution = pure_sine_1khz.sampling_rate / pure_sine_1khz.n_samples
    assert abs(peak_bin * freq_resolution - 1000.0) < freq_resolution  # Within 1 bin
```

### Rule: Tolerance Standards

| Category | Default Tolerance | Rationale |
|----------|------------------|-----------|
| Wrapper equivalence | `assert_allclose` (default rtol=1e-7) | Same algorithm, exact numeric result |
| Round-trip transforms | `atol=1e-6` | Windowing/overlap introduces small errors |
| Psychoacoustic metrics | `rtol=0.01` (1%) | Perceptual models have inherent approximations |
| Frequency peak detection | Within 1 FFT bin | Spectral leakage at bin boundaries |
| dB values | `atol=0.1` dB | Log-scale magnifies small differences |

### Checklist
- [ ] Wrapper operations compared against direct library calls
- [ ] Custom algorithms verified against theoretical values or known signals
- [ ] Tolerance explicitly specified with rationale comment
- [ ] Known-signal tests use analytically predictable inputs (pure sine, impulse, DC)

---

## Signal Processing Test Pyramid

テストケースを以下の 3 層で構成する。上位層ほどテスト数は少なくてよいが、欠かしてはならない。

### Layer 1: Unit Tests (Logic) — Base
- **対象**: クラスのメソッド単体、ユーティリティ関数
- **目的**: 境界値テスト、型チェック、基本的なデータフローの正当性確認
- **例**: `validate_sampling_rate(0)` が ValueError を返す、`ChannelFrame(3D_array)` が拒否される

### Layer 2: Domain Tests (Physics) — Middle
- **対象**: 物理モデル、JIS/IEC 規格準拠のアルゴリズム
- **目的**: 時定数（Fast/Slow）の応答特性、フィルタの遮断特性など、音響工学としての物理的妥当性の保証
- **例**: ローパスフィルタがカットオフ周波数以上を減衰させる、A 特性重み付けが 1kHz で 0dB

### Layer 3: Integration Tests (Wrapper & Authority) — Top
- **対象**: 外部ライブラリとの連携、エンドツーエンドの計算パイプライン
- **目的**: 既存ライブラリや理論式という「外部の権威」との比較による、実装の正当性の裏取り
- **例**: `cf.low_pass_filter()` の結果が `scipy.signal.filtfilt()` と一致する

### Layer Distribution Guideline
新しい処理を追加する場合、最低限以下のテスト構成を確保すること:
- **Unit**: 入力バリデーション + エラーケース 2-3 件
- **Domain**: 物理的妥当性のスモークテスト 1-2 件（既知信号を使用）
- **Integration**: ラッパーなら等価性テスト 1 件、独自実装なら理論値テスト 1 件

---

## Test Conventions

### Naming
```python
# Pattern: test_{what}_{condition}_{expected_outcome}
def test_low_pass_filter_above_nyquist_raises_error(): ...
def test_fft_pure_sine_peak_at_correct_frequency(): ...
```

### Error Message Assertions
Match first line only — resilient to detail changes:
```python
with pytest.raises(ValueError, match=r"Sampling rate mismatch"):  # GOOD
    ...
```

### Dask Array Creation
```python
from wandas.utils.dask_helpers import da_from_array as _da_from_array
dask_data = _da_from_array(numpy_array, chunks=(1, -1))  # Always channel-wise chunks
```

---

## Anti-Patterns (Avoid These)

```python
# BAD: No tolerance for floating point
assert result.data[0, 0] == 0.5

# BAD: Random data — not analytically verifiable
data = np.random.randn(1000)  # Use deterministic known signals instead

# BAD: Verifying only "no error"
assert result is not None  # Test the actual output values

# BAD: Magic numbers without explanation
np.testing.assert_allclose(result, expected, atol=0.0001)  # Why this tolerance?
```

---

## Cross-References
- [frames-design.instructions.md](frames-design.instructions.md) — Frame immutability and metadata rules
- [processing-api.instructions.md](processing-api.instructions.md) — Processing layer responsibilities
- [io-contracts.instructions.md](io-contracts.instructions.md) — I/O metadata preservation contracts
- [testing-workflow.instructions.md](testing-workflow.instructions.md) — TDD workflow and quality commands
