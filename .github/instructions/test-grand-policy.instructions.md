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
いかなる演算（フィルタリング、窓関数適用等）を行っても、元のインスタンスの内部データやメタデータが変更されてはならない。
常に「新しいインスタンス」が生成され、元のデータが破壊されていないことをアサートする。

```python
# GOOD: Immutability verification pattern
def test_operation_preserves_original(channel_frame):
    original_data = channel_frame.data.copy()
    original_sr = channel_frame.sampling_rate
    original_labels = [ch.label for ch in channel_frame._channel_metadata]

    result = channel_frame.normalize()

    # Original must be untouched
    np.testing.assert_array_equal(channel_frame.data, original_data)
    assert channel_frame.sampling_rate == original_sr
    assert [ch.label for ch in channel_frame._channel_metadata] == original_labels
    # Result must be a NEW instance
    assert result is not channel_frame
```

### Rule: Dask Graph Protection
Dask による遅延評価を維持するため、テスト中に意図しない `.compute()` が実行されていないかを監視する。

```python
# GOOD: Lazy evaluation verification pattern
from unittest import mock
from dask.array.core import Array as DaArray

def test_operation_stays_lazy(channel_frame):
    with mock.patch.object(DaArray, "compute", wraps=channel_frame._data.compute) as mock_compute:
        result = channel_frame.low_pass_filter(cutoff=1000)
        # Operation setup must NOT trigger compute
        mock_compute.assert_not_called()
        # Explicit compute triggers evaluation
        _ = result.data
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
物理単位（Unit）、サンプリングレート、基準値などのメタデータが、演算後も正確に継承または変換されていること。

```python
# GOOD: Metadata propagation test
def test_filter_preserves_metadata(channel_frame):
    result = channel_frame.low_pass_filter(cutoff=1000)

    # Sampling rate must survive non-resampling operations
    assert result.sampling_rate == channel_frame.sampling_rate
    # Channel count preserved
    assert result.n_channels == channel_frame.n_channels
    # Labels propagated (with operation annotation)
    assert len(result._channel_metadata) == len(channel_frame._channel_metadata)
```

### Rule: Operation History Traceability
`operation_history` に「どのようなパラメータで処理が行われたか」がアトミックに記録されていること。

```python
# GOOD: Operation history verification
def test_operation_history_recorded(channel_frame):
    result = channel_frame.low_pass_filter(cutoff=1000)

    assert len(result.operation_history) == len(channel_frame.operation_history) + 1
    last_op = result.operation_history[-1]
    assert last_op["operation"] == "low_pass_filter"
    assert "params" in last_op
    assert last_op["params"]["cutoff"] == 1000
```

### Rule: Domain Transition Metadata
ドメインが変わる処理（時間領域 → 周波数領域 → 時間-周波数領域）では、メタデータの変換が正しいことを検証する。

```python
# GOOD: Domain transition metadata check
def test_fft_metadata_transition(channel_frame):
    spectral = channel_frame.fft()

    # Frequency axis must be correct
    expected_freq_bins = channel_frame.n_samples // 2 + 1
    assert spectral.n_samples == expected_freq_bins
    # Sampling rate carried forward
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
STFT/ISTFT 等の変換において、逆変換によって元のドメインに戻した際の再現性。

```python
# GOOD: Round-trip verification
def test_stft_istft_roundtrip(channel_frame):
    stft_result = channel_frame.stft(n_fft=1024, hop_length=512)
    reconstructed = stft_result.istft()

    np.testing.assert_allclose(
        reconstructed.data[:, :channel_frame.n_samples],
        channel_frame.data,
        atol=1e-6,
        rtol=1e-6,
    )
```

### Rule: Shape & Type Consistency
配列の形状（Shape）やデータ型が理論値と一致すること。

```python
# GOOD: Shape validation
def test_stft_output_shape(channel_frame):
    n_fft = 1024
    hop_length = 512
    result = channel_frame.stft(n_fft=n_fft, hop_length=hop_length)

    expected_freq_bins = n_fft // 2 + 1
    expected_time_frames = (channel_frame.n_samples - n_fft) // hop_length + 1
    assert result.data.shape[-2] == expected_freq_bins
    # Time frames within acceptable range (windowing edge effects)
    assert abs(result.data.shape[-1] - expected_time_frames) <= 2
```

### Rule: Sampling Rate Constraint Enforcement
異なるサンプリングレート間での不正な演算には、適切な例外をスローすること。

```python
# GOOD: Physical constraint enforcement
def test_mismatched_sampling_rate_raises(channel_frame):
    other = ChannelFrame.from_numpy(
        np.zeros((1, 100)), sampling_rate=22050  # Different rate
    )
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
Wandas が Librosa や SciPy 等のラッパーである場合、外部ライブラリを直接呼び出した結果と比較する。

```python
# GOOD: Reference library comparison
import scipy.signal

def test_low_pass_matches_scipy(channel_frame):
    cutoff = 1000
    result = channel_frame.low_pass_filter(cutoff=cutoff)

    # Direct scipy computation for reference
    sos = scipy.signal.butter(4, cutoff, btype="low", fs=channel_frame.sampling_rate, output="sos")
    expected = scipy.signal.sosfilt(sos, channel_frame.data)

    np.testing.assert_array_equal(result.data, expected)
```

### Rule: Theoretical Value Verification
自前でアルゴリズムを実装する場合、解析解や物理法則に基づく理論値と比較する。

```python
# GOOD: Known-signal verification using standard fixture
def test_fft_of_known_sine(pure_sine_1khz):
    """FFT of a pure sine wave must show a single peak at the correct frequency."""
    spectral = pure_sine_1khz.fft()

    magnitudes = np.abs(spectral.data[0])
    peak_bin = np.argmax(magnitudes)
    freq_resolution = pure_sine_1khz.sampling_rate / pure_sine_1khz.n_samples
    detected_freq = peak_bin * freq_resolution

    # Peak must be at 1000 Hz (within 1 bin tolerance)
    assert abs(detected_freq - 1000.0) < freq_resolution
```

### Rule: Tolerance Standards
浮動小数点演算の性質を考慮した許容誤差の基準。

| Category | Default Tolerance | Rationale |
|----------|------------------|-----------|
| Wrapper equivalence | `array_equal` | Same algorithm |
| Round-trip transforms | `atol=1e-6` | Windowing/overlap introduces small errors |
| Psychoacoustic metrics | `rtol=0.01` (1%) | Perceptual models have inherent approximations |
| Frequency peak detection | Within 1 FFT bin | Spectral leakage at bin boundaries |
| dB values | `atol=0.1` dB | Log-scale magnifies small differences |

```python
# GOOD: Explicit tolerance with rationale
np.testing.assert_array_equal(result, expected)  # Wrapper equivalence: same scipy algorithm underneath
```

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
- **例**: `cf.low_pass_filter()` の結果が `scipy.signal.sosfilt()` と一致する

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
def test_low_pass_filter_below_nyquist_preserves_signal(): ...
def test_low_pass_filter_above_nyquist_raises_error(): ...
def test_fft_pure_sine_peak_at_correct_frequency(): ...
```

### Error Message Assertions
```python
# Match first line only — resilient to detail changes
with pytest.raises(ValueError, match=r"Sampling rate mismatch"):
    ...

# NOT this — brittle to message format changes
with pytest.raises(ValueError, match=r"Sampling rate mismatch.*Left operand.*Right operand"):
    ...
```

### Dask Array Creation
```python
from wandas.utils.dask_helpers import da_from_array as _da_from_array

# Always use channel-wise chunks
dask_data = _da_from_array(numpy_array, chunks=(1, -1))
```

---

## Anti-Patterns (Avoid These)

```python
# BAD: Testing implementation details instead of behavior
assert result._internal_buffer is not None  # Don't test private state

# BAD: No tolerance specified for floating point comparison
assert result.data[0, 0] == 0.5  # Will fail due to float precision

# BAD: Testing with random data — not reproducible, not analytically verifiable
data = np.random.randn(1000)  # Use deterministic known signals instead

# BAD: Verifying only "no error" — test the actual output
result = channel_frame.normalize()
assert result is not None  # This tells us nothing useful

# BAD: Hardcoded magic numbers without explanation
np.testing.assert_allclose(result, expected, atol=0.0001)  # Why this tolerance?
```

---

## Cross-References
- [frames-design.prompt.md](frames-design.prompt.md) — Frame immutability and metadata rules
- [processing-api.prompt.md](processing-api.prompt.md) — Processing layer responsibilities
- [io-contracts.prompt.md](io-contracts.prompt.md) — I/O metadata preservation contracts
- [testing-workflow.prompt.md](testing-workflow.prompt.md) — TDD workflow and quality commands
