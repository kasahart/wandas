---
description: "Processing test patterns: numerical accuracy, frequency-domain verification, and reference library comparison"
applyTo: "tests/processing/**"
---
# Wandas Test Policy: Processing (`tests/processing/`)

Processing テストは「数値計算の正確性」と「音響物理学の妥当性」を検証する層です。
`wandas/processing/` は純粋な数値ロジック層であり、frame のメタデータを扱いません。
このテストでは「数値が物理的に正しいか」と「外部権威との一致」に集中します。

**前提**: このファイルは [test-grand-policy.instructions.md](test-grand-policy.instructions.md) と同時に適用されます。

---

## Common Fixtures for Processing Tests

Processing 層の fixture は `(DaskArray, int)` タプルを返す（`_dask` suffix で命名）。
Frame を経由しない純粋な数値検証に使用する。

```python
import numpy as np
import pytest
from wandas.utils.dask_helpers import da_from_array as _da_from_array


@pytest.fixture
def sine_1khz_48k_dask():
    """1 kHz sine at 48 kHz SR — standard psychoacoustic test signal."""
    sr = 48000
    t = np.arange(sr) / sr
    data = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    return _da_from_array(data.reshape(1, -1), chunks=(1, -1)), sr


@pytest.fixture
def calibrated_sine_1khz_70dB_dask():
    """1 kHz sine at 70 dB SPL — standard IEC psychoacoustic test signal."""
    sr = 48000
    p_ref = 2e-5  # Pa
    level_db = 70
    amplitude = p_ref * 10 ** (level_db / 20) * np.sqrt(2)
    t = np.arange(sr) / sr
    data = (amplitude * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    return _da_from_array(data.reshape(1, -1), chunks=(1, -1)), sr


@pytest.fixture
def dual_tone_16k_dask():
    """50 Hz + 1000 Hz at 16 kHz SR — standard filter test signal."""
    sr = 16000
    t = np.arange(sr) / sr
    low = np.sin(2 * np.pi * 50 * t)
    high = np.sin(2 * np.pi * 1000 * t)
    data = (low + high).astype(np.float32)
    return _da_from_array(data.reshape(1, -1), chunks=(1, -1)), sr


@pytest.fixture
def impulse_16k_dask():
    """Unit impulse — useful for filter impulse response testing."""
    sr = 16000
    data = np.zeros(sr, dtype=np.float32)
    data[0] = 1.0
    return _da_from_array(data.reshape(1, -1), chunks=(1, -1)), sr
```

---

## Processing Module Test Strategy

### AudioOperation 基底クラス
- `__init__` でのパラメータバリデーション（WHAT/WHY/HOW エラーメッセージ）
- `process()` が DaskArray を受け取り DaskArray を返すこと
- `get_metadata_updates()` が正しい dict を返すこと
- `get_display_name()` がフォーマット通りの文字列を返すこと

### Per-Module Requirements

| Module | Reference Library | Key Verification |
|--------|------------------|-----------------|
| `filters.py` | `scipy.signal` | 周波数応答特性、遮断特性 |
| `spectral.py` | `scipy.signal`, `librosa` | FFT ピーク位置、STFT shape |
| `psychoacoustic.py` | `mosqito` | ラウドネス/ラフネス/シャープネス値 |
| `weighting.py` | `scipy.signal` | A/C 特性の周波数応答 |
| `temporal.py` | `numpy` | RMS, ゼロクロス率等の統計量 |
| `stats.py` | `numpy`, `scipy.stats` | 統計量の正確性 |
| `effects.py` | なし（理論値で検証） | フェード、クリッピング等 |

---

## Filter Tests: Frequency Domain Verification

フィルタテストでは、時間領域の波形ではなく **周波数領域での減衰量** を検証すること。

```python
# GOOD: Frequency-domain verification of filter behavior
def test_low_pass_attenuates_high_frequencies(dual_tone_16k_dask):
    """Low-pass filter must attenuate frequencies above cutoff."""
    dask_signal, sr = dual_tone_16k_dask
    signal_np = dask_signal.compute()[0]
    cutoff = 500  # Hz

    lpf = LowPassFilter(sampling_rate=sr, cutoff=cutoff)
    result = lpf.process(dask_signal).compute()

    # FFT of result
    spectrum = np.abs(np.fft.rfft(result[0]))
    freqs = np.fft.rfftfreq(len(signal_np), 1 / sr)

    # 50 Hz component should be preserved (within 3 dB)
    idx_50 = np.argmin(np.abs(freqs - 50))
    idx_1000 = np.argmin(np.abs(freqs - 1000))

    original_spectrum = np.abs(np.fft.rfft(signal_np))
    ratio_50 = spectrum[idx_50] / original_spectrum[idx_50]
    ratio_1000 = spectrum[idx_1000] / original_spectrum[idx_1000]

    assert ratio_50 > 0.7  # Passband: minimal attenuation
    assert ratio_1000 < 0.1  # Stopband: significant attenuation
```

### Filter Edge Cases
```python
def test_filter_cutoff_at_nyquist_raises():
    """Cutoff at or above Nyquist frequency must raise ValueError."""
    sr = 16000
    with pytest.raises(ValueError, match=r"cutoff"):
        LowPassFilter(sampling_rate=sr, cutoff=sr / 2)

def test_filter_cutoff_zero_raises():
    """Zero cutoff frequency must raise ValueError."""
    sr = 16000
    with pytest.raises(ValueError, match=r"cutoff"):
        LowPassFilter(sampling_rate=sr, cutoff=0)
```

---

## Spectral Tests: Known-Signal Verification

FFT/STFT テストでは、解析的に予測可能な信号を使用すること。

```python
# GOOD: FFT peak detection with known sine
def test_fft_peak_frequency(sine_1khz_48k_dask):
    """FFT of pure sine must show peak at the sine frequency."""
    dask_signal, sr = sine_1khz_48k_dask
    n_samples = sr  # 1 second
    freq = 1000.0

    fft_op = FFTOperation(sampling_rate=sr, n_fft=n_samples)
    result = fft_op.process(dask_signal).compute()

    magnitudes = np.abs(result[0])
    peak_bin = np.argmax(magnitudes)
    freq_resolution = sr / n_samples
    detected_freq = peak_bin * freq_resolution

    assert abs(detected_freq - freq) < freq_resolution  # Within 1 bin

# GOOD: STFT output shape verification
def test_stft_output_shape():
    """STFT output shape must match theoretical dimensions."""
    sr = 16000
    n_samples = 16000
    n_fft = 1024
    hop_length = 512
    signal = np.zeros((1, n_samples), dtype=np.float32)
    dask_signal = _da_from_array(signal, chunks=(1, -1))

    stft_op = STFTOperation(sampling_rate=sr, n_fft=n_fft, hop_length=hop_length)
    result = stft_op.process(dask_signal).compute()

    assert result.shape[1] == n_fft // 2 + 1  # Frequency bins
```

---

## Psychoacoustic Tests: MoSQITo Reference Verification

心理音響テストでは、MoSQITo ライブラリとの等価性を検証すること。

```python
# GOOD: MoSQITo reference comparison
from mosqito.sq_metrics import loudness_zwtv

def test_loudness_matches_mosqito(calibrated_sine_1khz_70dB_dask):
    """Wandas loudness must match MoSQITo implementation."""
    dask_signal, sr = calibrated_sine_1khz_70dB_dask
    signal_np = dask_signal.compute()[0]

    # MoSQITo reference
    n_mosqito, _, _, _ = loudness_zwtv(signal_np, sr, field_type="free")

    # Wandas computation
    loudness_op = LoudnessZwtv(sampling_rate=sr, field_type="free")
    result = loudness_op.process(dask_signal).compute()

    # Perceptual metrics: 1% relative tolerance
    np.testing.assert_allclose(
        result.mean(), n_mosqito.mean(),
        rtol=0.01,  # Psychoacoustic model approximation tolerance
    )
```

### Psychoacoustic Test Signal Standards
心理音響テストで使用する信号の標準:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sampling rate | 48000 Hz | MoSQITo/IEC 規格標準 |
| Reference pressure | 2e-5 Pa | 音響学標準基準値 |
| Test level | 70 dB SPL | 聴覚の線形応答領域 |
| Test frequency | 1000 Hz | A 特性の 0 dB 基準点 |
| Duration | >= 1.0 s | Slow 時定数の安定化に必要 |

---

## A-Weighting Tests: Known Frequency Response

```python
def test_a_weighting_at_reference_frequencies(sine_1khz_48k_dask):
    """A-weighting must match standard values at key frequencies."""
    dask_signal, sr = sine_1khz_48k_dask
    signal_np = dask_signal.compute()[0]

    aw = AWeighting(sampling_rate=sr)
    result = aw.process(dask_signal).compute()

    # At 1 kHz, A-weighting should have ~0 dB gain
    rms_in = np.sqrt(np.mean(signal_np**2))
    rms_out = np.sqrt(np.mean(result[0]**2))
    gain_db = 20 * np.log10(rms_out / rms_in)

    assert abs(gain_db - 0.0) < 0.5  # Within 0.5 dB at 1 kHz reference
```

---

## Operation Registration & Display Name Tests

```python
def test_operation_registered():
    """Operation must be discoverable via create_operation."""
    from wandas.processing import create_operation
    op = create_operation("low_pass_filter", sampling_rate=16000, cutoff=1000)
    assert op is not None

def test_display_name_format():
    """Display name must follow convention: 'OpName(key_param=value)'."""
    op = LowPassFilter(sampling_rate=16000, cutoff=1000)
    name = op.get_display_name()
    assert "LPF" in name or "low_pass" in name.lower()
    assert "1000" in name
```

---

## Anti-Patterns Specific to Processing Tests

```python
# BAD: Testing filter in time domain only
def test_filter_bad():
    result = lpf.process(signal)
    assert result is not None  # Tells us nothing about filter quality

# BAD: Comparing against self-computed reference (circular logic)
expected = my_low_pass(signal, cutoff)  # Your own code is not an "authority"
np.testing.assert_allclose(result, expected)

# BAD: Psychoacoustic test without MoSQITo reference
def test_loudness_bad():
    result = loudness_op.process(signal)
    assert result.mean() > 0  # Says nothing about correctness

# GOOD: Use scipy/librosa/mosqito as external authority
expected = scipy.signal.sosfilt(sos, signal)  # External reference
np.testing.assert_allclose(result, expected, rtol=1e-6)
```

---

## Cross-References
- [test-grand-policy.instructions.md](test-grand-policy.instructions.md) — 4 pillars and test pyramid
- [processing-api.prompt.md](processing-api.prompt.md) — Processing layer architecture
