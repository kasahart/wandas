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
また、Wrapper Equivalence として SciPy の `filtfilt(b, a, x)` と比較すること（実装が `filtfilt(b, a, x, axis=1)` を使用するため）。

```python
def test_low_pass_attenuates_high_frequencies(dual_tone_16k_dask):
    dask_signal, sr = dual_tone_16k_dask
    lpf = LowPassFilter(sampling_rate=sr, cutoff=500)
    result = lpf.process(dask_signal).compute()

    spectrum_out = np.abs(np.fft.rfft(result[0]))
    freqs = np.fft.rfftfreq(dask_signal.shape[1], 1 / sr)
    spectrum_in = np.abs(np.fft.rfft(dask_signal.compute()[0]))

    # 50 Hz preserved; 1000 Hz attenuated
    assert spectrum_out[np.argmin(np.abs(freqs - 50))] / spectrum_in[np.argmin(np.abs(freqs - 50))] > 0.7
    assert spectrum_out[np.argmin(np.abs(freqs - 1000))] / spectrum_in[np.argmin(np.abs(freqs - 1000))] < 0.1
```

### Filter Edge Cases
```python
def test_filter_cutoff_at_nyquist_raises():
    with pytest.raises(ValueError):
        LowPassFilter(sampling_rate=16000, cutoff=8000)  # Nyquist
```

---

## Spectral Tests: Known-Signal Verification

FFT/STFT テストでは、解析的に予測可能な信号を使用すること。
注意: クラス名は `FFT` および `STFT`（`FFTOperation` / `STFTOperation` ではない）。

```python
def test_fft_peak_frequency(sine_1khz_48k_dask):
    dask_signal, sr = sine_1khz_48k_dask
    fft_op = FFT(sampling_rate=sr, n_fft=sr)  # n_fft=sr → 1 Hz per bin
    result = fft_op.process(dask_signal).compute()
    peak_bin = np.argmax(np.abs(result[0]))
    assert abs(peak_bin - 1000) < 1  # 1 kHz sine → peak at bin 1000

def test_stft_output_shape(sine_1khz_48k_dask):
    dask_signal, sr = sine_1khz_48k_dask
    stft_op = STFT(sampling_rate=sr, n_fft=1024, hop_length=512)
    result = stft_op.process(dask_signal).compute()
    assert result.shape[1] == 1024 // 2 + 1  # Frequency bins
```

---

## Psychoacoustic Tests: MoSQITo Reference Verification

心理音響テストでは、MoSQITo ライブラリとの等価性を検証すること。
標準信号: SR=48000 Hz、70 dB SPL、1 kHz、1秒以上（Slow 時定数の安定化に必要）。

```python
from mosqito.sq_metrics import loudness_zwtv

def test_loudness_matches_mosqito(calibrated_sine_1khz_70dB_dask):
    dask_signal, sr = calibrated_sine_1khz_70dB_dask
    signal_np = dask_signal.compute()[0]
    n_mosqito, _, _, _ = loudness_zwtv(signal_np, sr, field_type="free")

    loudness_op = LoudnessZwtv(sampling_rate=sr, field_type="free")
    result = loudness_op.process(dask_signal).compute()

    np.testing.assert_allclose(result.mean(), n_mosqito.mean(), rtol=0.01)  # 1% tolerance
```

---

## A-Weighting Tests: Known Frequency Response

```python
def test_a_weighting_at_1khz(sine_1khz_48k_dask):
    dask_signal, sr = sine_1khz_48k_dask
    signal_np = dask_signal.compute()[0]
    result = AWeighting(sampling_rate=sr).process(dask_signal).compute()

    rms_in = np.sqrt(np.mean(signal_np**2))
    rms_out = np.sqrt(np.mean(result[0]**2))
    gain_db = 20 * np.log10(rms_out / rms_in)
    assert abs(gain_db) < 0.5  # A-weighting is ~0 dB at 1 kHz
```

---

## Operation Registration & Display Name Tests

注意: `create_operation` のキーは **レジストリ名**（例: `"lowpass_filter"`）でメソッド名（`"low_pass_filter"`）とは異なる。
`LowPassFilter.get_display_name()` は `"lpf"` を返す（パラメータは含まない）。

```python
def test_lowpass_filter_registered():
    from wandas.processing import create_operation
    op = create_operation("lowpass_filter", sampling_rate=16000, cutoff=1000)
    assert op is not None

def test_lowpass_display_name():
    op = LowPassFilter(sampling_rate=16000, cutoff=1000)
    assert op.get_display_name() == "lpf"
```

---

## Anti-Patterns Specific to Processing Tests

```python
# BAD: Testing filter in time domain only
assert result is not None  # Tells us nothing about filter quality

# BAD: Circular reference — own code is not an "authority"
expected = my_low_pass(signal, cutoff)
np.testing.assert_allclose(result, expected)

# BAD: Psychoacoustic test without MoSQITo reference
assert result.mean() > 0  # Says nothing about correctness
```

---

## Cross-References
- [test-grand-policy.instructions.md](test-grand-policy.instructions.md) — 4 pillars and test pyramid
- [processing-api.instructions.md](processing-api.instructions.md) — Processing layer architecture
