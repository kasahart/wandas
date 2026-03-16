---
description: "I/O test patterns: round-trip fidelity, metadata preservation, and format-specific contracts (WAV/WDF/CSV)"
applyTo: "tests/io/**"
---
# Wandas Test Policy: I/O (`tests/io/`)

I/O テストは「データ保全性」と「メタデータのラウンドトリップ完全性」を検証する層です。
ファイルの読み書きにおいて、一切のデータ欠損やメタデータ喪失があってはなりません。

**前提**: このファイルは [test-grand-policy.instructions.md](test-grand-policy.instructions.md) と同時に適用されます。

---

## Common Fixtures for I/O Tests

```python
import numpy as np
import pytest
from wandas.frames.channel import ChannelFrame


@pytest.fixture
def known_signal_frame():
    """2-channel frame for round-trip testing.

    Note: Seeded random data is acceptable here because the oracle is
    byte/data equality (not analytic expectation).
    """
    data = np.random.default_rng(42).standard_normal((2, 16000)).astype(np.float32)
    return ChannelFrame.from_numpy(
        data, sampling_rate=44100,
        ch_labels=["left", "right"],
        ch_units=["Pa", "Pa"],
    )


@pytest.fixture
def create_test_wav(tmp_path):
    """Factory fixture for creating int16 PCM WAV files."""
    def _create(sr=44100, n_channels=1, n_samples=44100):
        import scipy.io.wavfile as wav
        data = np.random.default_rng(42).integers(
            -(2**15), 2**15,
            size=(n_samples, n_channels), dtype=np.int16
        )
        if n_channels == 1:
            data = data.flatten()
        path = tmp_path / "test.wav"
        wav.write(str(path), sr, data)
        return path
    return _create
```

---

## I/O Test Strategy

### Format-Specific Requirements

| Format | Module | Key Concerns |
|--------|--------|-------------|
| WAV | `wav_io.py` | PCM 精度、正規化、サンプリングレート、チャンネル数 |
| WDF (HDF5) | `wdf_io.py` | 完全メタデータ保存、圧縮、dtype 変換 |
| CSV | `readers.py` | 時間列解釈、デリミタ、ヘッダー処理 |

---

## Core Pattern: Round-Trip Test

すべての I/O フォーマットに対して「書き込み → 読み込み → 元データと比較」のパターンを必須とする。

```python
def test_wdf_roundtrip_preserves_everything(known_signal_frame, tmp_path):
    cf_processed = known_signal_frame.normalize()
    path = tmp_path / "test.wdf"
    cf_processed.save(path)
    loaded = ChannelFrame.load(path)

    np.testing.assert_allclose(loaded.data, cf_processed.data, rtol=1e-6)
    assert loaded.sampling_rate == cf_processed.sampling_rate
    assert loaded.labels == cf_processed.labels
    assert loaded.operation_history == cf_processed.operation_history
```

---

## WAV I/O Tests

### Float Round-Trip
`ChannelFrame.to_wav` writes float data as IEEE FLOAT subtype when data is in `[-1, 1]`, so no PCM quantization occurs.

```python
def test_wav_float_write_read_preserves_data(tmp_path):
    sr = 44100
    data = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    cf = ChannelFrame.from_numpy(data, sampling_rate=sr)
    cf.to_wav(tmp_path / "test.wav")
    loaded = ChannelFrame.read_wav(str(tmp_path / "test.wav"), normalize=True)
    np.testing.assert_allclose(loaded.data, cf.data, atol=1e-6)  # Float WAV: near-exact
```

### PCM Round-Trip
PCM files (e.g., written with `scipy.io.wavfile.write` as int16) should use `normalize=False` to compare raw samples.

```python
def test_wav_pcm_read_roundtrip(create_test_wav):
    wav_path = create_test_wav()
    import scipy.io.wavfile as wav
    _, expected = wav.read(str(wav_path))
    cf = ChannelFrame.read_wav(str(wav_path), normalize=False)
    np.testing.assert_array_equal(cf.data[0].astype(np.int16), expected)
```

### Channel Count
```python
def test_wav_stereo_channel_count(create_test_wav):
    assert ChannelFrame.read_wav(str(create_test_wav(n_channels=2))).n_channels == 2

def test_wav_mono_channel_count(create_test_wav):
    assert ChannelFrame.read_wav(str(create_test_wav(n_channels=1))).n_channels == 1
```

---

## WDF I/O Tests

WDF は Wandas のネイティブフォーマットであり、全メタデータの保存を保証する。

```python
def test_wdf_preserves_channel_metadata(tmp_path):
    cf = ChannelFrame.from_numpy(
        np.zeros((2, 100), dtype=np.float32),
        sampling_rate=16000, ch_labels=["mic1", "mic2"], ch_units=["Pa", "Pa"],
    )
    cf.save(tmp_path / "test.wdf")
    loaded = ChannelFrame.load(tmp_path / "test.wdf")
    assert loaded.labels == ["mic1", "mic2"]

def test_wdf_overwrite_protection(tmp_path):
    cf = ChannelFrame.from_numpy(np.zeros((1, 100), dtype=np.float32), sampling_rate=16000)
    cf.save(tmp_path / "test.wdf")
    with pytest.raises(FileExistsError):
        cf.save(tmp_path / "test.wdf", overwrite=False)
```

---

## CSV I/O Tests

```python
def test_csv_read_with_header(tmp_path):
    (tmp_path / "test.csv").write_text("time,sensor_a,sensor_b\n0.0,1.0,2.0\n0.001,1.1,2.1\n")
    cf = ChannelFrame.read_csv(str(tmp_path / "test.csv"), time_column=0)
    assert cf.n_channels == 2
```

---

## File Error Handling Tests

```python
def test_read_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        ChannelFrame.from_file("/nonexistent/path/audio.wav")
```

---

## Lazy Loading Verification

```python
def test_wav_lazy_loading(create_test_wav):
    from dask.array.core import Array as DaArray
    cf = ChannelFrame.read_wav(str(create_test_wav()))
    assert isinstance(cf._data, DaArray)
```

---

## Cross-References
- [test-grand-policy.instructions.md](test-grand-policy.instructions.md) — 4 pillars and test pyramid
- [io-contracts.instructions.md](io-contracts.instructions.md) — I/O metadata preservation contracts
