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
    """2-channel frame with known data for round-trip testing."""
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

すべての I/O フォーマットに対して、Round-Trip テストを必須とする。
「書き込み → 読み込み → 元データと比較」のパターンで、データとメタデータの完全性を検証する。

```python
# GOOD: Complete round-trip with metadata verification
def test_wdf_roundtrip_preserves_everything(known_signal_frame, tmp_path):
    """WDF save/load must preserve all data and metadata."""
    cf_processed = known_signal_frame.normalize()

    # Save and reload
    path = tmp_path / "test.wdf"
    cf_processed.save(path)
    loaded = ChannelFrame.load(path)

    # Data fidelity
    np.testing.assert_allclose(
        loaded.data, cf_processed.data,
        rtol=1e-6,  # Float32 precision
    )
    # Metadata fidelity
    assert loaded.sampling_rate == cf_processed.sampling_rate
    assert loaded.n_channels == cf_processed.n_channels
    assert loaded.labels == cf_processed.labels
    # Operation history preserved
    assert loaded.operation_history == cf_processed.operation_history
```

---

## WAV I/O Tests

### Float Round-Trip Tests
`ChannelFrame.to_wav` writes float data as IEEE FLOAT subtype (not PCM), so round-trips for float32 data in `[-1, 1]` should have near-exact fidelity.

```python
def test_wav_float_write_read_preserves_data(tmp_path):
    """WAV float round-trip must preserve data with near-exact fidelity."""
    sr = 44100
    data = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    cf = ChannelFrame.from_numpy(data, sampling_rate=sr)

    path = tmp_path / "test.wav"
    cf.to_wav(path)
    loaded = ChannelFrame.read_wav(str(path), normalize=True)

    # Float WAV round-trip: near-exact fidelity (no PCM quantization)
    np.testing.assert_allclose(
        loaded.data, cf.data,
        atol=1e-6,  # Float32 precision tolerance
    )
    assert loaded.sampling_rate == sr
```

### PCM Round-Trip Tests
When testing PCM files (e.g., written by `scipy.io.wavfile.write` with int16), load with `normalize=False` to compare raw integer samples, or account for quantization when using `normalize=True`.

```python
def test_wav_pcm_read_roundtrip(tmp_path, create_test_wav):
    """PCM WAV read must recover original integer samples (normalize=False)."""
    import scipy.io.wavfile as wav

    wav_path = create_test_wav()
    _, expected = wav.read(str(wav_path))

    cf = ChannelFrame.read_wav(str(wav_path), normalize=False)
    np.testing.assert_array_equal(cf.data[0].astype(np.int16), expected)
```

### Normalization Modes
```python
def test_wav_read_raw_vs_normalized(tmp_path, create_test_wav):
    """Raw read preserves integer magnitudes; normalized read scales to [-1, 1]."""
    wav_path = create_test_wav()

    raw = ChannelFrame.read_wav(str(wav_path), normalize=False)
    normalized = ChannelFrame.read_wav(str(wav_path), normalize=True)

    # Raw values have integer-scale magnitudes
    assert np.max(np.abs(raw.data)) > 1.0
    # Normalized values are in [-1, 1]
    assert np.max(np.abs(normalized.data)) <= 1.0
```

### Stereo/Mono Channel Layout
```python
def test_wav_stereo_channel_count(create_test_wav):
    """Stereo WAV must produce 2-channel ChannelFrame."""
    wav_path = create_test_wav(n_channels=2)
    cf = ChannelFrame.read_wav(str(wav_path))
    assert cf.n_channels == 2

def test_wav_mono_channel_count(create_test_wav):
    """Mono WAV must produce 1-channel ChannelFrame."""
    wav_path = create_test_wav(n_channels=1)
    cf = ChannelFrame.read_wav(str(wav_path))
    assert cf.n_channels == 1
```

---

## WDF I/O Tests

### Complete Metadata Preservation
WDF は Wandas のネイティブフォーマットであり、全メタデータの保存を保証する。

```python
def test_wdf_preserves_channel_metadata(tmp_path):
    """WDF must preserve channel labels, units, and custom metadata."""
    cf = ChannelFrame.from_numpy(
        np.zeros((3, 100), dtype=np.float32),
        sampling_rate=16000,
        ch_labels=["mic1", "mic2", "mic3"],
        ch_units=["Pa", "Pa", "Pa"],
    )
    path = tmp_path / "test.wdf"
    cf.save(path)
    loaded = ChannelFrame.load(path)

    assert loaded.labels == ["mic1", "mic2", "mic3"]
    for ch in loaded._channel_metadata:
        assert ch.unit == "Pa"
```

### Dtype Conversion
```python
def test_wdf_dtype_conversion(tmp_path):
    """WDF save with dtype must convert data correctly."""
    data = np.random.default_rng(42).standard_normal((1, 100)).astype(np.float64)
    cf = ChannelFrame.from_numpy(data, sampling_rate=16000)

    path = tmp_path / "test.wdf"
    cf.save(path, dtype="float32")
    loaded = ChannelFrame.load(path)

    assert loaded.data.dtype == np.float32
    np.testing.assert_allclose(
        loaded.data, data.astype(np.float32),
        rtol=1e-6,
    )
```

### Compression Options
```python
def test_wdf_compression_roundtrip(tmp_path):
    """Compressed WDF must produce identical results after decompression."""
    data = np.random.default_rng(42).standard_normal((2, 1000)).astype(np.float32)
    cf = ChannelFrame.from_numpy(data, sampling_rate=16000)

    # Save with compression
    path_compressed = tmp_path / "compressed.wdf"
    cf.save(path_compressed, compress="gzip")

    # Save without compression
    path_raw = tmp_path / "raw.wdf"
    cf.save(path_raw, compress=None)

    loaded_compressed = ChannelFrame.load(path_compressed)
    loaded_raw = ChannelFrame.load(path_raw)

    np.testing.assert_array_equal(loaded_compressed.data, loaded_raw.data)
```

### Overwrite Protection
```python
def test_wdf_overwrite_protection(tmp_path):
    """WDF save must raise FileExistsError if file exists and overwrite=False."""
    cf = ChannelFrame.from_numpy(np.zeros((1, 100), dtype=np.float32), sampling_rate=16000)
    path = tmp_path / "test.wdf"
    cf.save(path)

    with pytest.raises(FileExistsError):
        cf.save(path, overwrite=False)
```

---

## CSV I/O Tests

```python
def test_csv_read_with_header(tmp_path):
    """CSV read must interpret column headers as channel labels."""
    csv_content = "time,sensor_a,sensor_b\n0.0,1.0,2.0\n0.001,1.1,2.1\n"
    path = tmp_path / "test.csv"
    path.write_text(csv_content)

    cf = ChannelFrame.read_csv(str(path), time_column=0)
    assert cf.n_channels == 2
    # Labels derived from CSV headers
    assert "sensor_a" in cf.labels[0] or "sensor_b" in cf.labels[1]
```

---

## File Error Handling Tests

```python
def test_read_nonexistent_file():
    """Reading non-existent file must raise FileNotFoundError with path info."""
    with pytest.raises(FileNotFoundError, match=r"Audio file not found"):
        ChannelFrame.from_file("/nonexistent/path/audio.wav")

def test_in_memory_without_file_type():
    """In-memory data without file_type must raise ValueError."""
    with pytest.raises(ValueError, match=r"File type is required"):
        ChannelFrame.from_file(b"fake data")
```

---

## Lazy Loading Verification

I/O テストでは Dask の遅延読み込みが正しく動作していることも検証する。

```python
def test_wav_lazy_loading(create_test_wav):
    """WAV read must return a Dask-backed frame without loading data immediately."""
    from dask.array.core import Array as DaArray

    wav_path = create_test_wav()
    cf = ChannelFrame.read_wav(str(wav_path))
    assert isinstance(cf._data, DaArray)
    # Data not yet in memory until .compute() or .data access
```

---

## Cross-References
- [test-grand-policy.instructions.md](test-grand-policy.instructions.md) — 4 pillars and test pyramid
- [io-contracts.instructions.md](io-contracts.instructions.md) — I/O metadata preservation contracts
