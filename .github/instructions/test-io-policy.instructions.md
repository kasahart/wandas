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

`conftest.py` に以下の fixture を定義すること。

- **`known_signal_frame`**: ラウンドトリップ検証用の 2 チャンネルフレーム（SR=44100 Hz、ch_labels=`["left", "right"]`、ch_units=`["Pa", "Pa"]`）。I/O テストでは「データの等価性」が検証の基準となるため、シード固定乱数（`np.random.default_rng(42)`）による信号が許容される。
- **`create_test_wav`**: `tmp_path` を受け取るファクトリ fixture。`sr`, `n_channels`, `n_samples` を引数に取り、`scipy.io.wavfile.write` で int16 PCM の WAV ファイルを作成して Path を返す。モノラル（n_channels=1）では 1 次元配列、ステレオ（n_channels=2）では 2 次元配列として書き込むこと。

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

WDF のラウンドトリップでは以下のすべてが保持されることを検証すること:
- 数値データ（`np.testing.assert_allclose` rtol=1e-6）
- サンプリングレート
- チャンネルラベル
- `operation_history`

---

## WAV I/O Tests

### Float Round-Trip
`ChannelFrame.to_wav` は浮動小数点データかつ `max(abs(data)) <= 1` の場合に IEEE FLOAT サブタイプで書き込むため、PCM 量子化誤差が発生しない。`normalize=True` で読み込み、元データと `atol=1e-6` 以内の一致を検証すること。

### PCM Round-Trip
`scipy.io.wavfile.write` で int16 として書き込んだ WAV ファイルは `normalize=False` で読み込み、元の整数サンプル値と一致することを検証すること（`cf.data[0].astype(np.int16)` で比較）。

### Channel Count
- ステレオ WAV（n_channels=2）を読み込んだ `ChannelFrame` の `n_channels` が 2 であることを検証すること。
- モノラル WAV（n_channels=1）を読み込んだ `ChannelFrame` の `n_channels` が 1 であることを検証すること。

---

## WDF I/O Tests

WDF は Wandas のネイティブフォーマットであり、全メタデータの保存を保証する。

- **チャンネルメタデータ**: ch_labels と ch_units が保存・復元されることを検証すること。
- **上書き保護**: `overwrite=False`（デフォルト）で既存ファイルに保存しようとすると `FileExistsError` が送出されることを検証すること。

---

## CSV I/O Tests

ヘッダー付き CSV（time 列 + データ列）を読み込み、チャンネル数が正しく取得されること（time 列を除いたデータ列数と一致すること）を検証すること。

---

## File Error Handling Tests

存在しないパスへの `ChannelFrame.from_file()` 呼び出しが `FileNotFoundError` を送出することを検証すること。

---

## Lazy Loading Verification

WAV 読み込み後の `cf._data` が `dask.array.core.Array` のインスタンスであることを検証し、データが即座にメモリに読み込まれていないことを確認すること。

---

## Cross-References
- [test-grand-policy.instructions.md](test-grand-policy.instructions.md) — 4 pillars and test pyramid
- [io-contracts.instructions.md](io-contracts.instructions.md) — I/O metadata preservation contracts
