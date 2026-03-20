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

`conftest.py` に以下の fixture を定義すること。信号は乱数ではなく決定論的な信号を使用すること。

- **`channel_frame`**: 標準的な複数チャンネルフレーム。解析解が既知の決定論的信号を使用する。
- **`mono_frame`**: 単チャンネルフレーム。解析解が既知の決定論的信号を使用する。

不変性・構造テストで信号内容が関係ない場合は、シード固定乱数も許容されるが、決定論的信号を優先すること。

---

## Frame Types & Their Test Concerns

### ChannelFrame (時間領域)
- **不変性**: 全演算で返り値が元のインスタンスと異なること、元データが変化していないことを検証する
- **チャンネル操作**: `add_channel`, `remove_channel`, `rename_channels` 後のラベルと data の整合性を検証する
- **スライシング**: int / slice / bool mask / label-based インデックスで正しいチャンネルが返ることを検証する
- **Dask laziness**: 演算後の `_data` が `DaskArray` インスタンスであることを検証する

### SpectralFrame (周波数領域)
- **Complex data 型**: `.data` が複素数配列であることを検証する
- **Derived properties**: `magnitude`, `phase`, `power`, `dB`, `dBA` が数学的に正しいことを検証する
- **周波数軸**: `frequency` 配列が Nyquist 周波数まで正しく構成されていることを検証する

### SpectrogramFrame (時間-周波数領域)
- **3 次元性**: shape が `(n_channels, n_freq_bins, n_time_frames)` であることを検証する
- **ISTFT round-trip**: 逆変換で元の時間領域信号が `atol=1e-6` 以内で復元できることを検証する
- **時間軸・周波数軸**: 両軸のメタデータが正確であることを検証する

### RoughnessFrame, NOctFrame (特殊解析)
- **MoSQITo 等価性**: Wandas は MoSQITo をラップしているため、外部ライブラリとの結果が完全に一致することを検証する
- **物理量メタデータ**: 単位（sone, asper 等）が正しく設定されていることを検証する

---

## Required Test Categories per Frame Operation

新しい Frame メソッドを追加する場合、以下の 4 カテゴリを必ず含めること:

**1. Immutability** — 元のフレームが変更されないこと:
演算前に元データをコピーし、演算後に元インスタンスのデータが変化していないこと、かつ返り値が新しいインスタンスであることを `assert result is not channel_frame` で検証すること。

**2. Metadata Propagation** — サンプリングレートと history が正しく更新されること:
演算後のサンプリングレートが保持されていること、`operation_history` の件数が 1 件増加していることを検証すること。

**3. Lazy Evaluation** — Dask 配列が保持されること:
演算後の `result._data` が `dask.array.core.Array` のインスタンスであることを検証すること。

**4. Chaining** — チェーン呼び出しに対応していること:
演算結果に続けて `.normalize()` 等を呼び出せること、返り値の型が `ChannelFrame` であることを検証すること。

---

## Domain Transition Test Patterns

ドメインが変わる処理（`fft()`, `stft()`, `loudness()` 等）では追加の検証が必要:

- 返り値の型が正しいフレーム型（`SpectralFrame` 等）に変換されていることを検証する
- 周波数ビン数が理論値（FFT の場合 N/2+1）と一致することを検証する
- サンプリングレートが正しく引き継がれていることを検証する

---

## Channel Collection Test Patterns

マルチチャンネル操作のテストで特に注意すべき点:

- `add_channel` 後のラベルと data の順序が一致すること
- 重複ラベルは `ValueError` になること
- 長さが違うデータを追加した場合は `ValueError` になること

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
