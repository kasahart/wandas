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
Frame を経由しない純粋な数値検証に使用する。`wandas.utils.dask_helpers.da_from_array` で `chunks=(1, -1)` の channel-wise chunks を作成すること。

以下の標準 fixture を `conftest.py` に定義すること:

- **`sine_1khz_48k_dask`**: 1 kHz 純音（SR=48000 Hz、1秒）。心理音響テストの標準信号。
- **`calibrated_sine_1khz_70dB_dask`**: 70 dB SPL の 1 kHz 純音（SR=48000 Hz）。IEC 規格準拠の心理音響テスト用。振幅は `p_ref=2e-5 Pa` から `amplitude = p_ref * 10**(70/20) * sqrt(2)` で計算する。
- **`dual_tone_16k_dask`**: 50 Hz + 1000 Hz の合成音（SR=16000 Hz）。フィルタの通過・遮断特性の検証に使用。
- **`impulse_16k_dask`**: 単位インパルス（SR=16000 Hz）。フィルタのインパルス応答検証に使用。

---

## Processing Module Test Strategy

### AudioOperation 基底クラス
以下の動作を検証すること:
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
フィルタ適用後の FFT スペクトルを取得し、各周波数成分の振幅比が設計上の理論値と相対誤差 1E-6 以内で一致することを検証する。

また、Wrapper Equivalence として `scipy.signal.filtfilt(b, a, x)` と比較すること。`LowPassFilter` の実装は `scipy.signal.butter(order, cutoff/nyquist, btype="low")` で係数を生成し `scipy.signal.filtfilt(b, a, x, axis=1)` で適用するため、同じ呼び出しと完全一致する。

### Filter Edge Cases
- カットオフ周波数が 0 以下、または Nyquist 周波数（`sampling_rate / 2`）以上の場合に `ValueError` が送出されることを検証すること。

---

## Spectral Tests: Known-Signal Verification

FFT/STFT テストでは、解析的に予測可能な信号を使用すること。

注意: クラス名は `FFT` および `STFT`（`FFTOperation` / `STFTOperation` ではない）。

- **FFT ピーク検証**: `n_fft=sr`（1 Hz/bin）で 1 kHz 純音の FFT を計算し、ピークビンが 1000 ±1 の範囲内にあることを検証する。
- **STFT shape 検証**: `n_fft=1024` の STFT を計算し、周波数軸のビン数が `1024 // 2 + 1 = 513` であることを検証する。

---

## Psychoacoustic Tests: MoSQITo Reference Verification

Wandas は MoSQITo をラップしているため、心理音響演算の結果は MoSQITo の参照実装と **完全に一致** しなければならない。
IEC 規格に準拠した校正済み信号（既知の音圧レベル、1秒以上）を使い、Wandas の演算結果と MoSQITo の `loudness_zwtv` 等を直接呼び出した結果が完全一致することを検証する。

---

## A-Weighting Tests: Known Frequency Response

A 特性フィルタは 1 kHz で 0 dB と定義されている。テストでは「理論値」を周波数応答として定義し、以下のいずれかの手順で検証すること:

- `A_weighting(fs, output="sos")` から得た係数に対して `scipy.signal.sosfreqz`/`freqz` 等で周波数応答を計算し、そのゲインと理論 A 特性曲線が十分小さい誤差（例: 相対誤差 1e-6 程度）で一致することを確認する。
- 既知の周波数の純音を入力信号として用いる場合は、`sosfilt` が因果フィルタであることに注意し、立ち上がり過渡を含む区間を評価から除外したうえで入出力の RMS 比を dB 換算し、理論周波数応答と十分小さい誤差で一致することを確認する（評価区間の取り方と許容誤差をテスト内で明示すること）。
---

## Operation Registration & Display Name Tests

- `create_operation` のキーは **レジストリ名**（例: `"lowpass_filter"`）であり、フレームメソッド名（`"low_pass_filter"`）とは異なる。`create_operation("lowpass_filter", ...)` が `None` 以外を返すことを検証すること。
- `LowPassFilter.get_display_name()` は `"lpf"` を返す（パラメータは含まない）。

---

## Anti-Patterns Specific to Processing Tests

以下のパターンは Processing テストの価値を損なうため避けること:

- **時間領域のみの検証**: フィルタ結果を時間領域で `assert result is not None` のみ検証しても、フィルタの品質は何も保証しない。必ず周波数領域で減衰量を検証すること。
- **自己参照的な期待値**: 自前で実装した関数で期待値を計算するのは「自己採点」に等しく、バグを検出できない。SciPy・librosa・MoSQITo 等の外部ライブラリを「権威」として使用すること。
- **心理音響テストの MoSQITo 比較省略**: `result.mean() > 0` のような検証は正しさを保証しない。必ず MoSQITo の参照実装と数値比較すること。

---

## Cross-References
- [test-grand-policy.instructions.md](test-grand-policy.instructions.md) — 4 pillars and test pyramid
- [processing-api.instructions.md](processing-api.instructions.md) — Processing layer architecture
