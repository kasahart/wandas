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

**Suffix なし（`ChannelFrame` 返却）の fixture 定義例:**

```python
import pytest
import numpy as np
from wandas.frames.channel import ChannelFrame

@pytest.fixture
def pure_sine_1khz() -> ChannelFrame:
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = np.sin(2 * np.pi * 1000 * t)
    return ChannelFrame.from_numpy(data.reshape(1, -1), sampling_rate=sr)
```

**`_dask` 付き（`(DaskArray, int)` タプル返却）の fixture 定義例:**

```python
import pytest
import numpy as np
from wandas.utils.dask_helpers import da_from_array

@pytest.fixture
def pure_sine_1khz_dask() -> tuple:
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = np.sin(2 * np.pi * 1000 * t)
    dask_array = da_from_array(data.reshape(1, -1), chunks=(1, -1))
    return (dask_array, sr)
```

### Standard Fixtures

以下の標準 fixture を `conftest.py` に定義すること。信号はすべて解析解が既知の決定論的信号を使用する。

- **純音 fixture**: FFT ピーク位置が解析的に予測可能な単一周波数信号。
- **複合音 fixture**: 複数の周波数成分を持つ合成信号。フィルタの通過・遮断特性の検証に使用。
- **校正済み純音 fixture**: IEC 規格に準拠した既知の音圧レベルを持つ信号。心理音響テスト標準信号として使用。
- **インパルス fixture**: 単位インパルス信号。フィルタのインパルス応答検証に使用。

Processing 層の fixture（`_dask` suffix）は `(DaskArray, int)` のタプルで返すこと。

---

## Pillar 1: Data Immutability & Lazy Evaluation Integrity

### Rule: Side-Effect-Free Operations
いかなる演算を行っても、元のインスタンスの内部データやメタデータが変更されてはならない。各演算テストでは、演算前に元のデータをコピーし、演算後に元のインスタンスが変更されていないこと、かつ返り値が新しいインスタンスであることを検証すること。

### Rule: Dask Graph Protection
Dask による遅延評価を維持するため、演算後の `result._data` が `DaskArray` インスタンスであることを確認すること。演算が新たな `.compute()` を意図せず呼び出していないかを検証する。

### Checklist
- [ ] Operation returns a new instance (`assert result is not original`)
- [ ] Original data unchanged after operation (`assert_array_equal`)
- [ ] Original metadata unchanged (sampling_rate, labels, operation_history)
- [ ] Dask graph preserved (no premature `.compute()`)
- [ ] `result._data` is a `DaskArray` instance after operation

---

## Pillar 2: Physical Domain Metadata Synchronization

### Rule: Metadata Auto-Tracking
演算後もサンプリングレート・チャンネル数・ラベルが正確に継承または変換されていること。リサンプリング以外の演算ではサンプリングレートが保持されること、チャンネル数が保持されることを検証すること。

### Rule: Operation History Traceability
`operation_history` に処理内容とパラメータが記録されていること。履歴を記録する設計のメソッドでは `operation_history` の件数がちょうど 1 件増加していることを検証し、最後のエントリに正しい **レジストリキー**（例: `"lowpass_filter"`）とパラメータが含まれていることを検証すること。チャンネル集合を変更するだけの構造的操作（`add_channel` / `remove_channel` / `rename_channels` など）では `operation_history` が変化していないことを検証すること。

注意: `operation_history` に記録されるのはメソッド名ではなく **レジストリキー** であり、フレームメソッド名（例: `low_pass_filter`）とは異なる場合がある。

### Rule: Domain Transition Metadata
ドメインが変わる処理（時間領域 → 周波数領域 → 時間-周波数領域）では、軸の次元数がドメイン変換の理論値と一致することを検証すること。例えば FFT 後の周波数ビン数は元の時間サンプル数の N/2+1 になる。

### Checklist
- [ ] Sampling rate preserved (or correctly updated for resampling)
- [ ] Channel labels propagated with operation annotation
- [ ] `operation_history` grows by exactly 1 entry per processing operation (registry-based); structural channel operations (add_channel / remove_channel / rename_channels) leave history unchanged
- [ ] History entry contains correct operation name (registry key) and params
- [ ] Domain transition produces correct axis dimensions

---

## Pillar 3: Mathematical Consistency & Transform Reversibility

### Rule: Domain Round-Trip Fidelity
STFT→ISTFT 等の可逆変換では、逆変換によって元の信号が `atol=1e-6` 以内で復元できることを検証すること。

### Rule: Shape & Type Consistency
出力配列の shape が理論値と一致することを検証すること。例えば STFT の周波数ビン数は `n_fft // 2 + 1` になる。

### Rule: Sampling Rate Constraint Enforcement
異なるサンプリングレートをもつフレーム間の不正な演算（加算・比較等）は `ValueError` を送出すること。

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
Wandas が SciPy 等のラッパーである場合、外部ライブラリを直接呼び出した結果と比較すること。実装と同じアルゴリズム・パラメータで比較することで、完全一致（`assert_allclose` デフォルト rtol=1e-7）を要求できる。

例: `LowPassFilter` は `scipy.signal.butter(order, cutoff/nyquist, btype="low")` で係数を生成し `scipy.signal.filtfilt(b, a, x, axis=1)` で適用するため、同じ呼び出しと比較すれば完全一致する。

### Rule: Theoretical Value Verification
自前でアルゴリズムを実装する場合は、解析的に予測可能な信号（純音・インパルス・DC）を使用して理論値と比較すること。例えば 1 kHz 純音の FFT では、ピークが 1 kHz のビン（±1 ビン以内）に現れることを検証する。

### Rule: Tolerance Standards

| Category | Default Tolerance | Rationale |
|----------|------------------|-----------|
| Wrapper equivalence | 完全一致（`assert_array_equal` または `assert_allclose` デフォルト） | Same algorithm, exact numeric result |
| Round-trip transforms | `atol=1e-6` | Windowing/overlap introduces small errors |
| Psychoacoustic metrics (MoSQITo wrapper) | 完全一致 | Wandas wraps MoSQITo directly — results must be identical |
| Theoretical value verification | `rtol=1e-6` | Known analytical result, numerical precision |
| Frequency peak detection | Within 1 FFT bin | Spectral leakage at bin boundaries |

許容誤差はコメントで根拠を明示すること（`# Float32 precision tolerance` 等）。

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
- **例**: ゼロや Nyquist を超えるカットオフ周波数が `ValueError` を返す、3 次元配列の入力が拒否される

### Layer 2: Domain Tests (Physics) — Middle
- **対象**: 物理モデル、JIS/IEC 規格準拠のアルゴリズム
- **目的**: 時定数（Fast/Slow）の応答特性、フィルタの遮断特性など、音響工学としての物理的妥当性の保証
- **例**: ローパスフィルタがカットオフ周波数以上の成分を十分に減衰させる、A 特性重み付けが 1 kHz で 0 dB に近い

### Layer 3: Integration Tests (Wrapper & Authority) — Top
- **対象**: 外部ライブラリとの連携、エンドツーエンドの計算パイプライン
- **目的**: 既存ライブラリや理論式という「外部の権威」との比較による、実装の正当性の裏取り
- **例**: `cf.low_pass_filter()` の結果が `scipy.signal.filtfilt()` を同パラメータで呼び出した結果と一致する

### Layer Distribution Guideline
新しい処理を追加する場合、最低限以下のテスト構成を確保すること:
- **Unit**: 入力バリデーション + エラーケース 2-3 件
- **Domain**: 物理的妥当性のスモークテスト 1-2 件（既知信号を使用）
- **Integration**: ラッパーなら等価性テスト 1 件、独自実装なら理論値テスト 1 件

---

## Test Conventions

### Naming
テスト関数名は `test_{what}_{condition}_{expected_outcome}` のパターンに従うこと。

例:
- `test_low_pass_filter_above_nyquist_raises_error`
- `test_fft_pure_sine_peak_at_correct_frequency`

### Error Message Assertions
`pytest.raises(ExceptionType, match=r"...")` でエラーメッセージを検証する際は、最初の行のみにマッチするパターンを使用すること。メッセージの詳細部分（HOW セクション等）は変更されやすいため、コアとなるエラー種別のみを検証する。

### Dask Array Creation
Processing 層のテストで Dask 配列を作成する際は `wandas.utils.dask_helpers.da_from_array` を使用し、`chunks=(1, -1)` の channel-wise chunks を指定すること。

---

## Anti-Patterns (Avoid These)

以下のパターンはテストの信頼性を損なうため避けること:

- **浮動小数点の直接比較**: 丸め誤差により常に失敗する。`np.testing.assert_allclose` を使い、許容誤差と根拠を明示すること。
- **乱数データの使用**: 再現性がなく、解析的な正しさを検証できない。決定論的な既知信号（純音・インパルス・DC）を使用すること。
- **「エラーなし」のみの検証**: `result is not None` のような検証は何も保証しない。実際の出力値・型・shape を検証すること。
- **根拠なき魔法数**: `atol=0.0001` のような説明なき許容誤差はレビュー不能。なぜその値を使うかをコメントで示すこと。

---

## Cross-References
- [frames-design.instructions.md](frames-design.instructions.md) — Frame immutability and metadata rules
- [processing-api.instructions.md](processing-api.instructions.md) — Processing layer responsibilities
- [io-contracts.instructions.md](io-contracts.instructions.md) — I/O metadata preservation contracts
- [testing-workflow.instructions.md](testing-workflow.instructions.md) — TDD workflow and quality commands
