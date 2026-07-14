# Extract a Spectral Envelope with the Real Cepstrum

Use the real cepstrum when you want to separate a spectrum's slowly varying
envelope from its fine harmonic structure. Wandas keeps each domain explicit:
`ChannelFrame` contains time samples, `CepstralFrame` contains real coefficients on
a quefrency axis, and `SpectralFrame` contains the reconstructed envelope.

実ケプストラムを使うと、スペクトルの緩やかな包絡と細かな調波構造を分離できます。
Wandasでは、時間波形を`ChannelFrame`、ケフレンシー軸上の実係数を`CepstralFrame`、
再構成した包絡を`SpectralFrame`として、各領域を明示的に扱います。

## Run the typed workflow / 型付きワークフローを実行する

```python
import wandas as wd

audio = wd.read("voice.wav")

cepstrum = audio.cepstrum(n_fft=2048, window="hann")
smooth_cepstrum = cepstrum.lifter(cutoff=0.002, mode="low")
envelope = smooth_cepstrum.to_spectral_envelope()

cepstrum.plot()
envelope.plot()
```

The three calls are lazy. They build a Dask graph while preserving channel
metadata, stable channel IDs, user metadata, sampling rate, and per-channel
`source_time_offset`. Plotting, accessing `.data`, or calling `.compute()` is an
explicit compute boundary.

3つの呼出しはすべて遅延実行です。チャンネルメタデータ、安定したチャンネルID、
ユーザーメタデータ、サンプリングレート、チャンネルごとの`source_time_offset`を維持したまま、
Daskグラフを構築します。描画、`.data`へのアクセス、`.compute()`の呼出しが明示的な計算境界です。

## Understand each step / 各ステップの意味

| Step / 手順 | Result / 結果 | Definition / 定義 |
| --- | --- | --- |
| `audio.cepstrum(...)` | `CepstralFrame` | `irfft(log(max(normalized_fft_magnitude, floor)))` |
| `.lifter(cutoff, mode="low")` | `CepstralFrame` | Keeps low quefrencies and their circular mirror / 低ケフレンシーと循環対称位置を保持 |
| `.to_spectral_envelope()` | `SpectralFrame` | `exp(real(rfft(liftered_cepstrum)))` |

The cepstrum has dimensions `(channel, quefrency)`. Quefrency is measured in
seconds with spacing `1 / sampling_rate`. The reconstructed `SpectralFrame` has
`n_fft // 2 + 1` frequency bins and zero phase. It is a magnitude envelope, not an
inverse reconstruction of the original waveform.

ケプストラムの次元は`(channel, quefrency)`です。ケフレンシーの単位は秒で、間隔は
`1 / sampling_rate`です。再構成した`SpectralFrame`は`n_fft // 2 + 1`個の周波数ビンと
ゼロ位相を持ちます。これは振幅包絡であり、元波形の逆変換結果ではありません。

Without liftering, the reconstructed envelope matches the magnitude returned by
`audio.fft(n_fft=..., window=...)`, apart from floating-point round-off. Wandas
normalizes before taking the logarithm, so no separate analysis-length or window-gain
state is needed during reconstruction.

リフタリングを行わない場合、再構成結果は浮動小数点誤差を除き
`audio.fft(n_fft=..., window=...)`の振幅と一致します。対数化の前に正規化するため、
再構成時に解析長や窓利得を別状態として保持する必要はありません。

## Choose the lifter / リフタを選ぶ

- Use `mode="low"` to keep the slowly varying spectral envelope. A cutoff around
  1–3 ms is a practical starting point for speech, but it is signal-dependent.
- Use `mode="high"` to keep the complementary fine structure, including periodic
  pitch information.
- The cutoff is in seconds. It must reach at least one quefrency bin and remain below
  half of the complete cepstrum so the positive and mirrored regions do not overlap.

- 緩やかなスペクトル包絡を残す場合は`mode="low"`を使います。音声では1–3 ms程度が
  出発点になりますが、適切な値は信号に依存します。
- 周期的なピッチ情報を含む細かな構造を残す場合は`mode="high"`を使います。
- カットオフの単位は秒です。少なくとも1ケフレンシービンに達し、正負の鏡像領域が
  重ならないよう、完全なケプストラムの半分未満にする必要があります。

## Set FFT size and floor / FFTサイズとfloorを設定する

`n_fft=None` uses the current sample count. A smaller value truncates the current
signal; a larger value zero-pads it. `floor` must be positive and finite and prevents
`log(0)`. The default `1e-12` is suitable for ordinary normalized audio analysis.

`n_fft=None`では現在のサンプル数を使います。小さい値は信号を切り詰め、大きい値は
ゼロパディングします。`floor`は正の有限値で、`log(0)`を防ぎます。通常の正規化音声解析では
既定値`1e-12`を利用できます。

Liftering and envelope reconstruction require the complete circular quefrency axis.
You may slice a `CepstralFrame` for inspection and plotting, and its coordinate values
remain accurate, but perform `lifter()` and `to_spectral_envelope()` before slicing.

リフタリングと包絡再構成には、完全な循環ケフレンシー軸が必要です。確認や描画のために
`CepstralFrame`をスライスでき、座標値も維持されますが、`lifter()`と
`to_spectral_envelope()`はスライス前に実行してください。

## Replay the workflow as a Recipe / Recipeとして再実行する

The complete typed workflow is portable through Recipe v2:
型付きワークフロー全体をRecipe v2で再利用できます。

```python
from wandas.pipeline import RecipePlan

processed = (
    audio.cepstrum(n_fft=2048, window="hann")
    .lifter(cutoff=0.002)
    .to_spectral_envelope()
)
plan = RecipePlan.from_frame(processed, input_names=("signal",))

payload = plan.to_dict()
loaded = RecipePlan.from_dict(payload)
replayed = loaded.apply({"signal": another_audio})
```

Extraction, serialization, loading, and graph construction do not compute Dask
arrays. When `n_fft` is omitted, the Recipe preserves that omission, so replay uses
the new input's sample count rather than freezing the original length.

Recipeの抽出、シリアライズ、読込、グラフ構築ではDask配列を計算しません。`n_fft`を省略した
場合、Recipeは省略した意図を維持するため、再実行時には元の長さを固定せず、新しい入力の
サンプル数を使います。
