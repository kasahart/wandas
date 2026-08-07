# Tutorial / チュートリアル

In five minutes, we will build one mono signal, put it in a Wandas
`ChannelFrame`, filter it, and compare the result in the time and frequency
domains. The example uses one NumPy array because `wd.generate_sin(freqs=[...])`
creates one channel per frequency; that is useful in other workflows, but not
for this mixed-signal filter example.

この5分チュートリアルでは、モノラル信号を1つ作り、Wandasの`ChannelFrame`に渡して、
時間領域と周波数領域で処理前後を比較します。ここでは、周波数ごとに別チャンネルを
作る`wd.generate_sin(freqs=[...])`ではなく、NumPyで同じ配列に2つの成分を合成します。

## Installation / インストール

```bash
pip install wandas
```

The optional `io`, `marimo`, and analysis extras are documented with the
workflow that needs them.

必要なworkflowに応じた`io`、`marimo`、解析extraは各ドキュメントで説明しています。

## 1. Create one mono signal / 1つのモノラル信号を作る

The two sine waves share one time axis and are added to one NumPy array. The
result is one mono signal containing 440 Hz and 1,800 Hz components.

2つの正弦波を同じ時間軸上で加算し、1つのNumPy配列にします。そのため、結果は440 Hzと
1,800 Hzの成分を持つ1つのモノラル信号です。

```python exec="on" session="wd_tutorial" source="above"
import numpy as np
import wandas as wd
```

```python exec="on" session="wd_tutorial" source="above"
sampling_rate = 16_000
duration = 1.0
time = np.arange(int(sampling_rate * duration), dtype=np.float64) / sampling_rate

mixed_signal = (
    np.sin(2 * np.pi * 440 * time)
    + 0.6 * np.sin(2 * np.pi * 1_800 * time)
)
signal = wd.from_numpy(mixed_signal, sampling_rate=sampling_rate).rename_channels({0: "Original"})

print(f"Frame / 型: {type(signal).__name__}")
print(f"Channels / チャンネル数: {signal.n_channels}")
print(f"Labels / ラベル: {signal.labels}")
```

`wd.from_numpy()` turns the one-dimensional array into a mono `ChannelFrame`.
Renaming the channel makes the comparison legend readable later.

`wd.from_numpy()`は1次元配列をモノラルの`ChannelFrame`にします。チャンネル名を変更しておくと、
後の比較図の凡例も読みやすくなります。

## 2. Filter and combine the Frames / Frameを処理して比較用にまとめる

The cutoff lies between the two components. `low_pass_filter()` returns a new
Frame, and `concat_frame()` keeps the original and filtered channels together
for a direct comparison.

カットオフ周波数を2つの成分の間に置きます。`low_pass_filter()`は新しいFrameを返し、
`concat_frame()`で元信号と処理後信号を比較用のFrameにまとめます。

```python exec="on" session="wd_tutorial" source="above"
processed = (
    signal
    .low_pass_filter(cutoff=1_000)
    .rename_channels({0: "After 1 kHz low-pass"})
)

comparison = signal.concat_frame(processed)
print(f"Comparison labels / 比較ラベル: {comparison.labels}")
```

The original `signal` remains available, while `processed` and `comparison`
are new Frames. The next two examples use the same `comparison` object, so the
legend identifies which curve is original and which is filtered.

元の`signal`はそのまま残り、`processed`と`comparison`は新しいFrameです。次の2つの例では
同じ`comparison`を使うため、凡例から元信号とフィルタ後信号を識別できます。

## 3. Compare the time waveform / 時間波形を比較する

`ChannelFrame.plot()` uses the Frame data and channel labels to draw both
waveforms. The SVG conversion is kept in a hidden document-only setup block;
the reader-facing code is only the Wandas plotting call.

`ChannelFrame.plot()`はFrameのデータとチャンネル名を使って2つの波形を描きます。SVG変換は
非表示のドキュメント用セットアップに分離し、読者向けコードはWandasの描画呼び出しだけにします。

```python exec="on" session="wd_tutorial"
from html import escape
import io
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def inline_svg(axes, caption):
    figure = axes.figure
    buffer = io.StringIO()
    try:
        figure.savefig(
            buffer,
            format="svg",
            bbox_inches="tight",
            metadata={"Date": None},
        )
        svg = buffer.getvalue()
        svg_start = svg.find("<svg")
        if svg_start < 0:
            raise ValueError("Matplotlib did not produce an SVG document")
        return f'<figure class="tutorial-figure"><figcaption>{escape(caption)}</figcaption>{svg[svg_start:]}</figure>'
    finally:
        plt.close(figure)
```

```python exec="on" session="wd_tutorial" source="above" html="on"
waveform_ax = comparison.plot(
    overlay=True,
    xlim=(0, 0.02),
    title="Original vs filtered",
)
print(inline_svg(waveform_ax, "Time waveform: original and filtered"))
```

The high-frequency ripple is reduced in the filtered curve, while the slower
440 Hz shape remains recognizable.

フィルタ後の曲線では高周波の細かな揺らぎが小さくなり、低い440 Hzの形はおおむね残ります。

## 4. Compare the frequency spectrum / 周波数スペクトルを比較する

`fft()` creates a Wandas `SpectralFrame`, and its `plot()` method displays the
frequency components. No signal samples or spectrum values need to be passed
to Matplotlib directly.

`fft()`はWandasの`SpectralFrame`を作り、その`plot()`で周波数成分を表示します。信号サンプルや
スペクトル値をMatplotlibへ直接渡す必要はありません。

```python exec="on" session="wd_tutorial" source="above" html="on"
spectrum_ax = comparison.fft().plot(
    overlay=True,
    xlim=(0, 2_500),
    title="FFT: original vs filtered",
)
print(inline_svg(spectrum_ax, "FFT spectrum: original and filtered"))
```

The original spectrum has clear peaks at 440 Hz and 1,800 Hz. After the
1,000 Hz low-pass filter, the 440 Hz peak remains close to its original level,
while the 1,800 Hz peak is much smaller.

元信号のスペクトルには440 Hzと1,800 Hzのピークがあります。1,000 Hzローパスフィルタ後は、
440 Hzのピークが元に近い大きさで残り、1,800 Hzのピークは大きく小さくなります。

## 5. Keep the workflow chainable / 処理をメソッドチェーンで書く

The complete processing and analysis path stays composable: create a Frame,
apply an operation, combine it for comparison, then run FFT and plotting.

Frameの作成、処理、比較、FFT、可視化までを自然なメソッドチェーンとして組み合わせられます。

```python exec="on" session="wd_tutorial" source="above"
chained = (
    signal
    .low_pass_filter(cutoff=1_000)
    .rename_channels({0: "After 1 kHz low-pass"})
)
chained_comparison = signal.concat_frame(chained)
chained_spectrum = chained_comparison.fft()
```

The same `chained_comparison` can be passed to `plot()` or `fft().plot()` when
you want another view, while the original Frame remains available for reuse.

同じ`chained_comparison`を`plot()`や`fft().plot()`へ渡して別の視点で確認できます。元のFrameも
再利用のために残ります。

## Next step / 次のステップ

Continue with the executable
<a href="../en/learning-path/00_why_wandas.html">English Learning Path</a>.
Start at 00 and continue in manifest order toward real data and signal-processing basics.

次は、実行可能な
<a href="../learning-path/00_why_wandas.html">日本語 Learning Path</a>
を00から順番に進めてください。実データの読み込み、基本的な信号処理へ進みます。

For a specific task or API contract, use the
[How-to guide](../how-to/cepstral-analysis.md) or
[API Reference](../api/index.md).

具体的な作業手順は[How-to guide](../how-to/cepstral-analysis.md)、
APIの契約は[API Reference](../api/index.md)を参照してください。
