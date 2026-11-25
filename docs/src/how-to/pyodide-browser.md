# Run Wandas in a Pyodide browser / PyodideブラウザでWandasを使う

Wandas can run core signal processing and WAV workflows in a browser with
Pyodide. This guide shows how to install Wandas and read WAV data in that
runtime. To see it run first, open the
[live demo](https://kasahart.github.io/wandas/demo/): an editor where you
change and run Python cells that read, plot, and filter a WAV file, including
one you choose, all inside the page.

WandasはPyodideを使って、ブラウザ内で基本的な信号処理とWAV処理を実行できます。
このガイドでは、Pyodide環境へのインストールとWAVデータの読み込み方法を説明します。
まず動くところを見る場合は、[ライブデモ](https://kasahart.github.io/wandas/demo/)を開いてください。
WAVの読み込み、描画、フィルタ処理をするPythonのセルを書き換えて実行できます。選んだWAVファイルもページ内だけで処理します。

## What works in Pyodide / Pyodideで使える機能

Checked with Pyodide 314.0.3, the version the
[browser example](https://github.com/kasahart/wandas/blob/main/examples/pyodide/index.html)
loads. The release workflow checks each published Wandas version in it.

[ブラウザ例](https://github.com/kasahart/wandas/blob/main/examples/pyodide/index.html)が読み込むPyodide 314.0.3で確認しています。
リリースのたびに、公開したWandasをこの版で確認します。

| Feature / 機能 | Install / インストール | In Pyodide / Pyodideでは | Checked by / 確認方法 |
| --- | --- | --- | --- |
| Core: Frames, filters, FFT, STFT, Welch, weighting, levels, Recipes / 基本機能 | `wandas` | Works / 使える | Core tests on each pull request that changes `wandas/` or the build / `wandas/`やビルド設定を変えるpull requestごとにcore tests |
| WAV read and write / WAVの読み書き | `wandas` | Works, from bytes or the virtual file system / bytesか仮想ファイルから使える | WAV smoke tests on the same pull requests / 同じpull requestでWAV smoke |
| Matplotlib plots, `describe()` / 図 | `wandas` | Works / 使える | Browser-guide check / ブラウザガイドの検証 |
| `loudness_zwst()`, `sharpness_din_st()`, `roughness_dw()`, `noct_spectrum()` / 心理音響指標の一部と`noct_spectrum()` | `wandas[psychoacoustic]` (MoSQITo) | Works / 使える | Nightly extras check / 毎晩の追加機能の検証 |
| WDF save and load / WDFの保存と読込 | `wandas[io]` (h5netcdf, h5py) | Works / 使える | Nightly extras check / 毎晩の追加機能の検証 |
| HPSS (`hpss_harmonic()`, `hpss_percussive()`) | `wandas[effects]` (librosa) | Not available: librosa needs numba, which has no Pyodide build. The call raises `ImportError` naming the extra / 使えない。librosaが必要とするnumbaにPyodide版がない。呼ぶと追加機能名を示す`ImportError`になる | Nightly extras check / 毎晩の追加機能の検証 |
| `to_tensor()` for PyTorch / TensorFlow / テンソル変換 | `wandas[ml]` | Not available: no Pyodide builds / 使えない。Pyodide版がない | Skipped in Pyodide core tests / core testsで除外 |
| Learning path lessons / 学習パス | — | Published as pre-run pages, not as in-browser notebooks. To run code in the browser, use the [live demo](https://kasahart.github.io/wandas/demo/) / 実行済みのページとして公開し、ブラウザ内ノートブックにはしない。ブラウザで実行するには[ライブデモ](https://kasahart.github.io/wandas/demo/)を使う | Not applicable / 対象外 |

An optional feature whose dependency is missing raises `ImportError` with the
`pip install "wandas[...]"` line to use. In Pyodide, install the same extra
with `await micropip.install("wandas[psychoacoustic]")`.

依存パッケージがない追加機能は、使うべき`pip install "wandas[...]"`を示す`ImportError`になります。
Pyodideでは`await micropip.install("wandas[psychoacoustic]")`のように同じ追加機能をインストールします。

## Install and process / installして処理する

Install Wandas in the current Pyodide runtime, generate a signal, and
process it. For a complete HTML page with compatible Pyodide and Wandas
versions, see the [browser example](https://github.com/kasahart/wandas/blob/main/examples/pyodide/index.html).

現在のPyodide環境にWandasをインストールし、信号を生成して処理します。
互換性のあるPyodideとWandasのバージョンを設定したHTMLページ全体は、
[ブラウザ例](https://github.com/kasahart/wandas/blob/main/examples/pyodide/index.html)を参照してください。

```python
import micropip

await micropip.install("wandas")

import numpy as np
import wandas as wd

sampling_rate = 8_000
time = np.arange(sampling_rate, dtype=np.float64) / sampling_rate
source = wd.from_numpy(np.sin(2 * np.pi * 440 * time), sampling_rate=sampling_rate)
filtered = source.low_pass_filter(cutoff=1_000)
```

## Read WAV bytes / WAV bytesを読む

Browser file inputs and fetch responses must be converted to bytes before
calling `wd.read()`:

```python
import wandas as wd

def read_selected_wav(wav_bytes: bytes):
    return wd.read(wav_bytes, file_type=".wav", source_name="selected.wav")

frame = read_selected_wav(wav_bytes)
```

For an external URL, fetch first and then decode the returned bytes:

```python
import wandas as wd
from pyodide.http import pyfetch

response = await pyfetch("https://example.com/recording.wav")
response.raise_for_status()
frame = wd.read(await response.bytes(), source_name="https://example.com/recording.wav")
```

ブラウザの`fetch`はCORS policyに従います。配信元が適切な
`Access-Control-Allow-Origin`を返さない場合、Wandasでは回避できません。
`wd.read(URL)`ではなく、`fetch → bytes → wd.read(...)`を使ってください。

DOM behavior, CORS headers, and audio autoplay depend on your browser and
origin. Check these in a real browser when deploying your application.

DOMの動作、CORSヘッダー、音声の自動再生はブラウザと配信元に依存します。
アプリを公開するときは実際のブラウザでも確認してください。
