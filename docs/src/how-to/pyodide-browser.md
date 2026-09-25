# Run Wandas in a Pyodide browser / PyodideブラウザでWandasを使う

Wandas can run core signal processing and WAV workflows in a browser with
Pyodide. This guide shows how to install Wandas and read WAV data in that
runtime.

WandasはPyodideを使って、ブラウザ内で基本的な信号処理とWAV処理を実行できます。
このガイドでは、Pyodide環境へのインストールとWAVデータの読み込み方法を説明します。

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
