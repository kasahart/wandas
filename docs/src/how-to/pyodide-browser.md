# Run Wandas in a Pyodide browser / PyodideブラウザでWandasを使う

Wandas can run signal-processing and WAV workflows entirely in a browser with
Pyodide. This guide uses the combination validated by the Wandas Pyodide test
harness:

- Pyodide: **314.0.3**
- Wandas: **0.6.1**

Both versions are pinned in the examples. Test a new combination before
upgrading either version. Pyodide 314.0.3 was the latest stable release when
this contract was validated on 2026-07-28; its Python 3.14.2 runtime includes
the compiled scientific packages and soundfile 0.12.1 needed by Wandas.
Pyodide is downloaded from its versioned jsDelivr distribution; Wandas is
installed from its pure-Python wheel on PyPI. See the
[Pyodide 314.0.3 release](https://github.com/pyodide/pyodide/releases/tag/314.0.3)
and the [micropip installation API](https://micropip.pyodide.org/en/stable/project/api.html)
for the upstream contracts used here.

WandasはPyodideを使うことで、信号処理とWAVワークフローをブラウザ内だけで実行できます。
このガイドはWandasのPyodideテストハーネスで検証した上記の組み合わせを対象にします。
どちらかを更新するときは、組み合わせを再検証してください。

## Try the complete browser example / 完全なブラウザ例を試す

Save the following as `index.html`, serve its directory over HTTP, and open it
in a modern browser. For example:

以下を`index.html`として保存し、そのディレクトリをHTTPで配信して、モダンブラウザで
開いてください。

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/`. Do not open the page as a `file://` URL:
browser origin rules differ and make fetch behavior harder to diagnose.

このページでは、次の処理を実行します。

1. Wandasをインストールしてimportする
2. 既知の信号を生成し、ローパスフィルタを適用してMatplotlibで描画する
3. PythonからDOMのstatusを更新する
4. ファイル選択で受け取ったWAVを`bytes`として読む
5. CORSが許可された外部URLを`pyfetch → bytes → wd.read`で読む
6. Wandasで生成したWAVを`<audio>`で再生できるようにする

```html
<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Wandas + Pyodide</title>
    <script src="https://cdn.jsdelivr.net/pyodide/v314.0.3/full/pyodide.js"></script>
  </head>
  <body>
    <h1>Wandas + Pyodide</h1>
    <p id="status">Pyodideを読み込んでいます…</p>

    <h2>信号処理とMatplotlib</h2>
    <img id="plot" alt="Wandasで処理した波形" style="max-width: 100%">

    <h2>ローカルWAV</h2>
    <input id="wav-file" type="file" accept=".wav,audio/wav" disabled>
    <p id="file-result">WAVを選択してください。</p>

    <h2>外部URL</h2>
    <button id="fetch-wav" type="button" disabled>CORS許可済みWAVを読む</button>
    <p id="fetch-result"></p>

    <h2>Wandasで生成したWAV</h2>
    <audio id="player" controls></audio>

    <script>
      const PYODIDE_VERSION = "314.0.3";
      const WANDAS_VERSION = "0.6.1";
      const PYODIDE_BASE =
        `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
      const SAMPLE_WAV_URL =
        "https://raw.githubusercontent.com/kasahart/wandas/03ec19684eb29fcfd0f3e3a31413ce434f87d356/learning-path/sample_audio.wav";

      let pyodide;
      let playerObjectUrl;

      async function initialize() {
        pyodide = await loadPyodide({ indexURL: PYODIDE_BASE });
        await pyodide.loadPackage("micropip");

        const plotDataUrl = await pyodide.runPythonAsync(`
import base64
from io import BytesIO
from importlib.metadata import version

import micropip

await micropip.install("wandas==${WANDAS_VERSION}")

import matplotlib.pyplot as plt
import numpy as np
import soundfile  # Verify that the tested WAV backend is importable.
import wandas as wd
from js import document

sampling_rate = 8_000
time = np.arange(sampling_rate, dtype=np.float64) / sampling_rate
samples = (
    0.6 * np.sin(2 * np.pi * 440 * time)
    + 0.2 * np.sin(2 * np.pi * 1_800 * time)
)

original = wd.from_numpy(
    samples,
    sampling_rate=sampling_rate,
    label="browser signal",
    ch_labels=["original"],
)
filtered = original.low_pass_filter(cutoff=1_000)

figure, axis = plt.subplots(figsize=(8, 3))
original.plot(ax=axis, color="0.65", label="original")
filtered.plot(ax=axis, color="C0", label="1 kHz low-pass")
axis.set_xlim(0, 0.02)
axis.legend()
figure.tight_layout()

png = BytesIO()
figure.savefig(png, format="png", dpi=120)
plt.close(figure)
plot_data_url = "data:image/png;base64," + base64.b64encode(
    png.getvalue()
).decode("ascii")

generated_wav_path = "/tmp/wandas-generated.wav"
filtered.to_wav(generated_wav_path)

document.getElementById("status").textContent = (
    f"ready: Wandas {wd.__version__}, soundfile {version('soundfile')}, "
    f"{filtered.sampling_rate:g} Hz, {filtered.n_channels} channel"
)

plot_data_url
        `);

        document.getElementById("plot").src = plotDataUrl;

        const wavBytes = pyodide.FS.readFile("/tmp/wandas-generated.wav");
        playerObjectUrl = URL.createObjectURL(
          new Blob([wavBytes], { type: "audio/wav" }),
        );
        document.getElementById("player").src = playerObjectUrl;
        document.getElementById("wav-file").disabled = false;
        document.getElementById("fetch-wav").disabled = false;
      }

      document.getElementById("wav-file").addEventListener(
        "change",
        async (event) => {
          const file = event.target.files[0];
          if (!file) return;

          const fileBytes = new Uint8Array(await file.arrayBuffer());
          pyodide.globals.set("selected_wav_bytes_js", fileBytes);
          try {
            const summary = await pyodide.runPythonAsync(`
selected_wav = wd.read(
    selected_wav_bytes_js.to_bytes(),
    file_type=".wav",
    source_name="browser-file-selection",
)
(
    f"{selected_wav.sampling_rate:g} Hz, "
    f"{selected_wav.n_channels} channel(s), "
    f"{selected_wav.duration:.3f} s, {selected_wav.data.dtype}"
)
            `);
            document.getElementById("file-result").textContent = summary;
          } catch (error) {
            document.getElementById("file-result").textContent =
              `読み込み失敗: ${error}`;
          } finally {
            pyodide.globals.delete("selected_wav_bytes_js");
          }
        },
      );

      document.getElementById("fetch-wav").addEventListener(
        "click",
        async () => {
          const result = document.getElementById("fetch-result");
          result.textContent = "取得しています…";
          try {
            const summary = await pyodide.runPythonAsync(`
from pyodide.http import pyfetch

remote_response = await pyfetch("${SAMPLE_WAV_URL}")
remote_response.raise_for_status()
remote_wav = wd.read(
    await remote_response.bytes(),
    file_type=".wav",
    source_name="${SAMPLE_WAV_URL}",
)
(
    f"{remote_wav.sampling_rate:g} Hz, "
    f"{remote_wav.n_channels} channel(s), "
    f"{remote_wav.duration:.3f} s"
)
            `);
            result.textContent = summary;
          } catch (error) {
            result.textContent =
              `取得失敗（URL、HTTP status、CORSを確認）: ${error}`;
          }
        },
      );

      initialize().catch((error) => {
        document.getElementById("status").textContent = `初期化失敗: ${error}`;
        console.error(error);
      });

      window.addEventListener("beforeunload", () => {
        if (playerObjectUrl) URL.revokeObjectURL(playerObjectUrl);
      });
    </script>
  </body>
</html>
```

The first load downloads Pyodide and Python packages and can take time. In a
production application, disable controls until initialization completes and
show package-download progress.

初回はPyodideとPythonパッケージをダウンロードするため時間がかかります。本番アプリでは
初期化完了まで操作を無効化し、進捗を表示してください。

## Read an external URL through browser fetch / 外部URLはfetch経由で読む

In a Pyodide browser, use the browser networking stack explicitly:

Pyodideブラウザでは、外部URLを次の経路で読んでください。

```python
from pyodide.http import pyfetch
import wandas as wd

url = "https://example.com/recording.wav"
response = await pyfetch(url)
response.raise_for_status()
wav_bytes = await response.bytes()
frame = wd.read(wav_bytes, file_type=".wav", source_name=url)
```

`file_type=".wav"` is explicit because a byte sequence has no filename from
which to infer a format. `source_name=url` is optional provenance; it does not
perform another download. `pyfetch` is asynchronous, uses the browser Fetch
API, supports HTTPS, and remains subject to the browser's CORS policy. Its
current response contract is documented in
[`pyodide.http`](https://pyodide.org/en/stable/usage/api/python-api/http.html).

byte列からは拡張子を推定できないため、`file_type=".wav"`を明示します。
`source_name=url`は任意の由来情報であり、再ダウンロードは行いません。

!!! warning "`wd.read(URL)` is not the browser URL API"

    Do **not** call `wd.read("https://...")` in a Pyodide browser. Wandas's
    current direct-URL path uses synchronous Python HTTP behavior that does not
    provide the required TLS transport in this environment. Use
    `pyfetch/fetch → bytes → wd.read(..., file_type=".wav")` instead.

    Pyodideブラウザでは`wd.read("https://...")`を直接呼び出さないでください。
    現行の同期HTTP経路はこの環境で必要なTLS transportを提供できません。
    `pyfetch/fetch → bytes → wd.read`を使ってください。

JavaScript's native `fetch()` is an equivalent entry point. Convert its
`ArrayBuffer` to `Uint8Array`, expose it with `pyodide.globals.set`, and call
`to_bytes()` in Python, as the file-input example does.

## Understand the supported boundary / 対応範囲を判断する

| Capability / 機能 | Pyodide browser status / 状態 | Use / 回避策 |
| --- | --- | --- |
| Core `ChannelFrame` creation, metadata, filtering, FFT, and other core operations | Supported / 対応 | Use the normal public Wandas frame APIs. |
| Matplotlib static plots | Supported / 対応 | Save a figure to PNG/SVG bytes and attach it to the DOM, as above. |
| WAV metadata and mono/multichannel reading | Supported / 対応 | Use a local virtual path, `bytes`, or a file-like object. |
| Integer PCM (`PCM_U8`, `PCM_16`, `PCM_24`, `PCM_32`) | Supported / 対応 | `wd.read()` returns channel-first `float64` with libsndfile full-scale conversion. |
| Floating-point WAV (`FLOAT`, `DOUBLE`) | Supported / 対応 | Values are preserved as `float64`; values outside ±1 are not clipped. |
| Partial WAV reads | Supported / 対応 | Pass `start=` and/or `end=` to `wd.read()`. |
| WAV writing and browser playback | Supported / 対応 | Write with `ChannelFrame.to_wav()`, make a `Blob`, and use an `<audio controls>` element. |
| `soundfile` / libsndfile WAV backend | Supported / 対応 | It is the tested backend; no SciPy WAV fallback is needed. |
| External HTTPS URL through `pyfetch` or JavaScript `fetch` | Supported when CORS permits / CORS許可時に対応 | Fetch bytes, then pass them to `wd.read(..., file_type=".wav")`. |
| Direct `wd.read("https://...")` | Not supported in the browser / 非対応 | Use the fetch-to-bytes route. |
| DOM access | Supported on the browser main thread / main threadで対応 | Import `document` from `js`; use worker messages when running Pyodide in a Web Worker. |
| HPSS (`librosa`-backed effects) | Not supported by this target / 非対応 | The required `numba` Pyodide wheel is unavailable; do not install `wandas[effects]` for HPSS here. |
| PyTorch and TensorFlow conversion | Outside the supported target / 対象外 | Their Pyodide wheels are not part of this contract; keep ML inference/training in a native or server environment. |
| `subprocess` and native executables | Not supported / 非対応 | Browser WebAssembly has no normal OS process model; move that step outside the browser. |
| WDF, psychoacoustic extras, marimo integration, and other optional extras | Not established by this guide / 未確立 | Validate separately before relying on them in a browser application. |

`soundfile` is working in the validated environment for WAV reads and writes,
including integer PCM, floating-point data, bytes/file-like sources, partial
reads, mono/multichannel data, and round trips. Adding a SciPy WAV fallback
would duplicate a functioning backend and is neither required nor recommended.

検証済み環境では`soundfile`がWAVの読み書きに正常に動作します。SciPy WAV
fallbackは不要です。

## Account for browser constraints / ブラウザ固有の制約に備える

### CORS

Same-origin URLs normally work. A cross-origin server must return an
appropriate `Access-Control-Allow-Origin` response header. CORS is enforced by
the browser and cannot be bypassed by Wandas or `pyfetch`. A failed request can
mean a network error, HTTP error, or CORS rejection; inspect the browser
developer console and Network panel.

同一originは通常取得できます。cross-originでは配信元のCORS許可が必要です。
Wandasや`pyfetch`からCORSを回避することはできません。

### Autoplay

Browsers commonly block `audio.play()` unless it follows a user gesture. Keep
the native `<audio controls>` UI or start playback from a click/tap handler.
Creating a `Blob` URL does not grant autoplay permission. Revoke old object
URLs with `URL.revokeObjectURL()` when replacing audio or leaving the page.

ブラウザはユーザー操作のない自動再生を通常ブロックします。`<audio controls>`を表示するか、
click/tap handlerから再生してください。

### Memory

Pyodide, scientific Python packages, decoded arrays, Matplotlib figures, and
WAV/PNG copies all consume browser memory. Wandas preserves Dask laziness for
frame operations, but decoding, plotting, WAV encoding, and crossing the
JavaScript/Python boundary eventually materialize data. Limit recording
duration and channel count, release figures and object URLs, and move genuinely
large or long-running work to a worker or server.

Pyodide本体、科学計算パッケージ、復号後の配列、描画、JavaScript/Python間のコピーは
ブラウザメモリを使います。Daskの遅延実行を保っていても、復号・描画・WAV書き出しでは
最終的にデータを実体化します。

### File system

Pyodide's default filesystem is an in-memory, sandboxed POSIX-like filesystem.
It cannot read an arbitrary desktop path, and its files disappear when the
page is reloaded. Copy a user-selected `File` into Python as bytes, as shown
above. If persistence is required, design it explicitly with browser storage
or a supported filesystem mount; do not treat `/tmp` as persistent storage.
The upstream [JavaScript filesystem API](https://pyodide.org/en/stable/usage/api/js-api.html#pyodide.FS)
documents the available mounts and notes that only the default MEMFS works in
all runtimes.

Pyodideの標準filesystemはsandbox内のmemory filesystemです。任意のdesktop pathを
直接読むことはできず、reloadすると内容は消えます。ファイル選択で得た`File`をbytesへ
変換してください。

## Know what CI proves / CIが保証する範囲を知る

The repository's reproducible Pyodide harness installs a wheel built from the
checkout and runs the core tests plus deterministic WAV smoke coverage. It
checks the Python/WASM contract without repeating the full native CPython test
suite.

DOM mutation, same-origin and CORS behavior, and audible playback require a
real browser and browser policy. They are intentionally not required CI in
this change, so applications should keep a small browser-level acceptance
check for their own origin, headers, and autoplay UX.

リポジトリの再現可能なPyodide harnessは、checkoutからbuildしたwheelを使ってcore testsと
決定論的なWAV smokeを実行します。一方、DOM、CORS、実際の音声再生はブラウザpolicyに
依存するため、この変更の必須CIには含めません。アプリ固有のorigin、header、autoplay UXは
実ブラウザで確認してください。
