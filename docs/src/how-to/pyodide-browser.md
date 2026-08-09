# Run Wandas in a Pyodide browser / PyodideブラウザでWandasを使う

Wandas can run core signal processing and WAV workflows in a browser with
Pyodide. The repository's `bash scripts/test_pyodide.sh` harness checks the
supported Python/WASM boundary and a deterministic WAV smoke test.

WandasはPyodideを使って、ブラウザ内で基本的な信号処理とWAV処理を実行できます。
repositoryの`bash scripts/test_pyodide.sh` harnessが、
Python／WASM境界と決定論的なWAV smokeを検証します。

## Install and process / installして処理する

Install Wandas in the current Pyodide runtime, import it, generate a signal,
and process it. This is a short API smoke example; the exact compatible
Pyodide/Wandas version set is maintained by the harness and the
[complete browser example](https://github.com/kasahart/wandas/blob/main/examples/pyodide/index.html).

現在のPyodide runtimeへWandasをinstallしてimportし、信号を生成して処理する簡易API smokeです。
互換性のあるPyodide／Wandasの正確なversion setは、harnessと
[完全なブラウザ例](https://github.com/kasahart/wandas/blob/main/examples/pyodide/index.html)で管理します。

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

## Run the repository check / repositoryの検証

From the repository root, run:

```bash
bash scripts/test_pyodide.sh
```

The default check builds a candidate wheel from the current checkout, verifies
that the browser example pins the same Wandas version, runs the browser-guide
workload against that wheel, and then runs the Pyodide test subset. It does not
install Wandas from PyPI.

To verify an exact version after it has been published to PyPI, run:

```bash
bash scripts/test_pyodide.sh published 0.7.1
```

The release workflow runs this published-package smoke after the PyPI upload
and before creating the GitHub Release. DOM behavior, CORS headers, and audio
autoplay still require a real browser check for your origin.

repository rootから引数なしで実行すると、checkoutから候補wheelをbuildし、ブラウザ例との
version整合性、候補wheelを使うbrowser-guide workload、Pyodide test subsetを検証します。
PyPIの公開artifactはinstallしません。公開後のversionを確認する場合は
`bash scripts/test_pyodide.sh published 0.7.1`を実行します。この公開artifact smokeは
release workflowでもPyPI upload後、GitHub Release作成前に実行されます。DOM、CORS header、
audio autoplayは対象originの実ブラウザでも確認してください。
