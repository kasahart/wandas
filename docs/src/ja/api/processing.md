# 信号処理モジュール（Processing）

このページではWandasの信号処理モジュールについて説明します。信号処理モジュールは、時系列データに対する様々な処理機能を提供します。

## 時系列処理

時系列データに対する処理関数です。これらの関数は基本的に`ChannelFrame`メソッドを通じて使用します。

```python
# 一般的な使用方法
import wandas
frame = wandas.read_wav("audio.wav")
filtered_frame = frame.filter(cutoff=1000, filter_type="lowpass")
resampled_frame = frame.resample(target_rate=16000)
```

### 主要な処理クラス

フィルタリングや他の信号処理機能は、内部的には以下のクラスによって実装されています：

::: wandas.processing.time_series.HighPassFilter
    handler: python

::: wandas.processing.time_series.LowPassFilter
    handler: python

::: wandas.processing.time_series.ReSampling
    handler: python

::: wandas.processing.time_series.AWeighting
    handler: python

## AudioOperation

`AudioOperation` クラスは音声処理操作の抽象化と連鎖処理を可能にします。

```python
from wandas.processing.time_series import AudioOperation

# 使用例
import wandas
frame = wandas.read_wav("audio.wav")

# 複数の処理を連鎖させる
operation = (
    AudioOperation()
    .add_step("filter", cutoff=1000, filter_type="lowpass")
    .add_step("normalize")
)

# 処理を適用
processed_frame = operation.apply(frame)
```

::: wandas.processing.time_series.AudioOperation
    handler: python
    selection:
      members:
        - __init__
        - add_step
        - apply
