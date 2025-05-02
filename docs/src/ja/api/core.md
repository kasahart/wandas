# コアモジュール（Core）

このページではWandasのコアモジュールについて説明します。コアモジュールはWandasの基本データ構造と機能を提供します。

## BaseFrame

`BaseFrame`はWandasの全てのフレームクラスの基底クラスです。直接このクラスをインスタンス化することはなく、派生クラス（ChannelFrameなど）を通じて使用します。

```python
from wandas.core.base_frame import BaseFrame
```

::: wandas.core.base_frame.BaseFrame
    handler: python
    selection:
      members:
        - __init__
        - data
        - metadata
        - shape
        - n_channels
        - channels
        - sampling_rate
        - compute
        - plot
        - persist

## ChannelMetadata

`ChannelMetadata`はチャンネルごとのメタデータを管理するためのクラスです。チャンネル名、単位、その他の付加情報を保持します。

```python
from wandas.core.metadata import ChannelMetadata

# 使用例
metadata = ChannelMetadata(label="左チャンネル", unit="dB", extra={"デバイス": "マイク"})
```

::: wandas.core.metadata.ChannelMetadata
    handler: python
    selection:
      members:
        - __init__
        - __getitem__
        - __setitem__
        - to_json
        - from_json
