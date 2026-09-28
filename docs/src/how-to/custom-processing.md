# Add your own processing / 独自処理を組み込む

Use a NumPy function with a Frame first, then give it a stable Recipe identity
when you need to save and replay the workflow. Both paths work in your own
project; you do not need to modify or fork Wandas.
まずNumPy関数をFrameへ適用し、処理手順を保存・再実行したい場合にRecipe対応へ進みます。
どちらも自分のプロジェクト内で実装でき、Wandas本体の変更やforkは不要です。

| Need / 目的 | Path / 方法 |
| --- | --- |
| Use an existing operation / 既存処理を使う | Prefer the standard Frame method / 標準Frameメソッドを使う |
| Run your Python function / 自分の関数を実行する | `frame.apply(func, **params)`; reusable Python code, runtime-only Recipe behavior / Pythonコードとして再利用可能、Recipe保存は不可 |
| Save your processing workflow / 独自処理を含む手順を保存する | Local Operation + Frame method + Recipe registry / 自分のOperation・Frameメソッド・Recipe registryを定義する |
| Contribute a built-in feature / 標準機能として本体へ追加する | [Contributor extension guide / 本体拡張ガイド](../contributing/frame-operation-extensions.md) |

For a guided exploration of shape, dtype, laziness, and metadata, see
<a href="../../learning-path/05_custom_functions.html">教材05（日本語）</a> or
<a href="../../en/learning-path/05_custom_functions.html">Learning Path 05 (English)</a>.
The following code blocks run in order in one Python session.
以下のコードは、同じPythonセッションで上から順に実行できます。

## Apply a NumPy function / NumPy関数を適用する

```python
import numpy as np
import wandas as wd


def scale_channels(data, factor):
    return data * factor


samples = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
source = wd.from_numpy(samples, sampling_rate=8_000, ch_labels=["left", "right"])
scaled = source.apply(scale_channels, factor=2.0)
np.testing.assert_allclose(scaled.to_numpy(), samples * 2.0)
np.testing.assert_allclose(source.to_numpy(), samples)
```

`apply()` builds a lazy Dask graph. When materialized (here by `to_numpy()`),
the callable receives the whole channel-first NumPy array `(channels, samples)`,
including the channel axis for mono input. It is not called once per chunk.
Return a new array with the same dtype; there is no output-dtype callback.
For a sample-axis shape change that preserves the channel count and order,
supply `output_shape_func` so Dask knows the shape before execution. Changing
channel count also requires new channel IDs and metadata: use a standard channel
operation or an extension that constructs those explicitly. A slice alone does
not update the sampling rate or source-time offset:
use a standard semantic operation such as `trim()` when it matches your intent.

`apply()`はDaskの遅延グラフを作り、`to_numpy()`などで実体化したときに、全channel・全sampleの
NumPy配列を関数へ渡します。monoでも入力は`(channels, samples)`で、chunkごとの呼出しでは
ありません。同じdtypeの新しい配列を返してください。shapeを変える場合は
`output_shape_func`も指定します。この方法ではchannel数と順序を維持してください。
channel数を変える場合はIDとmetadataの再定義も必要なので、標準channel操作か、
それらを明示的に構築する拡張を使います。配列をsliceするだけではsampling rateやsource-time offsetは
変わらないため、目的に合う場合は`trim()`などの意味付き標準操作を使います。

Only additional keyword arguments go to your function. `func`, `output_shape_func`,
`output_frame_class`, `output_frame_kwargs`, and `dask_pure` configure `apply()`
itself and are not forwarded; rename conflicting callable parameters.
`sampling_rate` and `pure` are also reserved and rejected. Use `fs` or `sr` for a
sample-rate argument and pass it explicitly (`fs=source.sampling_rate`).
Use `dask_pure=False` for nondeterministic functions. Avoid
mutating inputs or captured state. `apply()` records runtime lineage, but
`RecipePlan.from_frame(scaled)` rejects the arbitrary callable.

関数へ渡されるのは追加のキーワード引数だけです。`func`・`output_shape_func`・
`output_frame_class`・`output_frame_kwargs`・`dask_pure`は`apply()`自身の設定として消費されるため、
独自関数の引数名と重なる場合は関数側を改名してください。関数がsampling rateを必要とする場合は、予約名の
`sampling_rate`を避けて`fs=source.sampling_rate`などと明示します。`pure`も予約名です。
非決定的な関数には`dask_pure=False`を指定し、入力やclosure内の状態を変更しないでください。
履歴は記録されますが、このcallableを含む`scaled`からRecipeを抽出することはできません。

## Keep a portable extension in your project / 自分のプロジェクトでRecipe対応にする

Put the following definitions in your own module, for example `my_processing.py`.
The gain example keeps shape, channel order, and time axes unchanged and returns
`float64` for both integer and floating input. Choose an ID in your own namespace
and keep its meaning stable.

以下の定義を、例えば自分の`my_processing.py`へ置きます。このgain例ではshape・channel順・
時間軸を維持し、整数・浮動小数点の入力をどちらも`float64`で出力します。
Recipe IDには自分の名前空間を使い、保存後も意味を変えないでください。

```python
from collections.abc import Mapping
from typing import Any

import numpy as np

from wandas.frames import ChannelFrame
from wandas.pipeline import (
    RecipePlan,
    default_recipe_registry,
    recipe_definition,
    recipe_operation,
)
from wandas.processing import ChannelIndependentAudioOperation
from wandas.utils.types import NDArrayReal


def validate_gain(params: Mapping[str, Any]) -> None:
    if set(params) != {"factor"}:
        raise ValueError("gain requires exactly one parameter: factor")
    factor = params["factor"]
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not np.isfinite(factor):
        raise ValueError("factor must be a finite real number")


class Gain(ChannelIndependentAudioOperation[NDArrayReal, NDArrayReal]):
    name = "my_project_gain"

    def validate_params(self) -> None:
        validate_gain(self.to_params())

    def _process(self, data: NDArrayReal) -> NDArrayReal:
        return np.asarray(data, dtype=np.float64) * self._config_value("factor")

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        return np.dtype(np.float64)


class ProjectFrame(ChannelFrame):
    @recipe_operation("my_project.audio.gain", validate_params=validate_gain)
    def gain(self, factor: float) -> "ProjectFrame":
        operation = Gain(self.sampling_rate, factor=factor)
        return self._apply_operation_instance(operation)


registry = default_recipe_registry().with_operation(recipe_definition(ProjectFrame.gain))
```

The numerical kernel belongs in `_process()`; inherited `process()` handles lazy
execution. `ChannelIndependentAudioOperation` fits because each output channel
depends only on its corresponding input channel. Use `AudioOperation` for
cross-channel processing. Override `calculate_output_shape()` or
`calculate_output_dtype()` if their contracts change.
Here the explicit `float64` conversion and dtype declaration agree even when an
integer input is multiplied by a fractional factor. `validate_gain()` checks the
complete parameter set as well as the value, rejecting incompatible Recipe
parameters during loading, before execution.

数値計算は`_process()`へ置き、遅延実行は継承した`process()`へ任せます。各出力channelが対応する
入力channelだけに依存するため、この例は`ChannelIndependentAudioOperation`を使います。
channel間を参照する処理には`AudioOperation`を使い、shape・dtypeが変わる場合はそれぞれの
`calculate_output_shape()`・`calculate_output_dtype()`を実装します。
この例は`float64`への変換とdtype宣言を揃え、整数入力に小数のfactorを掛けても一致させます。
`validate_gain()`は値とparameter集合の両方を検証し、未知parameterを実行前のRecipe読込時に拒否します。

`_config_value()` and `_apply_operation_instance()` are protected extension hooks
used here to preserve configuration snapshots, Frame metadata, and lineage.
Keep their use inside the extension and test it when upgrading Wandas.
The operation is instantiated directly, so no numerical `register_operation()`
call is needed. The separate immutable Recipe registry maps the stable ID to its
implementation; it does not modify the default registry or install a method on
ordinary `ChannelFrame` objects. Start with `ProjectFrame.from_numpy()` to use
`.gain()` in your Python workflow.

`_config_value()`と`_apply_operation_instance()`は設定値のsnapshot・Frame metadata・lineageを
維持するために使うprotectedな拡張hookです。使用箇所を拡張内へ閉じ、Wandas更新時に検証してください。
Operationを直接生成するため数値処理registryへの`register_operation()`は不要です。
別途作ったRecipe registryがIDと実装を結び付けます。default registryや通常の`ChannelFrame`は
変更されません。Python上で`.gain()`を呼ぶ入力は`ProjectFrame.from_numpy()`から作ります。

## Extract, save, and replay / 抽出・保存・再実行する

```python
project_source = ProjectFrame.from_numpy(samples, sampling_rate=8_000)
processed = project_source.gain(factor=2.0)
plan = RecipePlan.from_frame(processed, input_names=("signal",), registry=registry)
payload = plan.to_dict()  # JSON-compatible workflow; no callable or samples
restored = RecipePlan.from_dict(payload, registry=registry)
replacement = ProjectFrame.from_numpy(samples + 1.0, sampling_rate=8_000)
replayed = restored.apply({"signal": replacement}, registry=registry)
np.testing.assert_allclose(replayed.to_numpy(), (samples + 1.0) * 2.0)
np.testing.assert_allclose(project_source.to_numpy(), samples)

path = plan.save("gain.recipe.json")
loaded = RecipePlan.load(path, registry=registry)
np.testing.assert_allclose(
    loaded.apply({"signal": replacement}, registry=registry).to_numpy(),
    replayed.to_numpy(),
)
```

Pass the extended registry at extraction, loading, and replay. On another machine
or in a fresh Python process, install/import your extension and construct the same
registry before loading the Recipe. The JSON stores the operation ID, version,
parameters, and input bindings; it does not ship Python code. Missing registrations
or unsupported versions fail explicitly. Change the version when serialized
behavior changes, and retain old implementations if old Recipes must still run.

抽出・復元・再実行の各段階に拡張registryを渡します。別環境や新しいPythonプロセスでも、先に
自分の拡張をinstall/importして同じregistryを構築します。JSONに入るのはID・version・parameter・
入力bindingで、Pythonコード自体は入りません。未登録IDや未対応versionはエラーになります。
保存する振る舞いを変更するときはversionを更新し、過去のRecipeを実行する必要があれば旧実装も維持します。

For domain transitions, multiple inputs, or a built-in contribution, continue with
the [Frame and Operation extension guide](../contributing/frame-operation-extensions.md).
Test numerical results, input immutability, metadata and axes, laziness, and a full
Recipe round trip for your actual processing.
領域変換・複数入力・本体への機能追加は[拡張ガイド](../contributing/frame-operation-extensions.md)へ
進んでください。実際の処理では数値結果・入力不変性・metadataとaxis・遅延性・Recipe往復を検証します。
