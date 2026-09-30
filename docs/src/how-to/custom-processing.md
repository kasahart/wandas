# Add your own processing / 独自処理を組み込む

This guide assumes you can write a Python function that works on a NumPy array.
You will first apply that function to data held by Wandas, then make the processing
steps reusable on other data. Both examples belong in your own project; you do
not need to change Wandas itself.

NumPy配列を受け取るPython関数を書ける方を対象に、自分の処理をWandasで使う方法を説明します。
まず手元の関数を適用し、次にその処理手順を保存して別のデータへ使う方法へ進みます。
どちらも自分のプロジェクト内で実装でき、Wandas本体の変更は不要です。

## Understand the objects in the examples / 例に登場するものを知る

| Term / 用語 | Meaning in this guide / このガイドでの意味 |
| --- | --- |
| **Frame** | A Wandas object that holds sample data together with information such as sampling rate and channel names. `ChannelFrame` represents waveforms over time. / 波形などのデータと、サンプリング周波数・チャネル名などを一緒に扱うオブジェクト。時間に沿った波形には`ChannelFrame`を使います。 |
| **Operation** | A class that implements one processing step, such as multiplying every sample by a factor. / 「各サンプルの値を2倍にする」など、1つの処理を実装するクラスです。 |
| **Recipe / RecipePlan** | A reusable description of processing steps. `RecipePlan` is the Python class used to extract, save, load, and run that description. / 別のデータにも使える「処理手順書」がRecipeです。Pythonでは`RecipePlan`というクラスで、その手順を取り出す・保存する・読み込む・実行する操作を行います。 |

For example, after doubling recording A, you can extract a Recipe describing
“multiply by 2” and apply it to recording B. It stores the processing steps and
settings, not recording A or its computed result. A Recipe also does not contain
the Python code for your custom function: the implementation must be available
where you run it.

例えば、録音Aの値を2倍にしたあと、「値を2倍にする」という手順をRecipeとして保存し、録音Bにも
適用できます。Recipeが保存するのは処理の種類や倍率などの設定で、録音Aの波形や計算結果ではありません。
独自処理のPythonコードもRecipeには入らないため、実行先でもそのコードを読み込む必要があります。

You only need the first section if you want to call your Python function.
Continue to the Recipe sections when you also want to save and reload the steps.

自分の関数を呼び出すだけなら、最初の「NumPy関数を適用する」節で十分です。
処理手順を保存・読込したい場合に、Recipeの節へ進んでください。

| Need / 目的 | Path / 方法 |
| --- | --- |
| Use an existing operation / 既存処理を使う | Use the corresponding Frame method, such as `trim()` / `trim()`など、目的に合うFrameのメソッドを使う |
| Run your Python function / 自分の関数を実行する | `frame.apply(func, **params)`; reusable as Python code, but these calls cannot be saved in a Recipe / Pythonコードとして繰り返し使えるが、この呼出しはRecipeには保存できない |
| Save your processing steps / 独自処理を含む手順を保存する | Define an Operation and a Frame method, then register how to run them from a Recipe, as shown below / 以下の例のように、処理クラスとFrameのメソッドを定義し、Recipeから呼び出せるよう登録する |
| Contribute a built-in feature / Wandas本体へ標準機能を追加する | [Contributor extension guide / 本体拡張ガイド](../contributing/frame-operation-extensions.md) |

For a runnable example of applying a function, giving your processing a name,
and saving and replaying its steps on another input, see
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

`apply()` prepares the calculation without running your function immediately.
This is called **lazy execution**. Dask, the execution library used by Wandas,
keeps a graph of the calculations to perform. Calling `to_numpy()` in the example
runs them and returns the resulting NumPy array.

`apply()`を呼んだ時点では関数を実行せず、「あとで行う計算」を組み立てます。これを**遅延実行**と
呼びます。Wandasが内部で使う計算ライブラリDaskが、計算の順序や依存関係を保持します。
上の例では`to_numpy()`を呼んだときに実際に計算し、結果をNumPy配列として取り出しています。

Your function receives the whole array with shape `(channels, samples)`:
rows are channels, such as microphones, and columns are successive samples.
This arrangement is called **channel-first**. Even a one-channel signal is a
2-D array here. The function receives all samples at once, rather than separate
blocks of the input (called chunks).

関数には`(チャネル数, サンプル数)`という形の配列が渡されます。行がマイクなどのチャネル、
列が時間に沿ったサンプルです。この並びを**channel-first**と呼びます。
1チャネルでも2次元配列です。入力を分割した小さなブロック（chunk）ごとではなく、
全チャネル・全サンプルが一度に渡されます。

Return a new array with the same **dtype** (element type, such as `float32`).
This API has no output-dtype callback. If you change the number of samples,
provide `output_shape_func`, a function that tells Dask the output **shape**
(array dimensions) before computation. Keep channel count and order unchanged.
Changing channel count also requires new channel IDs and metadata—information
attached to the data, such as channel names and units. Use a standard channel
operation or an extension that constructs that information explicitly.

入力と同じ**dtype**（`float32`など、配列の要素の型）の新しい配列を返してください。
`apply()`には出力dtypeを指定する機能がありません。サンプル数を変える場合は、出力の**shape**
（配列の各次元の大きさ）を計算前に伝える関数`output_shape_func`も指定します。
この方法ではチャネル数と順序を維持します。チャネル数を変える場合は、チャネルを識別するIDや、
名前・単位などの付属情報（**metadata／メタデータ**）の再定義も必要です。
その場合は標準のチャネル操作か、これらの情報を明示的に構築する拡張を使います。

Array slicing alone does not update sampling rate or **source-time offset**
(the starting time of the result within the original recording). For example,
use `trim()` to cut out a time interval and update its starting time together.

配列をsliceするだけでは、サンプリング周波数や**source-time offset**（元の収録のどの時刻から
始まるデータかを示す値）は変わりません。例えば時間区間を切り出すなら`trim()`を使うと、
波形とその開始時刻を一緒に更新できます。

Only additional keyword arguments go to your function. `func`, `output_shape_func`,
`output_frame_class`, `output_frame_kwargs`, and `dask_pure` configure `apply()`
itself and are not forwarded; rename conflicting callable parameters.
`sampling_rate` and `pure` are also reserved and rejected. Use `fs` or `sr` for a
sample-rate argument and pass it explicitly (`fs=source.sampling_rate`).
Use `dask_pure=False` for functions whose result may change for the same input,
such as an unseeded random calculation. Avoid changing the input array or mutable
objects referenced from outside the function. `apply()` records **lineage**,
which tracks the inputs and operations that produced a Frame. That record alone
does not make an arbitrary Python function (a callable) replayable from a Recipe:
`RecipePlan.from_frame(scaled)` rejects this step.

関数へ渡されるのは追加のキーワード引数だけです。`func`・`output_shape_func`・
`output_frame_class`・`output_frame_kwargs`・`dask_pure`は`apply()`自身の設定として消費されるため、
独自関数の引数名と重なる場合は関数側を改名してください。関数がsampling rateを必要とする場合は、予約名の
`sampling_rate`を避けて`fs=source.sampling_rate`などと明示します。`pure`も予約名です。
同じ入力でも結果が変わる関数（seedを固定しない乱数処理など）には`dask_pure=False`を指定します。
入力配列や、関数の外から参照している書き換え可能なオブジェクトは変更しないでください。
`apply()`も「どの入力にどの処理を行ってこのFrameになったか」という記録（**lineage**）を残します。
ただし、記録があっても任意のPython関数をRecipeから再実行できるわけではありません。
そのため、この関数を含む`scaled`から`RecipePlan.from_frame()`で手順を取り出そうとするとエラーになります。

## Give your processing a name a Recipe can use / Recipeから呼べる処理を定義する

A Recipe needs a name that identifies your processing and a way to find its
implementation. The example below defines three things:

1. `Gain`: the numerical Operation that multiplies samples by a factor.
2. `ProjectFrame.gain()`: the method you call to apply it to a Frame.
3. `registry`: a lookup table connecting the Recipe operation ID
   `my_project.audio.gain` to the method that implements it.

Recipeには、処理を識別する名前と、その名前から実装を見つける仕組みが必要です。
次の例では3つを定義します。

1. `Gain`：サンプルの値に倍率を掛ける計算を担当するOperation。
2. `ProjectFrame.gain()`：Frameに対してその処理を呼び出すメソッド。
3. `registry`：Recipeに保存する処理名`my_project.audio.gain`と、実際に呼ぶメソッドを結び付ける
   対応表。この対応表を**registry（レジストリ）**と呼びます。

Put these definitions in your own Python file (module), for example
`my_processing.py`. The gain example preserves shape, channel order, and time
axes, and returns `float64` for both integer and floating input. The prefix
`my_project` is your namespace: choose a project-specific prefix to distinguish
your IDs from other extensions, and keep each ID's meaning stable.

以下の定義を、例えば自分のPythonファイル`my_processing.py`へ置きます。
このgain例では配列の形・チャネル順・時間軸を維持し、整数・浮動小数点の入力をどちらも
`float64`で出力します。処理名の先頭の`my_project`は、他の拡張と名前が重ならないようにする
自分のプロジェクト用の名前（**名前空間**）です。保存後も同じ処理名が同じ意味を持つようにしてください。

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

The numerical calculation belongs in `_process()`; inherited `process()` handles lazy
execution. `ChannelIndependentAudioOperation` fits because each output channel
depends only on its corresponding input channel. Use `AudioOperation` for
cross-channel processing. Override `calculate_output_shape()` or
`calculate_output_dtype()` when the output dimensions or element type differ from the input.
Here the explicit `float64` conversion and dtype declaration agree even when an
integer input is multiplied by a fractional factor. `validate_gain()` checks the
allowed argument names as well as their values, rejecting unsupported Recipe
settings during loading, before execution.

数値計算は`_process()`へ置き、遅延実行は継承した`process()`へ任せます。各チャネルを
他のチャネルと独立に処理できるため、この例は`ChannelIndependentAudioOperation`を使います。
チャネル間を参照する処理には`AudioOperation`を使い、配列の形・要素の型が変わる場合はそれぞれの
`calculate_output_shape()`・`calculate_output_dtype()`を実装します。
この例は`float64`への変換とdtype宣言を揃え、整数入力に小数の倍率を掛けても一致させます。
`validate_gain()`は引数の名前と値の両方を検証し、未対応の引数を実行前のRecipe読込時に拒否します。

The names beginning with `_` here are **protected extension hooks**: helper
methods used inside an extension. `_config_value()` reads a saved copy of an
Operation setting; `_apply_operation_instance()` applies the Operation while
carrying Frame information and lineage into the result. Keep their use inside
the extension and test it when upgrading Wandas.

ここで使う`_`から始まるメソッドは、拡張を実装するための補助メソッド（**protected extension hook**）
です。`_config_value()`はOperationが保持する設定値のコピーを読み、`_apply_operation_instance()`は
Frameの付属情報や処理の記録を引き継ぎながらOperationを適用します。
使用箇所を拡張の定義内に閉じ、Wandas更新時に動作を検証してください。

`@recipe_operation(...)` declares the method's Recipe name and argument checks.
`recipe_definition(ProjectFrame.gain)` retrieves that declaration for registration.

`@recipe_operation(...)`はメソッドのRecipe用の名前と引数の検証方法を宣言します。
`recipe_definition(ProjectFrame.gain)`は、その宣言を対応表へ登録するために取り出します。

The Operation is created directly by `Gain(...)`, so this example needs no
`register_operation()` call (the separate registration used to look up numerical
Operations by name). The Recipe registry is for looking up **Recipe IDs**.
`default_recipe_registry()` provides the standard entries, and `with_operation()`
returns a new registry with your entry added, leaving the original unchanged.
This does not add `.gain()` to ordinary `ChannelFrame` objects: create your inputs
with `ProjectFrame.from_numpy()` to call your new method.

この例は`Gain(...)`で処理を直接作るため、数値処理を名前から探すための別の登録関数
`register_operation()`は不要です。ここで作るRecipe registryは、**Recipeに保存した処理名**を
探すためのものです。`default_recipe_registry()`で標準の対応表を取得し、`with_operation()`で
自分の処理を加えた新しい対応表を作ります。元の対応表は変更しません。
通常の`ChannelFrame`に`.gain()`が追加されるわけではないので、自分のメソッドを使う入力は
`ProjectFrame.from_numpy()`で作ります。

## Extract, save, and replay / 処理手順を取り出して保存・再実行する

`RecipePlan.from_frame()` builds a Recipe from the processing recorded on the
result Frame. We call this **extraction**. `input_names=("signal",)` gives the
replaceable input a name; `apply({"signal": replacement})` supplies new data for
that name. This name-to-input connection is an **input binding**.

`RecipePlan.from_frame()`は、処理後のFrameに記録された手順からRecipeを作ります。
これを**抽出**と呼びます。`input_names=("signal",)`は差し替えたい入力に付ける名前です。
`apply({"signal": replacement})`で、その名前に新しいデータを渡して再実行します。
この「名前と入力の対応」が**入力binding**です。

The two `apply()` methods have different roles: `frame.apply(func)` accepts a
Python function, while `restored.apply({"signal": replacement})` runs the steps
already stored in a Recipe.

同じ`apply()`という名前でも、`frame.apply(func)`はPython関数を受け取るメソッドで、
`restored.apply({"signal": replacement})`はRecipeに保存済みの手順を実行するメソッドです。

`to_dict()` returns the steps as a Python dictionary and `from_dict()` rebuilds
a Recipe from it. `save()` and `load()` use a JSON file, a text format for storing
names, numbers, and lists. The first comparison below proves the same gain runs
on different samples; the second proves loading the saved file gives the same result.

`to_dict()`は手順をPythonの辞書にし、`from_dict()`はその辞書からRecipeを復元します。
`save()`・`load()`は、名前・数値・リストなどをテキストで保存するJSON形式のファイルを使います。
以下の最初の比較で別のデータにも同じ倍率が適用されることを確認し、次の比較でファイルから
読み込んだ手順でも同じ結果になることを確認します。

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
registry before loading the Recipe. The JSON stores the operation ID, version (the revision of its saved behavior),
argument values, and named input connections; it does not ship Python code. Missing registrations
or unsupported versions fail explicitly. Change the version when serialized
behavior changes, and retain old implementations if old Recipes must still run.

抽出・復元・再実行の各段階に、自分の処理を登録した対応表`registry`を渡します。別の環境や新しく
起動したPythonでも、先に拡張コードをインストール・importして同じ対応表を作ります。
JSONに入るのは処理名・その処理の版（version）・引数の値・入力名の対応で、Pythonコード自体は入りません。
未登録の処理名や未対応の版を読み込むとエラーになります。
保存する処理の意味を変更するときはversionを更新し、過去のRecipeも実行したい場合は旧版の実装を残します。

For changes of data representation (such as waveforms to spectra), multiple inputs, or a built-in contribution, continue with
the [Frame and Operation extension guide](../contributing/frame-operation-extensions.md).
Test numerical results, unchanged inputs, attached information and axes, deferred computation, and saving/loading/replaying a Recipe for your actual processing.
波形から周波数スペクトルへの変換、複数の入力を使う処理、本体への機能追加は[拡張ガイド](../contributing/frame-operation-extensions.md)へ
進んでください。実際の処理では、値が正しいこと、元の入力が変わらないこと、付属情報と時間・周波数などの軸が
正しいこと、必要になるまで計算を待つこと、手順を保存・読込して再実行できることを検証します。
