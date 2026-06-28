# PR #236 実装複雑性メモ

## 対象

このメモは PR #236「Guard audio operation lineage config」の実装を確認し、複雑化している実装箇所と、その複雑さを要求している仕様を整理したものです。

現在のローカル作業ツリーは GitHub 上の PR head より進んでいます。特に、以前入っていた `AudioOperation` の public 属性ガード実装は簡素化され、現状は `to_params()` によって lineage 用パラメータを明示する方向に寄っています。

## PR #236 の設計意図

- `AudioOperation` を完全 immutable な値オブジェクトではなく、runtime lineage object として扱う。
- `operation_history` は serializable metadata として維持する。
- `operations` は live runtime lineage として維持する。
- 通常の public API 経由で、pending compute、`operations`、`operation_history` が不一致にならないようにする。
- private / reflective mutation、任意 callable の内部状態、Python object としての完全 immutability は対象外とする。
- この段階では `OperationRecord` のような transfer object は導入しない。

## 複雑化している実装

### 1. `operations` と `operation_history` の二重 provenance

現在の実装は provenance を 2 系統で管理しています。

- `operation_history`: JSON 化しやすい metadata record。
- `operations`: live な `AudioOperation` object の runtime lineage。

主な関係箇所:

- `BaseFrame.operations`
- `BaseFrame._create_new_instance`
- `BaseFrame._binary_op`
- `BaseFrame._apply_operation_impl`
- `BaseFrame._apply_operation_instance`
- `SpectralFrame.to_channel_frame()` や `SpectrogramFrame.to_channel_frame()` などの domain transition

この仕様により、frame を生成するほぼすべての経路で runtime operations を引き継ぐか、追加するか、マージするか、落とすかを判断する必要が出ています。

### 2. runtime params snapshot

`AudioOperation.params` は read-only な defensive snapshot として扱われます。

主な構成要素:

- `_snapshot_config_value`
- `_ParamsSnapshot`
- `_ACTIVE_OPERATION_PARAMS`
- `AudioOperation._create_named_wrapper`

重要な挙動は、Dask の lazy graph が「graph 構築時点の operation config」で compute されることです。compute 前に operation object や返された container を変更しても、既存 graph の計算結果が変わってはいけない、という契約を支えています。

単純な `dict` より重くなる理由:

- NumPy array を共有 mutable state にしない必要がある。
- Dask array は eager compute せずに扱う必要がある。
- nested mapping / list / tuple / set を snapshot する必要がある。
- NumPy array や Dask array を含む params equality が必要になる。
- delayed execution 中に operation code が `self.params` を読むケースがある。

### 3. 各 operation subclass の `to_params()` boilerplate

現在のローカル実装では、config の所有を concrete operation class に寄せています。多くの operation class が execution config を private 属性に保存し、`to_params()` で lineage 用 params を返します。

例:

- filters は `_cutoff`, `_order`, `_low_cutoff`, `_high_cutoff` を持つ。
- effects は `_norm`, `_axis`, `_threshold`, `_fill` や HPSS kwargs の snapshot を持つ。
- temporal operations は `_target_sr`, `_start`, `_end`, sample bounds, RMS params などを持つ。
- spectral operations は FFT/window/hop/detrend/scaling など正規化済み default を持つ。
- custom operation は callable を private に保持し、params は別途 snapshot する。

public 属性 interception よりは明示的で読みやすい一方、同じ contract を多数の subclass に展開するため、横断的な boilerplate が増えています。

### 4. default materialization の同期

一部 operation は内部で default を正規化します。`operations` が public runtime provenance になると、以下が一致している必要があります。

- operation object の params
- frame metadata
- `operation_history`
- 実際の compute 挙動

影響が大きい箇所:

- `fft(n_fft=None)`: lineage 作成前に `n_fft` を materialize する必要がある。
- `welch()`: `hop_length`, `win_length`, `detrend`, `noverlap` を一致させる必要がある。
- `coherence`, `csd`, `transfer_function`: 正規化済み spectral params を history / metadata に記録する必要がある。

これは「実装上の default」が「public provenance contract」になるため、レビュー follow-up が増えやすい領域です。

### 5. binary frame operation の lineage merge

frame 同士の binary operation では、左右 operand の runtime lineage と operation history をマージします。

現在の意味合い:

- left の `operations` を保持する。
- right の `operations` を後ろに追加する。
- right の `operation_history` を追加する。
- 最後に binary operator 自体の record を追加する。

妥当な仕様ではありますが、以前は data と metadata の結合だけで済んでいた低レイヤ helper に、lineage の順序や重複に関する判断が入り込んでいます。

### 6. legacy constructor 互換分岐

`_constructor_accepts_kwarg()` によって、frame constructor が `operations` を受け取れるかを判定しています。受け取れない場合は `operations` を渡しません。

これは `BaseFrame` を継承しつつ、新しい `operations` 引数を持たない legacy / test frame を壊さないための互換処理です。

複雑化している点:

- 新しい frame constructor には runtime lineage が渡る。
- legacy constructor では runtime lineage が silent に落ちる。
- ただし `operation_history` は維持される。

結果として、「`operations` は常に `operation_history` に追随する」という単純な invariant が弱くなっています。

### 7. JSON-friendly history 変換

`_mutable_config_value()` は operation params を history に保存できる形へ変換します。

扱っているもの:

- Dask array は shape / dtype / chunks の descriptor にする。
- NumPy array は list にする。
- NumPy scalar は Python scalar にする。
- 非 finite float は `None` にする。
- set / frozenset は list にする。
- 未知の object は string にする。

これは `operation_history` を serializable metadata として維持しつつ、`operations` では live Python object を保持するという二重構造から生じています。

## 複雑さを敷いている仕様

主に以下の仕様が実装の重さを生んでいます。

1. `operation_history` と `operations` の両方を public provenance surface として維持する。
2. `operation_history` は serializable metadata でなければならない。
3. `operations` は live runtime operation object を保持しなければならない。
4. 通常の API 利用では、2 つの provenance surface が一致していなければならない。
5. lazy Dask graph は operation 作成時点の config で compute されなければならない。
6. `params` から mutable internal state が露出してはいけない。
7. operation default は provenance として露出する前に materialize されなければならない。
8. domain transition でも lineage を落としてはいけない。
9. binary operation では左右 operand の lineage を保存しなければならない。
10. `operations` 引数を受け取らない legacy frame constructor も壊してはいけない。

## 主なトレードオフ

複雑さの根は、runtime operation object と serializable history の両方を first-class provenance として公開している点です。

両方を維持する場合、frame 作成経路のほぼ全域で同期処理が必要になります。一方で、片方をもう片方から derive する設計にする、または `operations` を private / 非 contract 扱いにするなら、伝播や default 同期の負担はかなり小さくできます。

現在のローカル実装は `to_params()` 方式に寄せることで public 属性ガードの魔術的な複雑さを減らしています。ただし、`operations` と `operation_history` を両方 contract として維持する限り、dual provenance 由来の複雑さは残ります。

## シンプル化の方向性: 計算 lineage を正にする

メタ情報変更やチャネル選択は一旦 lineage から外し、数値データを生成・変換する計算処理だけを厳密に記録する方針がよいです。

この場合、記録対象は以下のような処理です。

- `normalize`
- `low_pass_filter`
- `high_pass_filter`
- `resample`
- `trim`
- `fft`
- `stft`
- `welch`
- `coherence`
- `roughness`
- `add_with_snr`
- `apply(...)`
- `+`, `-`, `*`, `/`, `**` などの binary compute

記録しない対象は以下です。

- `rename_channels`
- channel selection / slicing
- metadata update
- label 変更
- `persist`
- `to_xarray`
- plot / describe 系

この方針では、`operation_history` を source of truth にしません。source of truth は計算 lineage です。`operation_history` は必要なときに lineage から生成する serializable view にします。

## 厳密に残すなら flat list ではなく tree / DAG

「何にどんな計算がされたか」を厳密に残すなら、`operations: tuple[...]` の flat list では足りません。

例えば以下の計算があります。

```python
left = raw.normalize()
right = noise.low_pass_filter(1000)
result = left + right
```

flat list だと次のようになります。

```text
normalize
lowpass_filter
+
```

これは単純ですが、`lowpass_filter` が `left` にかかったのか `right` にかかったのかが分かりません。厳密には次の構造です。

```text
add
├── left: normalize(raw)
└── right: low_pass_filter(noise)
```

したがって、厳密性を重視するなら lineage は tree / DAG として保持します。

最小構成のイメージ:

```python
@dataclass(frozen=True)
class LineageNode:
    operation: ComputeOperation
    inputs: tuple["LineageNode", ...] = ()
```

frame 側は、flat な `operations` tuple ではなく、現在の frame を作った計算 node を持ちます。

```python
class BaseFrame:
    lineage: LineageNode | None
```

serializable な履歴は `lineage` から都度生成します。

```python
{
    "operation": "+",
    "params": {},
    "inputs": [
        {
            "operation": "normalize",
            "params": {"norm": "inf"},
            "inputs": [{"source": "raw"}],
        },
        {
            "operation": "lowpass_filter",
            "params": {"cutoff": 1000, "order": 4},
            "inputs": [{"source": "noise"}],
        },
    ],
}
```

## `AudioOperation` だけでは足りない

計算処理だけを厳密に記録する場合でも、`AudioOperation` だけでは表現できない計算があります。

特に binary operation は数値データを変える compute ですが、現状は Dask array の直接演算であり、`AudioOperation` object ではありません。

そのため、共通の lineage record としては `AudioOperation` より少し広い抽象が必要です。

例:

```python
class ComputeOperation(Protocol):
    name: str

    def to_params(self) -> Mapping[str, Any]: ...
```

その上で:

- `AudioOperation` は unary compute operation。
- `BinaryOperation` は `+`, `-`, `*`, `/`, `**` などの binary compute record。
- `SourceNode` は元データの起点。
- `LineageNode` は operation と input node を保持する。

この構造にすると、`frame1 + frame2` のような計算でも、左右それぞれにどの処理が適用されていたかを失わずに保持できます。

## 推奨する最終 contract

推奨する contract は以下です。

```text
source of truth: frame.lineage
operation_graph: frame.lineage から生成される厳密な serializable tree
operation_history: compatibility 用の flat summary、または deprecated
```

`operation_history` を残す場合でも、正は `operation_history` ではなく `lineage` です。

```text
lineage -> operation_graph
lineage -> operation_history
```

という一方向の生成にします。

これにより、現在の dual provenance 同期問題を避けつつ、計算処理については「何にどんな処理がされたか」を厳密に残せます。

## この方針で減る複雑さ

- `operations` と `operation_history` を別々に更新する必要がなくなる。
- frame 作成経路ごとに history と operations の同期を考えなくてよくなる。
- binary operation の右辺 lineage を flat に混ぜて意味が曖昧になる問題を避けられる。
- metadata / channel selection / rename を計算 lineage から切り離せる。
- `operation_history` は互換用 view に下げられる。

## 残る設計判断

残る判断は以下です。

1. 既存 `operations` property を `lineage` に置き換えるか、互換 property として残すか。
2. `operation_history` を deprecated にするか、flat summary として維持するか。
3. I/O で保存する正式形式を `operation_graph` にするか。
4. old `operation_history` しかないファイルを読んだ場合、復元不能な imported history として扱うか。
5. `BinaryOperation` の params に scalar / array / frame operand をどこまで記録するか。

特に 5 は重要です。frame-frame binary operation では operand の実データ全体を history に入れるべきではありません。右辺は `inputs` によって表現し、params には operator 種別や scalar 値など、serializable で軽い情報だけを入れるのがよいです。
