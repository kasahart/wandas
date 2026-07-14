# Recipe v2 simplification priorities

- **Status:** Accepted
- **Date:** 2026-07-14
- **Compatibility:** v0.4.0で導入されたRecipe関連契約は再設計可能

## 結論

NumPy/Dask配列を一時Frameへ変換してFrame同士の演算へ寄せる案は採用しない。
sampling rateやmetadataを持たない配列へ仮の意味を与え、broadcastingとlazinessの
例外を増やすため、内部構造は単純にならない。配列はsemanticな`array` bindingとして
保持し、数値演算はdata-level kernelで行う。

単純化の主軸は、Recipeの事実を`LineageNode`上の単一semantic descriptorへ集約し、
Dask marker、runtime replay descriptor、call family、codec、history snapshotという
重複表現を削除することである。

## 確定優先度

| 優先度 | 項目 | 確定契約 | 期待する削減 |
| --- | --- | --- | --- |
| P0 | provenance正本化 | 全Frameが明示Source lineageを持ち、lineageだけをhistory/Recipeの正本にする | optional lineage、source fallback、Dask inspectionを削除 |
| P0 | semantic operation統一 | ID/version/bindings/paramsを1つの不変descriptorにする | descriptor family、codec、call familyを削除 |
| P0 | registry統一 | 不変registry entryがcapture/validation/Frame handlerを所有する | 中央family分岐、import path、mutable globalを削除 |
| P0 | indexing一本化 | selectorを1回正規化し、1つのprivate selection kernelへ渡す | 4箇所のgrammar再実装と再Frame化を削減 |
| P0 | history一本化 | 公開履歴は`operation_history`だけ。WDFはdisplay prefixとして保存 | summaries/snapshot/delta伝播を削除 |
| P1 | operation実行入口 | runtime operationは数値処理だけ。semantic captureは公開入口だけ | alias、duck-typed replay、marker taskを削除 |
| P1 | additionの役割分離 | `+`はelementwise、`mix`は長さ/SNR、`add_channel`はchannel追加 | 曖昧な`.add()`と暗黙変換を削除 |
| P1 | Frame immutability | `inplace`を廃止し、常に新しいFrameを返す | mutation分岐とRecipe補正を削除 |
| P1 | schema 2 | stable IDのみ保存し、v1をfail-closedで拒否 | callable loader、互換層、型family schemaを削除 |
| P2 | public surface縮小 | Recipe public APIをfrom_frame/apply/to_dict/from_dictに限定 | builder/call/codec public APIを削除 |
| P3 | `previous` | navigation/data比較に限定して維持 | provenanceとの同期責務を禁止 |

## 配列・算術・mixの契約

- NumPy/Daskは同じ`array`入力kindで、schema/historyへ実装種別を保存しない。
- 同じ配列objectを複数回渡してもbinding occurrenceごとに別inputにする。
- Frame-array elementwise演算はNumPy/Dask broadcastingを維持する。
- Frame-Frame elementwise演算はsampling rate、shape、semantic axesを一致させる。
- `mix(other, *, align="strict", snr_db=None)`はscalarを受け付けない。
- `mix`はsource-time alignmentを行わない。異なる時間帯のFrameでも配列indexでmixし、
  左Frameのsource-time offsetをそのまま残す。
- sampling rateは一致必須。channel数は一致またはotherがmonoの場合だけ許可する。
- 出力長は左Frameが所有する。`strict`は同長、`pad`は短いotherだけ、`truncate`は
  長いotherだけを受け付け、逆方向の不一致はerrorにする。
- `add_channel`のraw arrayは1Dまたは`(1, samples)`のみ。multi-channelは
  `ChannelFrame`を要求し、flattenしない。

## 今回行わないこと

DI context全面導入、Recipe以外を含むregistry全面再設計、`calls.py`の機械的分割、
任意Python objectを扱う汎用ReplayValue、source-timeベースの自動mix alignmentは行わない。
PLOCやmicrobenchmarkを優先して説明的な名前、validation、型安全性を削らない。
