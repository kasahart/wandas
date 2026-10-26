# WDF File I/O / WDFファイル入出力

The generated `save` and `load` docstrings are the public WDF API contract.
They describe typed Frame data, metadata, and lazy-load behavior. WDF stores a
concrete Frame result, not a replayable Recipe; use the
[Recipe How-to](../how-to/pipeline-recipes.md) when the artifact should replay
operations on another input. Saving and loading WDF need the `io` extra
(`pip install "wandas[io]"`).

生成された`save`／`load` docstringが、公開WDF API契約の正本です。具体的なFrame結果を保存する
WDFと、別入力へ処理を再実行するRecipeの使い分けは[Recipe How-to](../how-to/pipeline-recipes.md)
を参照してください。WDFの保存と読み込みには`io` extra（`pip install "wandas[io]"`）が必要です。
`wandas/io/`の実装不変条件は[I/O Contracts](../contributing/io-contracts.md)で管理します。

::: wandas.io.wdf_io.save

::: wandas.io.wdf_io.load
