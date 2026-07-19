# I/O Contracts / I/O 契約

Use this reference when changing `wandas/io/` or code that reads and writes
WAV, WDF, CSV, or sample data.
`wandas/io/`、WAV、WDF、CSV、sample data の読み書きを変更する際に、この reference を
使用してください。

## Ownership / 責務

I/O code converts external representations into Frames and back. Keep readers
and writers thin; signal processing belongs in `wandas/processing`, while Frame
construction owns axes, metadata, and domain state.
I/O code は external representation と Frame を相互変換します。reader／writer は薄く保ち、
signal processing は `wandas/processing`、axis、metadata、domain state は Frame construction が
所有します。

## Invariants / 不変条件

- Preserve sampling rate, channel layout and labels, and time or frequency axes
  whenever the format can represent them.
  format が表現できる場合は sampling rate、channel layout／label、time／frequency axis を維持します。
- Preserve user and recording metadata through formats that promise metadata
  round-trips, especially WDF.
  metadata round-trip を保証する format、特に WDF では user／recording metadata を維持します。
- Do not persist runtime lineage, `operation_history`, or operation graphs unless
  a format contract explicitly adds that behavior.
  format contract が明示的に追加しない限り、runtime lineage、`operation_history`、operation graph は
  永続化しません。
- For lossy or metadata-limited formats such as WAV and CSV, define and test what
  is retained, reconstructed, normalized, or rejected.
  WAV や CSV のような lossy／metadata 制限 format では、保持、再構築、normalize、reject の契約を
  定義して test します。
- Preserve Dask laziness at the public read boundary unless the format API
  explicitly requires eager materialization.
  format API が eager materialization を明示的に要求しない限り、public read boundary で Dask
  laziness を維持します。

Validate changes with focused round-trip, metadata, dtype or normalization,
error, and laziness tests appropriate to the format. Use the
[`wandas-test-authoring` Skill](https://github.com/kasahart/wandas/blob/main/.agents/skills/wandas-test-authoring/SKILL.md)
and its detailed [I/O test reference](https://github.com/kasahart/wandas/blob/main/.agents/skills/wandas-test-authoring/references/io.md).
変更は format に応じた round-trip、metadata、dtype／normalization、error、laziness の focused test
で検証します。詳細方針は `wandas-test-authoring` Skill と I/O test reference を参照してください。
