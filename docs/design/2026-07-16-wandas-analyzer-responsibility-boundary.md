# Wandas / Analyzer responsibility boundary

- Status: Accepted
- Date: 2026-07-16
- Applies from: Wandas 0.6

## Context

Wandas is a context-preserving signal-analysis library. Its notebook plots and
`describe()` view are part of the product: they let a user inspect a Frame before and
after analysis without discarding sampling rate, units, channel metadata, or lineage.
The separate Analyzer product is an interactive review application. Without a written
boundary, GUI session state and evidence-report features could gradually enter Frame
APIs and couple the library to one review interface.

## Decision

Wandas owns:

- immutable xarray/Dask-backed signal Frames;
- sampling rate, channels, units, references, metadata, and semantic lineage;
- lazy processing, typed analysis results, Recipe plans, and I/O;
- calibration semantics and physical-unit conversion;
- notebook-friendly static `plot()` and `describe()` output, including Figure saving.

Analyzer owns:

- interactive review UI in VS Code;
- playback, loops, mute, offsets, zoom/pan, and synchronized cursors;
- multi-track comparison sessions, selections, annotations, comments, and review state;
- GUI-oriented previews and PNG/CSV/Markdown evidence-report workflows;
- PR, Issue, and review-system integration.

The feature test is:

1. Is the feature required to preserve or transform the analytical meaning of waveform
   data? If yes, it belongs in Wandas.
2. Is it state or presentation used by a person to explore, compare, or explain results
   in an interactive GUI? If yes, it belongs in Analyzer.

APIs such as `preview_waveform()`, `review_artifact()`, `comparison_session()`,
`export_markdown_report()`, `timeline_state`, and `loop_range` therefore do not belong
in Wandas.

## Consequences

`ChannelFrame.describe()` remains public but delegates presentation work to
`wandas.visualization.describe` and optional notebook display work to
`wandas.visualization.notebook`. Frame classes retain orchestration and metadata
contracts, not IPython/Matplotlib lifecycle details.

Analyzer may consume public Frame, Recipe JSON, WDF, and Figure contracts. Wandas does
not add Analyzer-specific mutable state or a report/evidence compatibility layer.

## 日本語要約

Wandas は波形の解析上の意味、遅延処理、Recipe、I/O、校正、Notebook 上の静的可視化を担当します。Analyzer は再生、同期比較、選択・注釈、レビュー状態、GUI 向け証跡出力を担当します。解析意味論に必要なら Wandas、人が GUI で探索・比較・説明する状態なら Analyzer、という判定基準を採用します。
