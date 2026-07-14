# Recipe v2 schema-2 test migration record

## Public-contract migration

Tests for descriptor subclasses, call families, codecs, Dask graph marker extraction,
`_source_lineage`, `_lineage_or_source()`, operation-summary snapshots, and private
index-lineage builders are removed with those contracts. Their behavior moves to tests
through public Frame operations and the four public Recipe entry points.

The indexing matrix exercises every supported selector through `frame[key]`, then
checks data, metadata, source-time offsets, one semantic node/history record, schema-2
roundtrip, and applied output. Multidimensional selection is one atomic public
operation. Stepped or reversed non-channel axes and unsupported advanced indexing fail
at the public boundary. `get_channel` query tests preserve label and JSON-literal
intent and prove that regex/callable queries fail Recipe extraction.

## Required responsibilities

| Responsibility | Required public evidence |
| --- | --- |
| canonical graph | shared lineage identity, deterministic topological node order, one edge owner |
| validation | missing/extra/wrong-kind inputs, IDs, topology, arity, versions, binding/output kinds, dead nodes, terminal placement |
| persistence | schema 2 roundtrip, schema 1 rejection, unknown fields/tags/operations/versions rejected |
| values | NumPy scalar dtype fidelity, special floats/complex, mutation snapshots, unsupported object rejection |
| binary arrays | NumPy and Dask broadcasting/laziness, reflected order, no container detail in history/schema |
| Frame arithmetic | equal semantic shape/rate, left metadata and source offset, no timeline alignment |
| mixing | strict/pad/truncate direction, mono broadcast, SNR, offset-agnostic index mixing |
| channel addition | raw 1-D and `(1, n)`, multi-channel Frame, raw multi-channel rejection |
| WDF | one `operation_history` display prefix, defensive strict JSON, no executable Recipe |
| semantic atomicity | every public call creates one authoritative node, including external `BaseFrame` subclasses |
| failure classes | extraction, serialization/validation, and execution errors include node/operation context |

Mutation probes modify NumPy operands, source-time-offset arrays, mappings, and lists
after a public call. History, `RecipePlan`, and its serialized form must remain
unchanged. Dask compute bombs cover both `RecipePlan.from_frame` and public lazy graph
construction.

Extension probes are not descriptor unit tests. Each test-only extension runs the full
path:

```text
public Frame operation
  -> semantic lineage
  -> RecipePlan.from_frame(registry=...)
  -> to_dict()
  -> from_dict(registry=...)
  -> apply(registry=...)
```

The required probes are a unary audio operation, a typed Frame transition, and a true
multi-Frame operation. None may require changes to the model, compiler, validator,
executor, or serializer.

## Acceptance gates

Correctness and explicit contracts take priority over a strict historical PLOC target.
Production Python must not grow overall, extraction/build must perform zero Dask
computes, the schema and public Recipe surface above must be stable within schema 2,
and the full requested test/lint/type/docs/audit suite must pass. Benchmark timing and
memory are recorded only; they do not trigger implementation changes.

Final code receives five independent review rounds in each of four perspectives:
readability, maintainability/extensibility, testability/state isolation, and duplicated
state/dual entry points/reduction. A perspective passes when no reproducible actionable
High or Medium finding remains. Low findings are changed only when they plainly remove
duplication or ambiguity without adding state, branches, abstraction, compatibility
layers, UX regressions, eagerness, or implementation-coupled tests.
