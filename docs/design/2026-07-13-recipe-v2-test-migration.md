# Recipe v2 test migration record

The v1 monolithic test file mixed public behavior with assertions about deleted
private graph reconstruction helpers. Recipe v2 migrates contracts by responsibility:

| v1 concern | v2 evidence |
| --- | --- |
| numerical, class, labels, rate, shape, metadata, source time | `test_recipe_execution.py`, characterization tests, full frame suite |
| linear, scalar, reflected, frame-frame, external Dask | `test_recipe_compiler.py` |
| shared branch identity and topological graph | compiler and contract tests |
| missing input, dead nodes, kinds, arity, terminal placement | `test_recipe_contract.py`, error tests |
| schema/value/call/version/callable rejection | `test_recipe_serialization.py` |
| registry freeze and descriptor/binding agreement | `test_recipe_codecs.py` |
| unary, typed transition, true multi-input extension | `test_recipe_extensions.py` |
| sklearn adapter | `test_sklearn_adapter.py` |

Assertions whose subject was a deleted v1 type, step edge, compatibility schema,
step/call conversion, or dictionary graph reconstruction were intentionally removed;
their behavioral outcomes are covered through the public `RecipePlan` contract. Public
contract tests import no private production symbols. White-box codec tests remain next
to the production responsibility they exercise.

The private indexing lineage builders `_lineage_with_index` and
`_lineage_with_unsupported_indexing` are deleted. Their former white-box assertions
move to public `frame[key]` contract tests that verify the result lineage, history,
metadata, source-time offset, persistence roundtrip, and replay result. Every supported
indexing form, including multidimensional indexing, must produce exactly one semantic
node and one history entry for one public call.

Mutation tests own the snapshot boundary: changing an operand, add-channel parameters,
or source-time-offset input after the public call must not change history, operation
summaries, or a subsequently compiled plan. Semantic atomicity tests own the invariant
that the returned Frame carries the semantic lineage selected at public entry, including
for external `BaseFrame` subclasses. External-input tests own the complementary
contract that NumPy and Dask values compile as named inputs without embedding container
details or forcing Dask computation. Codec contract tests reject missing or substituted
Frame parents before compilation can reinterpret them as external inputs, and prove
that operation identity and scalar values have one descriptor source of truth.
Raw-array add tests require the same operation object in semantic lineage and the Dask
graph.

`uv run python scripts/recipe_v2_test_audit.py` emits all 192 baseline cases with a
`migrated` or `removed_contract` disposition, rationale, and an AST-verified current
pytest function for every migrated row. Migration entries are an explicit curated map,
with an explicit per-test map and no inferred keyword routing. The current audit
records 30 exactly retained cases and 162 contracts intentionally removed with the
destructive v1 API replacement. Retained entries include laziness, typed transition
chains, operand order, external arrays/operators, add-channel, indexing, custom
callable safety, terminal RMS, and missing-input validation.

## Extension amplification

| extension | future production modules | central modules changed |
| --- | ---: | ---: |
| unary same-frame opt-in operation | 1 | 0 |
| versioned typed frame transition | 1 | 0 |
| true multi-input operation plus stable handler | 2 | 0 |

The median is 1 and maximum is 2. The model, compiler, validator, executor,
serializer dispatch, and central allowlists change zero times for all three probes.

## Performance sample

`scripts/recipe_v2_benchmark.py` ran three 100-iteration samples in detached
baseline and v2 worktrees on the same environment. The table reports the median
of each sample's p95; lower is better.

| metric | v1 | v2 | v2 index |
| --- | ---: | ---: | ---: |
| extraction p95 | 175.8 µs | 237.6 µs | 135.2 |
| lazy graph-build p95 | 14,487.0 µs | 14,583.1 µs | 100.7 |
| traced peak memory | 838,457 B | 817,621 B | 97.5 |

This is a reproducible microbenchmark, not a claim about end-to-end numerical compute.
Neither measured path calls Dask `compute()`.

## Cleanup measurements

The final non-sklearn `wandas.pipeline` modules plus
`wandas/processing/semantic.py` total 1,792 PLOC, 39 lines above the 1,753 reference.
This is not a reason to remove explanatory names, validation, or type safety. Relative
to the pre-cleanup Recipe v2 head, production changes contain 232 additions and 219
deletions, a net increase of 13 lines after restoring NumPy scalar dtype fidelity and
the omitted-end `trim` intent in serialized recipes. This cleanup-only comparison is a
reference breakdown; relative to the v2 base, all production Python is down by 184
lines and satisfies the required non-increase gate.

The extension probes change no central model, compiler, call serializer, or
persistence serializer module and therefore add no central dispatch branch. The
NumPy scalar fidelity fix reuses the existing replay-value representation instead of
adding a serializer tag or schema field. Extraction and lazy graph construction
remain compute-free; the Dask compute-bomb tests and the benchmark both complete
without calling `compute()`.

Final validation completed with 157 pipeline tests, 40 core lineage tests, and
2,221 full-suite tests passing; three unrelated optional tests remained skipped.
