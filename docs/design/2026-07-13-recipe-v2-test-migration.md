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
details or forcing Dask computation.

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

`scripts/recipe_v2_benchmark.py` ran 100 iterations in detached baseline and v2
worktrees on the same environment. Lower is better.

| metric | v1 | v2 | v2 index |
| --- | ---: | ---: | ---: |
| extraction p95 | 300.8 µs | 212.0 µs | 70.5 |
| lazy graph-build p95 | 20,306.6 µs | 14,089.3 µs | 69.4 |
| traced peak memory | 811,977 B | 763,825 B | 94.1 |

This is a reproducible microbenchmark, not a claim about end-to-end numerical compute.
Neither measured path calls Dask `compute()`.
