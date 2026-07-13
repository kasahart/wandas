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
