---
name: wandas-scalability-benchmark
description: Use when deciding, running, comparing, or recording representative Wandas scalability evidence for changes affecting WDF save/load, whole-Frame materialization, Dask chunking or graph task counts, AudioOperation.process, benchmark semantics, or Dask/xarray/HDF5 dependencies.
---

# Wandas Scalability Benchmark

Treat [`AGENTS.md`](../../../AGENTS.md) as the repository source of truth. Use the
repository benchmark for manual before/after evidence supporting the
[scalability contract](../../../docs/src/explanation/scalability-contract.md); treat
[`test_scalability_benchmark.py`](../../../tests/test_scalability_benchmark.py) as
schema and smoke coverage only.

## Decide whether representative evidence is required

1. Identify whether the change can affect a metric emitted by
   [`scalability_benchmark.py`](../../../scripts/scalability_benchmark.py) or a measured
   materialization boundary.
2. Require representative base/candidate evidence for an affected path.
3. Otherwise record why the changed paths cannot affect the benchmark.

## Prepare comparable revisions

- Record the base and candidate commit SHAs before either run.
- Use the same machine, operating system, Python version, and dependency lock.
- When dependencies intentionally change, record both lock identities and treat their
  effect as part of the result.
- Use separate clean worktrees when switching revisions would disturb active work.
- If the base has an incompatible benchmark schema, use a bridge command accepted by
  both revisions and identify non-comparable fields. If it has no benchmark, state that
  no before comparison exists.

## Run schema version 2

Run the representative matrix, which isolates every sample-count/chunk-size pair:

```bash
uv run --locked --no-dev --extra io python scripts/scalability_benchmark.py --channels 2 --samples 480000 4800000 --chunk-samples 48000 480000 --sampling-rate 48000
```

Use a small matrix only for smoke or debugging:

```bash
uv run --locked --no-dev --extra io python scripts/scalability_benchmark.py --channels 2 --samples 8000 --chunk-samples 1000 4000 --sampling-rate 48000
```

Compare fixed data size across chunk sizes and fixed chunk size across data sizes.
Compare task counts and WDF file sizes exactly when their definitions are unchanged.
Compare timings and absolute worker-lifetime RSS only within the same environment.
Rerun a material timing or RSS difference to distinguish it from ordinary noise. Do
not invent a repository-wide pass/fail budget.

## Record the evidence

Record base and candidate SHAs, exact commands, environment, lock identity, raw JSON,
reruns, non-comparable fields, and a concise conclusion. When skipped, record the
specific reason. Do not commit transient benchmark JSON unless it is an intentional
fixture.
