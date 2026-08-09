# Issue #438 resident-cache dtype evidence

This archived evidence measures the resident process memory and raw tensor size for
the explicit cache-dtype workflow introduced by Issue #438. The base revision is
`38d4cd29705e2a430119d2941b8d33138c730a0e`; the candidate revision is
`19c13ce656d8334809d473e04fe47d3171815b09`. Both revisions used `uv.lock`
SHA-256 `4ef530af6b76bfa0ad997392615109f16cbe677b374f5ce9c87aacff4f242db4`.

Each scenario materialized a lazy `(2, 8_000_000)` `float64` `ChannelFrame` in an
isolated process. The base and candidate controls called `cache()` directly. The
compact candidate called `astype("float32").cache()`. Three trials were interleaved
in base-control, candidate-control, candidate-compact order.

| Scenario | Cached dtype | Raw bytes | Median current RSS |
| --- | --- | ---: | ---: |
| Base control | `float64` | 128,000,000 | 278,294,528 |
| Candidate control | `float64` | 128,000,000 | 278,659,072 |
| Candidate compact | `float32` | 64,000,000 | 214,630,400 |

The compact raw tensor is exactly 50% of the candidate control. Its median current
RSS is 22.98% lower, while the base and candidate control medians differ by 0.13%.
These process-level measurements are representative evidence, not a general memory
guarantee. In particular, upstream operations may retain or create wider temporary
arrays during materialization, and peak RSS is allocator- and workload-dependent.

## Already-resident source

A second scenario starts from a 128,000,000-byte `float64` NumPy array through
`ChannelFrame.from_numpy`. Frame results retain their immediate receiver through
`previous`, so both variants keep that source reachable after callers delete their
own references:

| Scenario | Retained source | Cached raw tensor | Median current RSS |
| --- | ---: | ---: | ---: |
| Eager-backed control | 128,000,000 | 128,000,000 | 406,376,448 |
| Eager-backed compact | 128,000,000 | 64,000,000 | 342,462,464 |

The compact cache still halves the new cached tensor, but does not release the wider
source. Median process RSS was therefore 15.73% lower rather than 22.98% lower. The
overall resident saving depends on whether upstream Frame data is virtual/lazy or
already resident and retained by the Frame chain.

The complete observations, including environment versions and peak RSS, are in
[`resident-memory.json`](resident-memory.json). The bridge script is
[`benchmark.py`](benchmark.py), SHA-256
`f316888d8c26d410c138fa261ef80f15e5601e7d7a972f2461fdd2270267d120`.
Measurements used the command template recorded in the JSON, with the current
directory set to the corresponding clean detached worktree. To reproduce the three
scenarios from a checkout containing the candidate commit:

```bash
repo_root=$PWD
benchmark_script=$repo_root/docs/src/assets/benchmarks/issue-438/benchmark.py

git worktree add --detach /tmp/wandas-issue438-base 38d4cd29705e2a430119d2941b8d33138c730a0e
cd /tmp/wandas-issue438-base
uv run --locked python "$benchmark_script" --mode control --channels 2 --samples 8000000

cd "$repo_root"
git worktree add --detach /tmp/wandas-issue438-candidate 19c13ce656d8334809d473e04fe47d3171815b09
cd /tmp/wandas-issue438-candidate
uv run --locked python "$benchmark_script" --mode control --channels 2 --samples 8000000
uv run --locked python "$benchmark_script" --mode compact --channels 2 --samples 8000000
```

The environment was Linux x86-64, Python 3.10.20, NumPy 2.2.6, and Dask 2025.11.0.

The eager-backed observations are in
[`resident-source-memory.json`](resident-source-memory.json). Its bridge script is
[`benchmark_resident_source.py`](benchmark_resident_source.py), SHA-256
`d15f5351269dbae9210a701360533185de7ac824f578f2138ad5f20805dfa51f`, and uses the
same environment, lock, shape, and three interleaved trials at candidate revision
`21cc9fba428acbcd71d6afdf3cff6eed0e027ea3`.
