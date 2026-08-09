# Issue #326 cache timing evidence

This archived evidence compares base `7f52b534a09559d35b459dfebe05db3dc2e429ac`
with candidate `926f1ed961a64c69132995a66374a094c82bb6a4` using `uv.lock` SHA-256
`4ef530af6b76bfa0ad997392615109f16cbe677b374f5ce9c87aacff4f242db4`.
The environment and complete trials are recorded in the [base](base-7f52b534.json)
and [candidate](candidate-926f1ed9.json) JSON files.

The scenario is a 10-second, 48 kHz mono STFT with three `.dB` accesses per trial.
Median milliseconds for base uncached runs were 93.03 and 91.99. Candidate uncached,
cache creation, and cached-access medians were 88.63/35.28/29.27 and
95.97/37.02/32.22. This is representative evidence, not a performance guarantee.

Run the committed bridge script from clean worktrees with the same lock:

```bash
repo_root=$PWD
benchmark_script=$repo_root/docs/src/assets/benchmarks/issue-326/benchmark.py

git worktree add --detach /tmp/wandas-issue326-base 7f52b534a09559d35b459dfebe05db3dc2e429ac
cd /tmp/wandas-issue326-base
uv run --locked python "$benchmark_script" > "$repo_root/docs/src/assets/benchmarks/issue-326/base-7f52b534.json"

cd "$repo_root"
git worktree add --detach /tmp/wandas-issue326-candidate 926f1ed961a64c69132995a66374a094c82bb6a4
cd /tmp/wandas-issue326-candidate
uv run --locked python "$benchmark_script" > "$repo_root/docs/src/assets/benchmarks/issue-326/candidate-926f1ed9.json"
```

The bridge script is [benchmark.py](benchmark.py), SHA-256
`807e853d70618c805137570795bdc5a35cb1d5aad27e88002ed60340d92393d9`.
The base has no `cache()` fields. Later changes only reject masked-array compute
results; the ordinary `np.ndarray` path measured here is unchanged.
