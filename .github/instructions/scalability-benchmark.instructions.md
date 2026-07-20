---
description: "Route code or dependency changes that may affect representative scalability measurements"
applyTo: "scripts/scalability_benchmark.py,tests/test_scalability_benchmark.py,docs/src/explanation/scalability-contract.md,wandas/**,pyproject.toml,uv.lock"
---
# Wandas Scalability Benchmark

Use the repo-shared
[`wandas-scalability-benchmark`](../../.agents/skills/wandas-scalability-benchmark/SKILL.md)
skill to decide, run, compare, and record representative scalability evidence. Treat
the benchmark pytest as smoke coverage only.
