---
description: "Route scalability-sensitive Wandas changes to representative benchmark validation"
applyTo: "scripts/scalability_benchmark.py,tests/test_scalability_benchmark.py,docs/src/explanation/scalability-contract.md,wandas/core/base_frame.py,wandas/io/wdf_frames.py,wandas/io/wdf_io.py,wandas/processing/base.py,wandas/processing/effects.py,wandas/processing/semantic.py,wandas/frames/channel.py,wandas/frames/mixins/channel_processing_mixin.py,wandas/pipeline/decorators.py,wandas/pipeline/model.py,pyproject.toml,uv.lock"
---
# Wandas Scalability Benchmark

Use the repo-shared
[`wandas-scalability-benchmark`](../../.agents/skills/wandas-scalability-benchmark/SKILL.md)
skill to decide, run, compare, and record representative scalability evidence. Treat
the benchmark pytest as smoke coverage only.
