"""Dependency-free Recipe extraction and lazy-build microbenchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from typing import Any

import numpy as np

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan

MIN_ITERATIONS = 2


def p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < MIN_ITERATIONS:
        parser.error(f"--iterations must be at least {MIN_ITERATIONS}")
    source = ChannelFrame.from_numpy(np.ones((2, 4096)), sampling_rate=16000)
    processed = source.normalize().remove_dc()

    def extract() -> RecipePlan:
        return RecipePlan.from_frame(processed)

    def build(recipe: RecipePlan) -> Any:
        return recipe.apply({"input_0": source})

    extraction: list[float] = []
    graph_build: list[float] = []
    tracemalloc.start()
    for _ in range(args.iterations):
        start = time.perf_counter_ns()
        recipe = extract()
        middle = time.perf_counter_ns()
        result = build(recipe)
        end = time.perf_counter_ns()
        assert result.shape == source.shape
        extraction.append((middle - start) / 1_000)
        graph_build.append((end - middle) / 1_000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        json.dumps(
            {
                "api": "v2",
                "iterations": args.iterations,
                "extraction_p95_us": p95(extraction),
                "graph_build_p95_us": p95(graph_build),
                "peak_bytes": peak,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
