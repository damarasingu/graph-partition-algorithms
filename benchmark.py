"""Benchmark the exact DP solver against exhaustive search."""

from __future__ import annotations

from time import perf_counter

from graph_partitioning import (
    BayesianMarkovBoundaryModel,
    dynamic_program_partition,
    exhaustive_partition,
    generate_instance,
    greedy_partition,
)


def timed(function, *args):
    start = perf_counter()
    result = function(*args)
    return result, perf_counter() - start


def main() -> None:
    print("Graph Partition Algorithms | inference-guided benchmark")
    print("size parts | exhaustive ms | DP ms | speedup | greedy cost | DP cost | guided cost")
    for size, parts in ((10, 3), (14, 4), (18, 4), (40, 5)):
        weights = generate_instance(size, seed=size)
        if size <= 18:
            exhaustive, exhaustive_time = timed(exhaustive_partition, weights, parts)
        else:
            exhaustive, exhaustive_time = None, None
        greedy, greedy_time = timed(greedy_partition, weights, parts)
        optimized, optimized_time = timed(
            dynamic_program_partition, weights, parts, BayesianMarkovBoundaryModel(smoothness=0.0)
        )
        guided = dynamic_program_partition(weights, parts, BayesianMarkovBoundaryModel())
        speedup = exhaustive_time / optimized_time if exhaustive_time else None
        exhaustive_text = f"{exhaustive_time * 1000:12.3f}" if exhaustive_time else "        n/a"
        speedup_text = f"{speedup:7.2f}x" if speedup else "      n/a"
        print(
            f"{size:4d} {parts:5d} | {exhaustive_text} | {optimized_time * 1000:6.3f} "
            f"| {speedup_text} | {greedy.cost:11.4f} | {optimized.cost:8.4f} | {guided.cost:11.4f}"
        )
        if exhaustive and abs(exhaustive.cost - optimized.cost) > 1e-9:
            raise AssertionError("DP result diverged from exhaustive optimum")
        if greedy_time < 0:
            raise AssertionError("unreachable timing guard")


if __name__ == "__main__":
    main()
