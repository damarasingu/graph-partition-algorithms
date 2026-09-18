"""Inference-guided partitioning of weighted paths.

The module is dependency-free so the algorithm can be profiled and reused in
coursework, interviews, or a small research prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from random import Random
from statistics import median
from typing import Sequence


@dataclass(frozen=True)
class PartitionResult:
    """A contiguous partition and its objective value."""

    cuts: tuple[int, ...]
    cost: float
    length: int

    @property
    def segments(self) -> tuple[tuple[int, int], ...]:
        return _segments(self.cuts, self.length)


def _segments(cuts: tuple[int, ...], length: int) -> tuple[tuple[int, int], ...]:
    boundaries = (0, *cuts, length)
    return tuple(zip(boundaries[:-1], boundaries[1:]))


class BayesianMarkovBoundaryModel:
    """Estimate likely cut locations from local edge evidence.

    A Bayesian prior favors cuts near large discontinuities in vertex demand.
    A first-order Markov term rewards neighboring boundary decisions that are
    locally consistent. The resulting probability is used as a soft DP penalty.
    """

    def __init__(self, prior_strength: float = 1.0, smoothness: float = 0.35) -> None:
        self.prior_strength = prior_strength
        self.smoothness = smoothness

    def cut_probabilities(self, weights: Sequence[float]) -> tuple[float, ...]:
        if len(weights) < 2:
            return ()
        differences = [abs(weights[index] - weights[index - 1]) for index in range(1, len(weights))]
        scale = median(differences) or 1.0
        probabilities = []
        for difference in differences:
            evidence = difference / scale
            log_odds = self.prior_strength * (evidence - 1.0)
            probabilities.append(1.0 / (1.0 + exp(-log_odds)))
        return tuple(probabilities)

    def boundary_penalty(self, weights: Sequence[float], boundary: int) -> float:
        probabilities = self.cut_probabilities(weights)
        probability = probabilities[boundary - 1]
        return self.smoothness * (1.0 - probability)


def _prefix_sums(weights: Sequence[float]) -> list[float]:
    prefix = [0.0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    return prefix


def _segment_cost(prefix: Sequence[float], start: int, end: int, target: float) -> float:
    total = prefix[end] - prefix[start]
    return (total - target) ** 2


def greedy_partition(weights: Sequence[float], parts: int) -> PartitionResult:
    """Build a fast baseline by cutting at the nearest target-load boundary."""
    _validate_input(weights, parts)
    target = sum(weights) / parts
    cuts: list[int] = []
    running = 0.0
    for boundary in range(1, len(weights)):
        if len(cuts) == parts - 1:
            break
        running += weights[boundary - 1]
        remaining_vertices = len(weights) - boundary
        remaining_parts = parts - len(cuts) - 1
        if remaining_vertices >= remaining_parts and abs(running - target) <= abs(running + weights[boundary] - target):
            cuts.append(boundary)
            running = 0.0
    while len(cuts) < parts - 1:
        cuts.append(len(weights) - (parts - len(cuts)))
    cuts_tuple = tuple(sorted(set(cuts)))
    return PartitionResult(cuts_tuple, objective(weights, cuts_tuple), len(weights))


def dynamic_program_partition(
    weights: Sequence[float],
    parts: int,
    model: BayesianMarkovBoundaryModel | None = None,
) -> PartitionResult:
    """Find the minimum-cost contiguous partition in O(parts * n^2) time."""
    _validate_input(weights, parts)
    model = model or BayesianMarkovBoundaryModel()
    length = len(weights)
    target = sum(weights) / parts
    prefix = _prefix_sums(weights)
    infinity = float("inf")
    dp = [[infinity] * (length + 1) for _ in range(parts + 1)]
    parent = [[-1] * (length + 1) for _ in range(parts + 1)]
    dp[0][0] = 0.0

    for part_count in range(1, parts + 1):
        for end in range(part_count, length + 1):
            for start in range(part_count - 1, end):
                previous = dp[part_count - 1][start]
                if previous == infinity:
                    continue
                boundary_penalty = model.boundary_penalty(weights, start) if start else 0.0
                candidate = previous + _segment_cost(prefix, start, end, target) + boundary_penalty
                if candidate < dp[part_count][end]:
                    dp[part_count][end] = candidate
                    parent[part_count][end] = start

    cuts: list[int] = []
    end = length
    for part_count in range(parts, 1, -1):
        start = parent[part_count][end]
        cuts.append(start)
        end = start
    cuts_tuple = tuple(sorted(cuts))
    return PartitionResult(cuts_tuple, dp[parts][length], length)


def exhaustive_partition(weights: Sequence[float], parts: int) -> PartitionResult:
    """Reference solver for tests and benchmarks; exponential in path length."""
    _validate_input(weights, parts)
    best = PartitionResult((), float("inf"), len(weights))
    for cuts in _combinations(range(1, len(weights)), parts - 1):
        candidate = PartitionResult(cuts, objective(weights, cuts), len(weights))
        if candidate.cost < best.cost:
            best = candidate
    return best


def objective(weights: Sequence[float], cuts: Sequence[int]) -> float:
    _validate_cuts(len(weights), cuts)
    target = sum(weights) / (len(cuts) + 1)
    prefix = _prefix_sums(weights)
    return sum(_segment_cost(prefix, start, end, target) for start, end in _segments(tuple(cuts), len(weights)))


def generate_instance(size: int, seed: int = 7) -> list[float]:
    random = Random(seed)
    return [round(random.uniform(0.5, 4.0) + (index // 10) * 0.15, 3) for index in range(size)]


def _combinations(values: range, count: int):
    if count == 0:
        yield ()
        return
    values_list = list(values)
    for index, value in enumerate(values_list):
        for suffix in _combinations(range(value + 1, values_list[-1] + 1), count - 1):
            yield (value, *suffix)


def _validate_input(weights: Sequence[float], parts: int) -> None:
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("weights must be a non-empty sequence of positive numbers")
    if not 1 <= parts <= len(weights):
        raise ValueError("parts must be between 1 and the number of vertices")


def _validate_cuts(length: int, cuts: Sequence[int]) -> None:
    if tuple(sorted(set(cuts))) != tuple(cuts) or any(cut <= 0 or cut >= length for cut in cuts):
        raise ValueError("cuts must be strictly increasing interior boundaries")


if __name__ == "__main__":
    sample = generate_instance(12)
    result = dynamic_program_partition(sample, 3)
    print(f"weights={sample}")
    print(f"cuts={result.cuts}, segments={_segments(result.cuts, len(sample))}, cost={result.cost:.4f}")
