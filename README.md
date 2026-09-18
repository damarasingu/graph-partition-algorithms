# Graph Partition Algorithms

An inference-guided design project for partitioning a weighted path into contiguous regions. It combines an exact dynamic-programming optimizer with a small Bayesian/Markov boundary model that turns local discontinuities into soft cut penalties.

## Why this is resume-worthy

- Designed a dependency-free dynamic program with `O(k n^2)` time and `O(k n)` recoverable state for contiguous path partitioning.
- Integrated Bayesian evidence and a first-order Markov smoothness prior to guide boundary selection without sacrificing deterministic optimization.
- Built an exhaustive reference solver, greedy baseline, reproducible generator, and benchmark harness to validate quality and quantify runtime improvements.
- Added tests for optimality, path coverage, inference behavior, and invalid inputs.

## Run it

Requires Python 3.10+.

```powershell
python graph_partitioning.py
python -m unittest -v
python benchmark.py
python visualizer.py
```

The benchmark compares exhaustive search for small paths with the DP solver. Exact speedup depends on the machine and Python version; use the output from your own run when writing a quantified resume bullet.

## Model

For each interior boundary, the model computes a posterior-like cut probability from the local demand discontinuity. The DP objective is:

`load_balance_error + smoothness * (1 - boundary_probability)`

The optimizer still considers every valid contiguous partition, so the inference layer guides the objective while the dynamic program guarantees the best solution for that objective.

## Resume version

> Built an inference-guided graph partitioning prototype in Python, combining Bayesian/Markov boundary scoring with an exact dynamic-programming optimizer; added exhaustive baselines, reproducible benchmarks, and tests to quantify path-partitioning speedups and solution quality.

Replace “quantify path-partitioning speedups” with an exact percentage only after running `python benchmark.py` and recording the result on your machine.

## Interactive visualizer

Run `python visualizer.py`, then open `http://127.0.0.1:8765`. The browser view renders the weighted path, guided DP cuts, greedy baseline cuts, segment loads, and posterior-like boundary probabilities. Edit the weights, number of regions, or inference smoothness and select **Repartition path** to see the model update live.
