"""Compute k = 3 efficient frontiers with the current GA for port1..port4.

Uses the Python port of the GA in mutation_experiment.py (random mutation,
same settings as src/solve_ga.m): one GA run per target return, target returns
from min(mu) to max(mu) in steps of 1e-4, as in the 2006 runs. Writes
results/2026/portN_frontier_k3.csv with [mean return, variance] rows, the same
format as results/2006/.
"""
from itertools import combinations
from pathlib import Path

import numpy as np

from mutation_experiment import K, STEP, fitness_table, load_orlib, run_ga

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "2026"
SEED = 2026


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for p in range(1, 5):
        mu, sigma = load_orlib(ROOT / "data" / "orlib" / f"port{p}.txt")
        n = len(mu)
        subsets = np.array(list(combinations(range(n), K)))
        rows = []
        for target in np.arange(mu.min(), mu.max() + 1e-12, STEP):
            table, _ = fitness_table(mu, sigma, subsets, target, n)
            variance, _ = run_ga(table, n, K, "2026", rng)
            rows.append((target, variance))
        np.savetxt(OUT / f"port{p}_frontier_k3.csv", rows, delimiter=",", fmt="%.10g")
        print(f"port{p}: {len(rows)} points", flush=True)


if __name__ == "__main__":
    main()
