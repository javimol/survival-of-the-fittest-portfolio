"""Compare the 2006 and 2026 mutation operators against the exact optimum.

Benchmark: Hang Seng (OR-Library port1, 31 stocks), cardinality k = 3, the same
grid of target returns as the 2006 runs (min(mu) to max(mu), step 1e-4).

For k = 3 the inner QP has a closed-form solution: two equality constraints
leave a one-dimensional segment of feasible weights, so the minimum variance
is a clipped 1-D quadratic. That makes it cheap to score all C(31, 3) = 4,495
subsets at every target return, which gives

  * the exact cardinality-constrained frontier, and
  * a lookup table the GA uses as its fitness function (identical to calling
    the QP, only faster).

The GA is a Python port of the MATLAB setup in src/solve_ga.m, mirroring the
behaviour of MATLAB's ga: population 2N, 2 elite, crossover fraction 0.8, rank
fitness scaling, stochastic uniform selection, R3 crossover, at most 100
generations, stop after 10 generations without improvement. The two variants
differ ONLY in the mutation operator:

  2006  remove a random held stock, insert the lowest-numbered stock not held
  2026  remove a random held stock, insert a random stock not held

Outputs:
  results/mutation_experiment_port1_k3.csv   per-target metrics
  docs/img/mutation_experiment.png           chart
  summary printed to stdout
"""
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "orlib" / "port1.txt"
RESULTS_2006 = ROOT / "results" / "2006" / "port1_frontier_k3.csv"
OUT_CSV = ROOT / "results" / "mutation_experiment_port1_k3.csv"
OUT_IMG = ROOT / "docs" / "img" / "mutation_experiment.png"

K = 3
STEP = 1e-4
RUNS = 30            # independent GA runs per target return and variant
SEED = 2026
MAX_GENERATIONS = 100
STALL_GENERATIONS = 10
ELITE = 2
CROSSOVER_FRACTION = 0.8
TOL = 1e-6           # relative improvement below which a generation "stalls"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#a3a29c"
INK = "#0b0b0b"
INK_2 = "#52514e"


# --------------------------------------------------------------------------- data

def load_orlib(path):
    with open(path) as fh:
        tokens = fh.read().split()
    n = int(tokens[0])
    stats = np.array(tokens[1:1 + 2 * n], dtype=float).reshape(n, 2)
    pairs = np.array(tokens[1 + 2 * n:], dtype=float).reshape(-1, 3)
    mu, sd = stats[:, 0], stats[:, 1]
    rho = np.zeros((n, n))
    i, j = pairs[:, 0].astype(int) - 1, pairs[:, 1].astype(int) - 1
    rho[i, j] = pairs[:, 2]
    rho[j, i] = pairs[:, 2]
    return mu, rho * np.outer(sd, sd)


# ------------------------------------------------------------- exact k = 3 QP

def min_variance_all_subsets(mu, sigma, subsets, target):
    """Minimum variance for every 3-asset subset at one target return.

    min w'Sw  s.t.  mu_s'w = target, 1'w = 1, w >= 0   (w <= 1 is implied)
    Returns an array of variances, Inf where the target is unreachable.
    """
    m = mu[subsets]                                   # (M, 3)
    s = sigma[subsets[:, :, None], subsets[:, None, :]]  # (M, 3, 3)
    ones = np.ones_like(m)

    # Feasible line w = w0 + t d: d spans the null space of [m; 1].
    d = np.cross(m, ones)
    # Least-norm particular solution w0 = A' (A A')^-1 b.
    a11 = (m * m).sum(1)
    a12 = m.sum(1)
    a22 = np.full(len(m), 3.0)
    det = a11 * a22 - a12 ** 2
    degenerate = np.abs(det) < 1e-18
    det = np.where(degenerate, 1.0, det)
    y1 = (a22 * target - a12 * 1.0) / det
    y2 = (-a12 * target + a11 * 1.0) / det
    w0 = y1[:, None] * m + y2[:, None] * ones

    # w >= 0 bounds the line parameter t to [lo, hi].
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = -w0 / d
    lo = np.where(d > 0, bound, -np.inf).max(1)
    hi = np.where(d < 0, bound, np.inf).min(1)
    blocked = ((d == 0) & (w0 < -1e-12)).any(1)

    sd = np.einsum("mij,mj->mi", s, d)
    qa = (d * sd).sum(1)
    qb = (w0 * sd).sum(1)
    qc = np.einsum("mi,mij,mj->m", w0, s, w0)
    t = np.clip(-qb / qa, lo, hi)
    var = qa * t ** 2 + 2 * qb * t + qc

    infeasible = degenerate | blocked | (lo > hi + 1e-12)
    return np.where(infeasible, np.inf, var)


def fitness_table(mu, sigma, subsets, target, n):
    """Dense n x n x n lookup of subset variance (sorted index triples)."""
    var = min_variance_all_subsets(mu, sigma, subsets, target)
    table = np.full((n, n, n), np.inf)
    table[subsets[:, 0], subsets[:, 1], subsets[:, 2]] = var
    return table, var.min()


# ------------------------------------------------------------------------- GA

def score(pop, table):
    p = np.sort(pop, axis=1)
    return table[p[:, 0], p[:, 1], p[:, 2]]


def rank_scaling(scores, n_parents):
    """MATLAB fitscalingrank: expectation proportional to 1/sqrt(rank)."""
    order = np.argsort(scores, kind="stable")
    expectation = np.empty(len(scores))
    expectation[order] = 1.0 / np.sqrt(np.arange(1, len(scores) + 1))
    return n_parents * expectation / expectation.sum()


def stochastic_uniform(expectation, n_parents, rng):
    """MATLAB selectionstochunif: one spin, equally spaced pointers."""
    wheel = np.cumsum(expectation) / n_parents
    positions = rng.random() / n_parents + np.arange(n_parents) / n_parents
    return np.minimum(np.searchsorted(wheel, positions, side="right"), len(wheel) - 1)


def crossover_r3(parents, pop, k, rng):
    """Same logic as src/ga_crossover_r3.m."""
    pool = list(parents)
    kids = []
    while len(pool) >= 2:
        p1 = pop[pool[0]]
        j = 1
        while j < len(pool) - 1 and set(pop[pool[j]]) == set(p1):
            j += 1
        p2 = pop[pool[j]]
        pool[1], pool[j] = pool[j], pool[1]
        del pool[:2]
        common = np.intersect1d(p1, p2)
        others = np.setxor1d(p1, p2)
        fill = rng.choice(others, k - len(common), replace=False)
        kids.append(np.concatenate([common, fill]))
    return np.array(kids, dtype=int).reshape(-1, k)


def mutate(parents, pop, n, rng, variant):
    kids = pop[parents].copy()
    k = kids.shape[1]
    for kid in kids:
        outside = np.setdiff1d(np.arange(n), kid)     # sorted, like MATLAB setdiff
        new = outside[0] if variant == "2006" else rng.choice(outside)
        kid[rng.integers(k)] = new
    return kids


def run_ga(table, n, k, variant, rng):
    pop_size = 2 * n
    n_xover = round(CROSSOVER_FRACTION * (pop_size - ELITE))
    n_mutate = pop_size - ELITE - n_xover
    n_parents = 2 * n_xover + n_mutate

    pop = np.array([rng.choice(n, k, replace=False) for _ in range(pop_size)])
    scores = score(pop, table)
    best_history = [scores.min()]

    for generation in range(1, MAX_GENERATIONS + 1):
        expectation = rank_scaling(scores, n_parents)
        parents = rng.permutation(stochastic_uniform(expectation, n_parents, rng))
        elite = pop[np.argsort(scores, kind="stable")[:ELITE]]
        xover = crossover_r3(parents[:2 * n_xover], pop, k, rng)
        mutants = mutate(parents[2 * n_xover:], pop, n, rng, variant)
        pop = np.vstack([elite, xover, mutants])
        scores = score(pop, table)
        best_history.append(scores.min())

        if generation >= STALL_GENERATIONS:
            old, new = best_history[-1 - STALL_GENERATIONS], best_history[-1]
            if old == new or (np.isfinite(new) and (old - new) <= TOL * abs(new)):
                break
    return scores.min(), generation


# ---------------------------------------------------------------- experiment

def main():
    mu, sigma = load_orlib(DATA)
    n = len(mu)
    subsets = np.array(list(combinations(range(n), K)))
    targets = np.arange(mu.min(), mu.max() + 1e-12, STEP)
    rng = np.random.default_rng(SEED)

    rows = []
    best = {"2006": [], "2026": []}
    single = {"2006": [], "2026": []}
    exact = []
    for target in targets:
        table, v_star = fitness_table(mu, sigma, subsets, target, n)
        exact.append(v_star)
        row = {"target": target, "exact_variance": v_star}
        for variant in ("2006", "2026"):
            results = [run_ga(table, n, K, variant, rng) for _ in range(RUNS)]
            v = np.array([r[0] for r in results])
            gens = np.array([r[1] for r in results])
            hit = np.isclose(v, v_star, rtol=1e-9, atol=0)
            excess = np.where(np.isfinite(v), (np.sqrt(v / v_star) - 1) * 100, np.nan)
            row[f"hit_rate_{variant}"] = hit.mean()
            row[f"mean_excess_vol_pct_{variant}"] = np.nanmean(excess)
            row[f"max_excess_vol_pct_{variant}"] = np.nanmax(excess) if np.isfinite(excess).any() else np.nan
            row[f"failed_{variant}"] = int((~np.isfinite(v)).sum())
            row[f"mean_generations_{variant}"] = gens.mean()
            single[variant].append(v[0])
            best[variant].append(excess)
        rows.append(row)
        print(f"target {target:.6f}  hit 2006 {row['hit_rate_2006']:.2f}  "
              f"hit 2026 {row['hit_rate_2026']:.2f}", flush=True)

    exact = np.array(exact)
    keys = list(rows[0])
    OUT_CSV.write_text(",".join(keys) + "\n" + "\n".join(
        ",".join(f"{r[key]:.10g}" for key in keys) for r in rows) + "\n")

    # The original 2006 run (one GA run per target), scored against the exact optimum.
    old = np.loadtxt(RESULTS_2006, delimiter=",", ndmin=2)
    old_exact = np.array([fitness_table(mu, sigma, subsets, t, n)[1] for t in old[:, 0]])
    ok = np.isfinite(old[:, 1]) & (old[:, 1] > 0)
    old_excess = (np.sqrt(old[ok, 1] / old_exact[ok]) - 1) * 100
    old_hit = np.isclose(old[ok, 1], old_exact[ok], rtol=1e-4)  # 2006 files keep 5 digits

    print("\nSUMMARY  Hang Seng, k = 3, {} target returns, {} runs each".format(len(targets), RUNS))
    for variant in ("2006", "2026"):
        hit = np.mean([r[f"hit_rate_{variant}"] for r in rows])
        all_excess = np.concatenate(best[variant])
        gens = np.mean([r[f"mean_generations_{variant}"] for r in rows])
        failed = sum(r[f"failed_{variant}"] for r in rows)
        print(f"  {variant} mutation: optimum found in {hit:.1%} of runs | "
              f"excess volatility mean {np.nanmean(all_excess):.2f}%  "
              f"median {np.nanmedian(all_excess):.2f}%  "
              f"95th pct {np.nanpercentile(all_excess, 95):.2f}%  "
              f"max {np.nanmax(all_excess):.2f}% | generations {gens:.1f} | "
              f"no feasible portfolio {failed}")
    print(f"  Original 2006 run: optimum found at {old_hit.mean():.1%} of {ok.sum()} points | "
          f"excess volatility mean {old_excess.mean():.2f}%  max {old_excess.max():.2f}%  "
          f"| points below exact beyond 5-digit rounding: {(old_excess < -0.01).sum()}")

    plot(targets, exact, single, rows)


def plot(targets, exact, single, rows):
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.2),
                                      gridspec_kw={"width_ratios": [1.15, 1]})
    ret = targets * 100
    i_min = np.argmin(exact)
    eff = slice(i_min, None)

    for variant, color in (("2006", ORANGE), ("2026", BLUE)):
        v = np.array(single[variant])
        left.plot(np.sqrt(v[eff]) * 100, ret[eff], color=color, linewidth=1.4,
                  label=f"One GA run per target, {variant} mutation")
    left.plot(np.sqrt(exact[eff]) * 100, ret[eff], color=INK, linewidth=2.2,
              label="Exact optimum (all 4,495 subsets)", zorder=5)
    left.set_xlabel("Risk: volatility of weekly return (%)")
    left.set_ylabel("Expected weekly return (%)")
    left.set_title("Efficient frontier: single runs vs the exact optimum",
                   loc="left", color=INK, fontsize=11)
    left.grid(alpha=0.25)
    left.legend(frameon=False, loc="lower right", fontsize=9)

    for variant, color in (("2006", ORANGE), ("2026", BLUE)):
        hit = np.array([r[f"hit_rate_{variant}"] for r in rows]) * 100
        right.plot(ret, hit, color=color, linewidth=2, label=f"{variant} mutation")
    right.set_ylim(0, 102)
    right.set_xlabel("Target weekly return (%)")
    right.set_ylabel(f"Runs that found the exact optimum (%, of {RUNS})")
    right.set_title("How often each version finds the best 3 stocks",
                    loc="left", color=INK, fontsize=11)
    right.grid(alpha=0.25)
    right.legend(frameon=False, loc="lower left", fontsize=9)

    fig.text(0.01, 0.98, "Same GA, two mutation operators - Hang Seng, 3 stocks",
             fontsize=13, fontweight="bold", color=INK, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_IMG, dpi=130)
    print(f"wrote {OUT_IMG}")


if __name__ == "__main__":
    main()
