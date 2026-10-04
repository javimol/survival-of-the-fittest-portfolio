"""Plot the k = 3 efficient frontiers found by the genetic algorithm.

Reads results/2026/portN_frontier_k3.csv ([mean return, variance] rows, written
by run_frontiers.py) and the OR-Library data in data/orlib/, and writes:

  docs/img/frontier_hang_seng.png   annotated headline figure (port1)
  docs/img/frontiers_benchmarks.png all four benchmarks side by side

Risk is shown as volatility (standard deviation of weekly return, in %) and
return as expected weekly return (in %), so individual stocks and portfolios
can be compared on the same axes.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "2026"
DATA = ROOT / "data" / "orlib"
IMG = ROOT / "docs" / "img"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#a3a29c"
INK = "#0b0b0b"
INK_2 = "#52514e"

BENCHMARKS = [
    ("port1", "Hang Seng", 31),
    ("port2", "DAX 100", 85),
    ("port3", "FTSE 100", 89),
    ("port4", "S&P 100", 98),
]

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": GREY,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_assets(name):
    """Mean and standard deviation of weekly return per asset, in %."""
    with open(DATA / f"{name}.txt") as fh:
        n = int(fh.readline())
        stats = np.array([fh.readline().split() for _ in range(n)], dtype=float)
    return stats[:, 0] * 100, stats[:, 1] * 100


def load_frontier(name):
    """GA frontier as (volatility %, return %), sorted by return."""
    data = np.loadtxt(RESULTS / f"{name}_frontier_k3.csv", delimiter=",", ndmin=2)
    data = data[np.isfinite(data[:, 1]) & (data[:, 1] > 0)]
    data = data[np.argsort(data[:, 0])]
    return np.sqrt(data[:, 1]) * 100, data[:, 0] * 100


def draw(ax, name, annotate=False):
    asset_ret, asset_vol = load_assets(name)
    vol, ret = load_frontier(name)
    i_min = np.argmin(vol)

    ax.scatter(asset_vol, asset_ret, s=22, color=GREY, alpha=0.8, linewidths=0,
               label="Individual stock", zorder=2)
    # Lower branch: same risk as some point above it, but less return.
    ax.plot(vol[:i_min + 1], ret[:i_min + 1], color=BLUE, alpha=0.35, linewidth=1.5,
            linestyle="--", zorder=3)
    ax.plot(vol[i_min:], ret[i_min:], color=BLUE, linewidth=2, zorder=3,
            label="Best 3-stock portfolio (GA)")
    ax.scatter([vol[i_min]], [ret[i_min]], s=70, color=ORANGE, edgecolor="white",
               linewidth=1.5, zorder=4, label="Lowest-risk portfolio")

    ax.axhline(0, color=GREY, linewidth=0.6, zorder=1)
    ax.grid(alpha=0.25)
    ax.set_xlim(left=0)
    ax.set_xlabel("Risk: volatility of weekly return (%)")
    ax.set_ylabel("Expected weekly return (%)")

    if annotate:
        arrow = dict(arrowstyle="->", color=INK_2, lw=0.8)
        j = np.argmin(np.abs(ret - 0.8))
        ax.annotate("Efficient frontier: for each target\nreturn, the lowest risk the GA found\nwith exactly 3 stocks",
                    xy=(vol[j], ret[j]), xytext=(0.25, 1.0), color=INK, fontsize=9.5,
                    va="center", arrowprops=arrow)
        ax.annotate("Lowest-risk 3-stock portfolio",
                    xy=(vol[i_min], ret[i_min]), xytext=(0.25, 0.42), color=INK, fontsize=9.5,
                    va="center", arrowprops=arrow)
        k = np.argmin(np.abs(ret - 0.15))
        ax.annotate("Inefficient branch: a portfolio\nabove it has the same risk\nand more return",
                    xy=(vol[k], ret[k]), xytext=(0.25, 0.12), color=INK_2, fontsize=9,
                    va="center", arrowprops=arrow)
        s = np.argmin(np.abs(asset_vol - 5.37) + np.abs(asset_ret - 0.175))
        ax.annotate("Grey dots: individual\nstocks, each held alone",
                    xy=(asset_vol[s], asset_ret[s]), xytext=(5.75, 0.12), color=INK_2,
                    fontsize=9, va="center", arrowprops=arrow)


def headline():
    fig, ax = plt.subplots(figsize=(9, 5.8))
    draw(ax, "port1", annotate=True)
    fig.text(0.015, 0.965, "Three well-chosen stocks carry far less risk than any single stock",
             fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.915, "Hang Seng, 31 stocks, weekly returns 1992-1997. Portfolios limited to "
             "3 stocks, chosen by the genetic algorithm.",
             fontsize=9.5, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(IMG / "frontier_hang_seng.png", dpi=150)


def benchmarks():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (name, title, n) in zip(axes, BENCHMARKS):
        draw(ax, name)
        ax.set_title(f"{title} ({n} stocks)", loc="left", fontsize=10.5, color=INK)
    for ax in axes[1:]:
        ax.set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(IMG / "frontiers_benchmarks.png", dpi=130)


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    headline()
    benchmarks()
    print(f"wrote figures to {IMG}")


if __name__ == "__main__":
    main()
