"""Statistical estimates of the base model outputs (scheme 10/10).

python -m src.experiments.estimates [--T 480 2400] [--runs 1600]

Terminating simulation from empty queues, horizon T. For every
output the sample over p independent runs gives the mean, the
standard deviation, the 95 % Student confidence interval of the
mean and the smallest and largest value over the runs. Seeds 1..p
are the ones of the 10/10 level in anova.py, so the means coincide
with results/anova_T*.csv.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

from src.experiments.anova import RESPONSES
from src.experiments.common import replicate_parallel

RESULTS = Path("results")
BETA = 0.95


def estimate(values, beta=BETA):
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    t_crit = stats.t.ppf((1 + beta) / 2, n - 1)
    half = float(t_crit * std / math.sqrt(n))
    return {"mean": mean, "std": std, "ci": half,
            "ci_low": mean - half,
            "ci_high": mean + half, "min": float(x.min()),
            "max": float(x.max())}


def run(T, runs, seed=1):
    data = replicate_parallel(runs, T, seed=seed, k=10,
                              period=10.0)
    rows = []
    for key, label in RESPONSES:
        row = {"T": T, "runs": runs, "response": key,
               "label": label}
        row.update(estimate([d[key] for d in data]))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0])
    ap.add_argument("--T", type=float, nargs="+",
                    default=[480.0, 2400.0])
    ap.add_argument("--runs", type=int, default=1600)
    args = ap.parse_args()
    rows = []
    for T in args.T:
        rows += run(T, args.runs)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "base_estimates.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"T = {r['T']:g}  {r['label']:<48} "
              f"{r['mean']:9.4f} ± {r['ci']:.4f}  "
              f"σ = {r['std']:.4f}  [{r['min']:g}; {r['max']:g}]")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
