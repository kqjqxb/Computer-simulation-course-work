"""One-factor analysis of variance: scheme 10/10 vs variant 20/20.

python -m src.experiments.anova [--T 480] [--runs 1600]

Terminating simulation from empty queues, horizon T (one shift by
default). The factor is the conveyor scheme (q = 2 levels), each
level has p runs.

    D_f = p * sum_j (y_j - y)^2 / (q - 1)           factor
    D_r = sum_j sum_s (y_js - y_j)^2 / (q (p - 1))  residual
    F = D_f / D_r, compared with
    F_crit(alpha = 0.05; q - 1, q (p - 1)).

The guide's formula (3.10) divides the factor sum of squares by
q * p - 1; the standard one-factor ANOVA (used here, and by
scipy.stats.f_oneway, which is run as a cross-check) divides it
by q - 1. Formula (3.11) is the standard one.

ANOVA needs independent samples, so the levels use disjoint seed
ranges (1..p and 1000001..). With the same seeds the first arrival
intervals of both variants come from the same random numbers and
the paired outputs correlate (about +0.3), which breaks that
assumption.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

from src.experiments.common import replicate_parallel

RESULTS = Path("results")
ALPHA = 0.05

VARIANTS = [("10/10", dict(k=10, period=10.0)),
            ("20/20", dict(k=20, period=20.0))]
RESPONSES = [
    ("p_skip", "Ймовірність пропуску секції, –"),
    ("q1_mean", "Середня довжина черги виробів типу 1, од."),
    ("q2_mean", "Середня довжина черги виробів типу 2, од."),
    ("q1_max", "Максимальна довжина черги виробів типу 1, од."),
    ("q2_max", "Максимальна довжина черги виробів типу 2, од."),
    ("items_per_hour", "Пропускна здатність, виробів/год"),
]


def oneway_anova(groups, alpha=ALPHA):
    """groups: equal-size samples. Own formulas + scipy check."""
    y = [np.asarray(g, dtype=float) for g in groups]
    q, p = len(y), len(y[0])
    means = np.array([g.mean() for g in y])
    grand = means.mean()
    ss_f = p * np.sum((means - grand) ** 2)
    ss_r = sum(np.sum((g - m) ** 2) for g, m in zip(y, means))
    df_f, df_r = q - 1, q * (p - 1)
    d_f, d_r = ss_f / df_f, ss_r / df_r
    f = d_f / d_r if d_r > 0 else math.inf
    f_crit = stats.f.ppf(1 - alpha, df_f, df_r)
    sp = stats.f_oneway(*y)
    return {"D_f": d_f, "D_r": d_r, "df_f": df_f, "df_r": df_r,
            "F": f, "F_crit": f_crit,
            "p_value": float(stats.f.sf(f, df_f, df_r)),
            "significant": bool(f > f_crit),
            "F_scipy": float(sp.statistic),
            "p_scipy": float(sp.pvalue)}


def diff_ci(a, b, beta=0.95):
    """Mean of b - a with a Welch confidence interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = ((va + vb) ** 2
          / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
          if se > 0 else 1)
    t_crit = stats.t.ppf((1 + beta) / 2, df)
    return float(b.mean() - a.mean()), float(t_crit * se)


SEED_STEP = 1_000_000


def run(T, runs, seed=1):
    data = {name: replicate_parallel(runs, T,
            seed=seed + i * SEED_STEP, **kw)
            for i, (name, kw) in enumerate(VARIANTS)}
    (na, ra), (nb, rb) = [(n, data[n]) for n, _ in VARIANTS]
    rows = []
    for key, label in RESPONSES:
        a = [r[key] for r in ra]
        b = [r[key] for r in rb]
        res = oneway_anova([a, b])
        d, h = diff_ci(a, b)
        rows.append({"T": T, "runs": runs, "response": key,
                     "label": label,
                     "mean_" + na: float(np.mean(a)),
                     "mean_" + nb: float(np.mean(b)),
                     "diff": d, "diff_ci": h,
                     "pair_corr": float(np.corrcoef(a, b)[0, 1]),
                     **res})
    return rows


def practical(rows, T):
    """Effect in shop terms: per shift of 480 min."""
    by = {r["response"]: r for r in rows}
    shift = 480 / T
    out = {}
    p = by["p_skip"]
    # sections per T: T/10 and T/20; items not assembled = P *
    # sections * k = P * T (each type)
    out["empty_sections_10"] = p["mean_10/10"] * T / 10 * shift
    out["empty_sections_20"] = p["mean_20/20"] * T / 20 * shift
    out["items_not_assembled_10"] = p["mean_10/10"] * T * shift
    out["items_not_assembled_20"] = p["mean_20/20"] * T * shift
    return out


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, nargs="+", default=[480.0])
    ap.add_argument("--runs", type=int, default=1600)
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    for T in a.T:
        rows = run(T, a.runs)
        write_csv(RESULTS / f"anova_T{T:g}.csv", rows)
        print(f"T_mod = {T:g} хв, {a.runs} прогонів на варіант, "
              f"α = {ALPHA}")
        for r in rows:
            verdict = ("значущий" if r["significant"]
                       else "незначущий")
            print(f"  {r['response']:14} 10/10 "
                  f"{r['mean_10/10']:9.4f}  20/20 "
                  f"{r['mean_20/20']:9.4f}  "
                  f"різниця {r['diff']:+8.4f} ± "
                  f"{r['diff_ci']:.4f}  F = {r['F']:8.3f} "
                  f"(F_кр = {r['F_crit']:.3f}, scipy "
                  f"{r['F_scipy']:.3f}, p = {r['p_value']:.3g}) "
                  f"{verdict}; кор. пар {r['pair_corr']:+.3f}")
        for k, v in practical(rows, T).items():
            print(f"  {k}: {v:.3f} за зміну")


if __name__ == "__main__":
    main()
