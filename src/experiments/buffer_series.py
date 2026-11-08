"""Buffer (kanban) model in steady state: series over the buffer sizes (L1, L2).

python -m src.experiments.buffer_series [--runs 120] [--level 0.02]

T_mod = 40000 min, statistics after a 3000-min warm-up (transient.py),
p = 120 runs per grid point (replications.py), seeds 1..p at every point.

1) Full search over the grid: the minimal total buffer L1 + L2 for which the
   upper bound of the 95 % confidence interval of P_skip does not exceed the level.
2) Regression of P_skip on L2 (L1 fixed): factor normalized to [-1; 1] by (3.6),
   coefficients b = (X^T X)^-1 X^T Y (3.9), correlation index R. It is fitted
   on L2 >= 40: L2 = 30 is a separate regime (30 = one type-2 batch + one
   section, the next batch cannot start until a section takes items, P jumps
   from 0.04 to 0.20), outside the region of interest P <= 0.05.

Cost of the buffer: block_i - share of time the type-i supplier waits for room;
lost production = 60 * lambda_i * block_i items/h of each type.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

from src.experiments.common import mean_ci, replicate_parallel
from src.plots.style import RUN_COLORS, decimal_comma, plt

RESULTS = Path("results")
T_MOD, WARMUP = 40000, 3000
L1_GRID = (15, 20, 25, 30)
L2_GRID = (30, 40, 50, 60, 70, 80)
LAMBDA = 1.0  # nominal supply of each type, items/min
KEYS = ["p_skip", "q1_mean", "q2_mean", "q1_max", "q2_max", "items_per_hour", "block1", "block2"]


def grid_study(runs):
    rows = []
    for l1 in L1_GRID:
        for l2 in L2_GRID:
            res = replicate_parallel(runs, T_MOD, warmup=WARMUP, buffer=(l1, l2))
            row = {"L1": l1, "L2": l2, "runs": runs, "T": T_MOD, "warmup": WARMUP}
            for k in KEYS:
                row[k], row[k + "_ci"] = mean_ci([r[k] for r in res])
            row["q1_max_all"] = max(r["q1_max"] for r in res)
            row["q2_max_all"] = max(r["q2_max"] for r in res)
            for i in (1, 2):
                row[f"lost{i}_per_hour"] = 60 * LAMBDA * row[f"block{i}"]
            rows.append(row)
            print(f"L = ({l1}, {l2}): P = {row['p_skip']:.4f} ± {row['p_skip_ci']:.4f}, "
                  f"Q1 = {row['q1_mean']:.2f}, Q2 = {row['q2_mean']:.2f}, "
                  f"блок. {row['block1']:.4f}/{row['block2']:.4f}")
    return rows


def best_buffer(rows, level):
    ok = [r for r in rows if r["p_skip"] + r["p_skip_ci"] <= level]
    return min(ok, key=lambda r: (r["L1"] + r["L2"], r["p_skip"])) if ok else None


def normalize(x):
    """(3.6): x_norm = (X - X0) / delta,  X0 = (Xmax + Xmin) / 2,  delta = (Xmax - Xmin) / 2."""
    x = np.asarray(x, dtype=float)
    x0, delta = (x.max() + x.min()) / 2, (x.max() - x.min()) / 2
    return (x - x0) / delta, x0, delta


def least_squares(columns, y):
    """(3.9): b = (X^T X)^-1 X^T Y."""
    X = np.column_stack([np.ones(len(y))] + list(columns))
    return np.linalg.solve(X.T @ X, X.T @ y), X


def correlation_index(y, y_hat):
    return math.sqrt(max(0.0, 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)))


REG_L2_MIN = 40


def regression(rows, l1, l2_min=REG_L2_MIN):
    pts = sorted((r for r in rows if r["L1"] == l1 and r["L2"] >= l2_min), key=lambda r: r["L2"])
    L2 = np.array([r["L2"] for r in pts], dtype=float)
    P = np.array([r["p_skip"] for r in pts])
    x, x0, delta = normalize(L2)
    models = {
        "P = b0 + b1 x": (lambda: [x], False),
        "P = b0 + b1 x + b2 x^2": (lambda: [x, x ** 2], False),
        "ln P = b0 + b1 x": (lambda: [x], True),
        "ln P = b0 + b1 x + b2 x^2": (lambda: [x, x ** 2], True),
    }
    out = []
    for name, (cols, log) in models.items():
        target = np.log(P) if log else P
        b, X = least_squares(cols(), target)
        fit = X @ b
        p_hat = np.exp(fit) if log else fit
        out.append({"L1": l1, "L2_from": float(L2.min()), "L2_to": float(L2.max()),
                    "model": name, "x0": x0, "delta": delta,
                    **{f"b{i}": float(v) for i, v in enumerate(b)},
                    "R": correlation_index(P, p_hat),
                    "max_abs_err": float(np.max(np.abs(P - p_hat)))})
    return out, (L2, P, x0, delta)


def predict(model_row, L2):
    x = (np.asarray(L2, float) - model_row["x0"]) / model_row["delta"]
    b = [model_row.get(f"b{i}", 0.0) for i in range(3)]
    b = [0.0 if (isinstance(v, float) and math.isnan(v)) else v for v in b]
    y = b[0] + b[1] * x + b[2] * x ** 2
    return np.exp(y) if model_row["model"].startswith("ln") else y


def solve_level(model_row, level):
    """Continuous L2 with predicted P_skip = level (bisection on the fitted range)."""
    lo, hi = model_row["L2_from"], model_row["L2_to"]
    f = lambda v: float(predict(model_row, [v])[0]) - level
    if f(lo) * f(hi) > 0:
        return math.nan
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def plot_series(rows, level, best, reg_row, path_p, path_q):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, l1 in enumerate(L1_GRID):
        pts = sorted((r for r in rows if r["L1"] == l1), key=lambda r: r["L2"])
        ax.errorbar([r["L2"] for r in pts], [r["p_skip"] for r in pts],
                    yerr=[r["p_skip_ci"] for r in pts], marker="o", ms=4, capsize=3,
                    color=RUN_COLORS[i], lw=1, label=f"L1 = {l1}")
    grid = np.linspace(reg_row["L2_from"], reg_row["L2_to"], 200)
    pretty = {"ln P = b0 + b1 x + b2 x^2": r"$\ln P = b_0 + b_1 x + b_2 x^2$",
              "ln P = b0 + b1 x": r"$\ln P = b_0 + b_1 x$",
              "P = b0 + b1 x + b2 x^2": r"$P = b_0 + b_1 x + b_2 x^2$",
              "P = b0 + b1 x": r"$P = b_0 + b_1 x$"}
    ax.plot(grid, predict(reg_row, grid), "k--", lw=1,
            label=f"регресія: {pretty.get(reg_row['model'], reg_row['model'])}")
    ax.axhline(level, color="gray", ls=":", lw=1.2, label=f"рівень {level:g}".replace(".", ","))
    if best:
        ax.plot(best["L2"], best["p_skip"], "s", ms=10, mfc="none", mec="k",
                label=f"мінімальний буфер ({best['L1']}, {best['L2']})")
    ax.set_yscale("log")
    ticks = [0.01, 0.02, 0.05, 0.1, 0.2]
    ax.set_yticks(ticks, [f"{t:g}".replace(".", ",") for t in ticks])
    ax.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Ємність буфера виробів типу 2 L2, од.")
    ax.set_ylabel("Ймовірність пропуску секції, –")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path_p)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, key, title in ((axes[0], "q1_mean", "типу 1"), (axes[1], "q2_mean", "типу 2")):
        for i, l1 in enumerate(L1_GRID):
            pts = sorted((r for r in rows if r["L1"] == l1), key=lambda r: r["L2"])
            ax.plot([r["L2"] for r in pts], [r[key] for r in pts], marker="o", ms=4,
                    color=RUN_COLORS[i], lw=1, label=f"L1 = {l1}")
        ax.set_xlabel("Ємність буфера L2, од.")
        ax.set_ylabel(f"Середня довжина черги\nвиробів {title}, од.")
        decimal_comma(ax)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path_q)
    plt.close(fig)


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=120)
    ap.add_argument("--level", type=float, default=0.02)
    ap.add_argument("--reuse", action="store_true", help="read results/buffer_series.csv")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    if a.reuse:
        with (RESULTS / "buffer_series.csv").open(encoding="utf-8") as f:
            rows = [{k: (int(v) if k in ("L1", "L2", "runs") else float(v)) for k, v in r.items()}
                    for r in csv.DictReader(f)]
    else:
        rows = grid_study(a.runs)
        write_csv(RESULTS / "buffer_series.csv", rows)
    best = best_buffer(rows, a.level)
    print(f"\nМінімальний буфер з P_skip (верхня межа ДІ) <= {a.level}: "
          + (f"({best['L1']}, {best['L2']}), P = {best['p_skip']:.4f} ± {best['p_skip_ci']:.4f}" if best else "немає"))

    l1 = best["L1"] if best else 20
    reg, _ = regression(rows, l1)
    write_csv(RESULTS / "buffer_regression.csv", reg)
    print(f"\nРегресія P_skip від L2 при L1 = {l1}, L2 від {REG_L2_MIN}, "
          f"x = (L2 - {reg[0]['x0']:g}) / {reg[0]['delta']:g}")
    for r in reg:
        coefs = ", ".join(f"b{i} = {r[f'b{i}']:.4f}" for i in range(3) if f"b{i}" in r)
        print(f"  {r['model']:28} {coefs}  R = {r['R']:.4f}, max |похибка| = {r['max_abs_err']:.4f}")
    chosen = max(reg, key=lambda r: r["R"])
    chosen["L2_at_level"] = solve_level(chosen, a.level)
    write_csv(RESULTS / "buffer_regression.csv", reg)
    print(f"Обрано: {chosen['model']}; P = {a.level} при L2 = {chosen['L2_at_level']:.1f}")
    plot_series(rows, a.level, best, chosen,
                RESULTS / "buffer_p_skip.png", RESULTS / "buffer_queues.png")


if __name__ == "__main__":
    main()
