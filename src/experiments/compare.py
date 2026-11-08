"""Proposal (kanban buffer) compared with the base scheme and variant 20/20.

python -m src.experiments.compare [--T 480 2400] [--runs 1600] [--buffers 20 60 25 80]

1) Same conditions: empty queues at t = 0, horizon T, the same seeds 1..p for
   every model (common random numbers); differences to the base model are
   paired (run s of a model minus run s of the base), CI by Student.
2) Steady state: base model and the buffer models, T = 40000 min, warm-up 3000,
   120 runs - the buffers keep the queues bounded, the base model does not.

Two proposal variants: (20, 60) - minimal buffer with P_skip <= 0.02,
(25, 80) - smaller loss of completeness (P_skip about 0.01).

Cost of the buffer: block_i (share of time the supplier waits) and the lost
production 60 * lambda_i * block_i items/h of each type (lambda_i = 1 item/min).
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

from src.experiments.common import mean_ci, replicate_parallel
from src.plots.style import RUN_COLORS, decimal_comma, plt

RESULTS = Path("results")
KEYS = ["p_skip", "q1_mean", "q2_mean", "q1_max", "q2_max", "items_per_hour",
        "block1", "block2", "lost1", "lost2"]
STEADY_T, STEADY_WARM, STEADY_RUNS = 40000, 3000, 120


def models(buffers):
    return ([("база 10/10", dict()), ("варіант 20/20", dict(k=20, period=20.0))]
            + [(f"буфер ({b[0]}, {b[1]})", dict(buffer=tuple(b))) for b in buffers])


def with_lost(res):
    for r in res:
        r["lost1"] = 60.0 * r["block1"]
        r["lost2"] = 60.0 * r["block2"]
    return res


def paired_ci(a, b, beta=0.95):
    d = np.asarray(b, float) - np.asarray(a, float)
    h = stats.t.ppf((1 + beta) / 2, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return float(d.mean()), float(h)


def compare(T, runs, buffers, warmup=0.0, model_list=None):
    model_list = model_list or models(buffers)
    data = {name: with_lost(replicate_parallel(runs, T, seed=1, warmup=warmup, **kw))
            for name, kw in model_list}
    base = model_list[0][0]
    rows = []
    for name, kw in model_list:
        row = {"T": T, "warmup": warmup, "runs": runs, "model": name}
        for k in KEYS:
            vals = [r[k] for r in data[name]]
            row[k], row[k + "_ci"] = mean_ci(vals)
            if name != base:
                row["d_" + k], row["d_" + k + "_ci"] = paired_ci([r[k] for r in data[base]], vals)
        row["q1_max_all"] = max(r["q1_max"] for r in data[name])
        row["q2_max_all"] = max(r["q2_max"] for r in data[name])
        # per shift of 480 min: empty sections and items of each type not assembled
        # (a section carries k items, 480 / period sections a shift, k / period = 1 item/min)
        row["empty_sections_per_shift"] = row["p_skip"] * 480 / kw.get("period", 10.0)
        row["items_not_assembled_per_shift"] = row["p_skip"] * 480
        rows.append(row)
    return rows


def plot_name(model):
    return model.replace("база", "базовий варіант").replace("буфер", "накопичувач")


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return [{k: (v if k == "model" else float(v) if v else float("nan")) for k, v in r.items()}
                for r in csv.DictReader(f)]


def plot_compare(rows, path):
    Ts = sorted({r["T"] for r in rows})
    names = list(dict.fromkeys(r["model"] for r in rows))
    panels = [("p_skip", "Ймовірність пропуску секції, –"),
              ("q1_mean", "Середня довжина черги\nвиробів типу 1, од."),
              ("q2_mean", "Середня довжина черги\nвиробів типу 2, од."),
              ("q2_max", "Максимальна довжина черги\nвиробів типу 2, од.")]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6))
    width = 0.8 / len(names)
    for ax, (key, label) in zip(axes.flat, panels):
        for i, n in enumerate(names):
            vals = [next(r for r in rows if r["T"] == T and r["model"] == n) for T in Ts]
            x = np.arange(len(Ts)) + (i - (len(names) - 1) / 2) * width
            ax.bar(x, [v[key] for v in vals], width, yerr=[v[key + "_ci"] for v in vals],
                   capsize=3, color=RUN_COLORS[i], label=plot_name(n))
        ax.set_xticks(np.arange(len(Ts)),
                      [rf"$T_{{\mathrm{{мод}}}}$ = {T:g} хв" for T in Ts])
        ax.set_ylabel(label)
        decimal_comma(ax, "y")
    axes[0, 0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def show(rows):
    for r in rows:
        line = f"  {r['model']:16} T={r['T']:6g}  " + "  ".join(
            f"{k} {r[k]:.4f}±{r[k + '_ci']:.4f}" for k in ("p_skip", "q1_mean", "q2_mean", "q1_max", "q2_max",
                                                            "items_per_hour", "lost1", "lost2"))
        print(line)
        if "d_p_skip" in r:
            print("      різниця з базою: " + "  ".join(
                f"{k} {r['d_' + k]:+.4f}±{r['d_' + k + '_ci']:.4f}"
                for k in ("p_skip", "q1_mean", "q2_mean", "q2_max", "items_per_hour")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, nargs="+", default=[480.0, 2400.0])
    ap.add_argument("--runs", type=int, default=1600)
    ap.add_argument("--buffers", type=int, nargs="+", default=[20, 60, 25, 80],
                    help="pairs L1 L2")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the chart from results/compare_terminating.csv")
    a = ap.parse_args()
    if a.plot_only:
        plot_compare(read_csv(RESULTS / "compare_terminating.csv"),
                     RESULTS / "compare_terminating.png")
        return
    buffers = [tuple(a.buffers[i:i + 2]) for i in range(0, len(a.buffers), 2)]
    RESULTS.mkdir(exist_ok=True)

    rows = []
    for T in a.T:
        rows += compare(T, a.runs, buffers)
    write_csv(RESULTS / "compare_terminating.csv", rows)
    plot_compare(rows, RESULTS / "compare_terminating.png")
    print("Порожній старт, однакові seed:")
    show(rows)

    ml = [m for m in models(buffers) if "20/20" not in m[0]]
    steady = compare(STEADY_T, STEADY_RUNS, buffers, warmup=STEADY_WARM, model_list=ml)
    write_csv(RESULTS / "compare_steady.csv", steady)
    print(f"\nСталий режим: T = {STEADY_T} хв, прогрів {STEADY_WARM} хв, {STEADY_RUNS} прогонів")
    show(steady)


if __name__ == "__main__":
    main()
