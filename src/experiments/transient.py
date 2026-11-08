"""Transient period study (guide 3.1.4, criterion 3.1).

python -m src.experiments.transient

Base model and variant 20/20 (no buffer): running means of 5 runs over a long
horizon. The queues do not settle (load factor 1, Q1 - Q2 is a zero-drift
random walk), so criterion (3.1) is not reached for them.

Buffer model (kanban):
1) warm-up - ensemble mean over many runs of the time-weighted queue in short
   windows, for the grid corner with the largest buffers (slowest to settle);
2) run length - running means of 5 runs collected after the warm-up, for the
   point of the grid where the decision P_skip <= level is taken; criterion (3.1).
"""
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np

from src.model.conveyor import FILL, SKIP, build_conveyor_net
from src.petri import Simulator
from src.plots.style import RUN_COLORS, decimal_comma, plt

RESULTS = Path("results")

N_RUNS = 5
BASE_T, BASE_STEP = 20000, 50

BUF_CORNER = (30, 80)      # largest buffers of the series grid
BUF_DECISION = (20, 60)    # region where P_skip crosses the chosen level
ENS_RUNS, ENS_T, ENS_STEP = 400, 6000, 50
RUN_T, RUN_STEP = 100000, 100
T_MOD_BUF = 40000          # chosen run length of the buffer model (warm-up 3000 min)
# eps_T of criterion (3.1): run-to-run spread of single-run means; the final
# accuracy of the estimates is reached by averaging p runs (replications.py)
EPS = {"p_skip": 0.005, "q1_mean": 2.0, "q2_mean": 2.0}

LABELS = {"q1_mean": "Середня довжина черги\nвиробів типу 1, од.",
          "q2_mean": "Середня довжина черги\nвиробів типу 2, од.",
          "p_skip": "Ймовірність пропуску\nсекції, –"}


def trace_run(seed, T, step, warmup=0.0, **model):
    """Running (cumulative) means and window means on a time grid.

    q1, q2 are time-weighted means over each grid window (t - step, t].
    Instantaneous markings are not used: the grid points are multiples of the
    conveyor period, where the queues sit at their pre-fill maximum.
    """
    rows = []

    def sample(sim):
        n_fill, n_skip = sim.window_out[FILL], sim.window_out[SKIP]
        q1, q2 = sim.net.places["Q1"].stat, sim.net.places["Q2"].stat
        rows.append((sim.t, q1.area, q2.area, q1.mean, q2.mean,
                     n_skip / (n_fill + n_skip) if n_fill + n_skip else math.nan))

    Simulator(build_conveyor_net(**model), seed=seed).run(
        T, warmup=warmup, sample_every=step, on_sample=sample)
    a = np.array(rows, dtype=float)
    win = np.diff(np.concatenate(([0.0], a[:, 0])))
    win[a[:, 0] <= warmup] = math.nan
    out = {"t": a[:, 0],
           "q1": np.diff(np.concatenate(([0.0], a[:, 1]))) / win,
           "q2": np.diff(np.concatenate(([0.0], a[:, 2]))) / win,
           "q1_mean": a[:, 3], "q2_mean": a[:, 4], "p_skip": a[:, 5]}
    keep = out["t"] > warmup
    return {k: v[keep] for k, v in out.items()}


def traces(seeds, T, step, warmup=0.0, **model):
    with ProcessPoolExecutor() as ex:
        return list(ex.map(partial(trace_run, T=T, step=step, warmup=warmup, **model), seeds))


def max_pair_diff(runs, key):
    return np.max([np.abs(a[key] - b[key]) for a, b in combinations(runs, 2)], axis=0)


def settle_time(t, diff, eps):
    """First grid time after which the spread stays below eps (criterion 3.1)."""
    bad = np.where(~(diff < eps))[0]
    if len(bad) == 0:
        return t[0]
    return math.inf if bad[-1] == len(t) - 1 else t[bad[-1] + 1]


def plot_running(runs, path, x0=0.0, t_mod=None):
    keys = list(LABELS)
    fig, axes = plt.subplots(len(keys), 1, figsize=(7.5, 2.6 * len(keys)), sharex=True)
    for ax, key in zip(axes, keys):
        for i, r in enumerate(runs):
            ax.plot(r["t"], r[key], color=RUN_COLORS[i], lw=1, label=f"прогін {i + 1}")
        ax.set_ylabel(LABELS[key])
        if key == "p_skip":
            ax.set_ylim(bottom=0)
        if t_mod:
            ax.axvline(t_mod, color="k", ls="--", lw=1, label=f"T_mod = {t_mod:g} хв")
        decimal_comma(ax)
    axes[-1].set_xlabel("Модельний час t, хв")
    axes[-1].set_xlim(left=x0)
    axes[0].legend(ncol=3, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def base_study(name, model, file):
    runs = traces(range(1, N_RUNS + 1), BASE_T, BASE_STEP, **model)
    t = runs[0]["t"]
    rows = []
    for T in (480, 2400, 5000, 10000, 20000):
        i = np.searchsorted(t, T)
        rows.append({"model": name, "T": T,
                     **{f"spread_{k}": float(max_pair_diff(runs, k)[i]) for k in LABELS},
                     **{f"mean_{k}": float(np.mean([r[k][i] for r in runs])) for k in LABELS}})
    plot_running(runs, RESULTS / file)
    return rows


def warmup_study(buffer):
    ens = traces(range(1, ENS_RUNS + 1), ENS_T, ENS_STEP, buffer=buffer)
    t = ens[0]["t"]
    q1 = np.mean([r["q1"] for r in ens], axis=0)
    q2 = np.mean([r["q2"] for r in ens], axis=0)
    tail = t > ENS_T * 2 / 3
    lvl1, lvl2 = q1[tail].mean(), q2[tail].mean()
    # warm-up ends when the ensemble mean stays within 5 % of the steady level
    ok = (np.abs(q1 - lvl1) <= 0.05 * lvl1) & (np.abs(q2 - lvl2) <= 0.05 * lvl2)
    bad = np.where(~ok)[0]
    t_raw = float(t[bad[-1] + 1]) if len(bad) else float(t[0])
    t_warm = float(math.ceil(t_raw / 1000) * 1000)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(t, q1, color=RUN_COLORS[0], lw=1, label="черга виробів типу 1")
    ax.plot(t, q2, color=RUN_COLORS[1], lw=1, label="черга виробів типу 2")
    for lvl, c in ((lvl1, RUN_COLORS[0]), (lvl2, RUN_COLORS[1])):
        ax.axhspan(0.95 * lvl, 1.05 * lvl, color=c, alpha=0.12)
    ax.axvline(t_warm, color="k", ls="--", lw=1, label=f"кінець перехідного періоду, {t_warm:g} хв")
    ax.set_xlabel("Модельний час t, хв")
    ax.set_ylabel(f"Середня за {ENS_RUNS} прогонами\nдовжина черги, од.")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="lower right")
    decimal_comma(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "transient_buffer_warmup.png")
    plt.close(fig)
    return {"level_q1": lvl1, "level_q2": lvl2, "t_warm_raw": t_raw, "t_warm": t_warm}


def run_length_study(buffer, t_warm, file=None):
    runs = traces(range(1, N_RUNS + 1), RUN_T, RUN_STEP, warmup=t_warm, buffer=buffer)
    t = runs[0]["t"]
    settle = {k: float(settle_time(t, max_pair_diff(runs, k), e)) for k, e in EPS.items()}
    final = {k: float(max_pair_diff(runs, k)[-1]) for k in EPS}
    if file:
        plot_running(runs, RESULTS / file, x0=t_warm, t_mod=T_MOD_BUF)
    return settle, final


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    base = base_study("10/10", dict(), "transient_base.png")
    base += base_study("20/20", dict(k=20, period=20.0), "transient_variant20.png")
    write_csv(RESULTS / "transient_base.csv", base)
    print("Без буфера: розкид поточних середніх між 5 прогонами (max |y_i - y_j|) / середнє")
    for r in base:
        print(f"  {r['model']:6} T={r['T']:6}  " + "  ".join(
            f"{k}: {r['spread_' + k]:.4f} / {r['mean_' + k]:.4f}" for k in LABELS))

    warm = warmup_study(BUF_CORNER)
    print(f"\nБуфер {BUF_CORNER}: сталий рівень Q1 = {warm['level_q1']:.2f}, Q2 = {warm['level_q2']:.2f}; "
          f"перехідний період {warm['t_warm_raw']:g} -> {warm['t_warm']:g} хв")
    rows = []
    for buf, file in ((BUF_DECISION, "transient_buffer_running.png"), (BUF_CORNER, None)):
        settle, final = run_length_study(buf, warm["t_warm"], file)
        rows.append({"buffer": f"{buf[0]}/{buf[1]}", **warm, "T": RUN_T,
                     **{f"eps_{k}": v for k, v in EPS.items()},
                     **{f"settle_{k}": v for k, v in settle.items()},
                     **{f"spread_at_T_{k}": v for k, v in final.items()}})
        print(f"Буфер {buf}: критерій (3.1) виконується з t = "
              + ", ".join(f"{k} (eps {EPS[k]:g}): {v:g}" for k, v in settle.items())
              + f"; розкид при T = {RUN_T}: " + ", ".join(f"{k} {v:.4f}" for k, v in final.items()))
    write_csv(RESULTS / "transient_buffer.csv", rows)


if __name__ == "__main__":
    main()
