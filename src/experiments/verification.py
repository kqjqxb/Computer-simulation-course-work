"""Verification of the simulation algorithm.

python -m src.experiments.verification [--runs 100] [--T 2000]

Each row changes one or two inputs of the base model and compares the outputs
with an analytical expectation (src/experiments/theory.py) or, for fixed
arrival intervals, with a hand calculation. Writes
    results/verification.csv
    results/protocol_deterministic.txt           (full event protocol)
    results/protocol_deterministic_fragment.txt  (first events, for the paper)
"""
import argparse
import csv
import math
from pathlib import Path

from src.experiments.common import mean_ci, replicate_parallel, run_once
from src.experiments.theory import excess_growth_coef, fluid, queue_difference, rate
from src.petri import EventProtocol, const, unif

RESULTS = Path("results")

BASE = dict(k=10, period=10.0, batch1=5, interval1=unif(5, 1), batch2=20, interval2=unif(20, 7))
BASE_TEXT = ("k = 10 виробів/секцію, конвеєр 1 секція / 10 хв, тип 1: 5 шт. кожні unif(5, 1) хв, "
             "тип 2: 20 шт. кожні unif(20, 7) хв, початкові черги 0, збір статистики з 0 хв")

ROWS = [
    ("1", "базова модель", {}, "random"),
    ("2", "детерміновані інтервали надходження", dict(interval1=const(5), interval2=const(20)), "det"),
    ("3", "інтервал надходження типу 2 у 2 рази більший", dict(interval2=unif(40, 7)), "random"),
    ("4", "конвеєр у 2 рази швидший", dict(period=5.0), "random"),
    ("5", "конвеєр у 2 рази повільніший", dict(period=20.0), "random"),
    ("6", "варіант 20/20", dict(k=20, period=20.0), "random"),
    ("7", "партія типу 1 у 2 рази більша", dict(batch1=10), "random"),
]

PROTOCOL_PLACES = ["Q1", "Q2", "Section", "Filled", "Skipped"]
METRICS = ["p_skip", "q1_mean", "q2_mean", "q1_max", "q2_max"]


def describe(params):
    names = {"k": "k", "period": "період, хв", "batch1": "b1", "interval1": "інтервал 1, хв",
             "batch2": "b2", "interval2": "інтервал 2, хв"}
    return "; ".join(f"{names[key]} = {val!r}" if not isinstance(val, (int, float))
                     else f"{names[key]} = {val:g}" for key, val in params.items())


def deterministic_expectation(T):
    """Hand calculation for intervals exactly 5 and 20 min, k = 10, period 10.

    t = 10: Q1 = 10, Q2 = 0  -> the only skipped section.
    From t = 20 every section is filled; Q1 is 10 on [20 + 10j, 25 + 10j) and 15
    on [25 + 10j, 30 + 10j); Q2 is 10 on [20 + 20j, 30 + 20j) and 0 on
    [30 + 20j, 40 + 20j). Before t = 20: Q1 = 0, 5, 10, 15 on 5-min steps, Q2 = 0.
    """
    sections = math.floor(T / 10)
    q1_area = (0 + 5 + 10 + 15) * 5 + 12.5 * (T - 20)
    q2_area = 5.0 * (T - 20)
    return {"p_skip": 1 / sections, "q1_mean": q1_area / T, "q2_mean": q2_area / T,
            "q1_max": 15, "q2_max": 10, "delta": (q1_area - q2_area) / T}


def analytic_expectation(params, T):
    lam1 = rate(params["batch1"], params["interval1"])
    lam2 = rate(params["batch2"], params["interval2"])
    fl = fluid(params["k"], params["period"], lam1, lam2)
    exp = {"lambda1": lam1, "lambda2": lam2, **fl,
           "delta": queue_difference(T, params["batch1"], params["interval1"],
                                     params["batch2"], params["interval2"]),
           "items_per_hour": 60 * fl["fill_rate"]}
    for i in (1, 2):
        s = fl[f"slope{i}"]
        exp[f"q{i}_mean"] = s * T / 2 if s > 0 else math.nan
        exp[f"q{i}_max"] = s * T if s > 0 else math.nan
    return exp


def save_protocol(T):
    proto = EventProtocol(places=PROTOCOL_PLACES, headers=("t, хв", "фаза", "перехід"),
                          phase_names={"IN": "вхід", "OUT": "вихід"})
    run_once(1, T, protocol=proto, **{**BASE, "interval1": const(5), "interval2": const(20)})
    text = proto.to_text()
    (RESULTS / "protocol_deterministic.txt").write_text(text + "\n", encoding="utf-8")
    fragment = "\n".join(text.splitlines()[:40])
    (RESULTS / "protocol_deterministic_fragment.txt").write_text(fragment + "\n", encoding="utf-8")
    return len(proto.rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--T", type=float, default=2000)
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    out = []
    for no, change, changed, kind in ROWS:
        params = {**BASE, **changed}
        runs = 1 if kind == "det" else a.runs
        res = replicate_parallel(runs, a.T, **params)
        row = {"no": no, "change": change, "inputs": describe(changed), "T": a.T, "runs": runs}
        for key in METRICS + ["items_per_hour"]:
            row[key], row[key + "_ci"] = mean_ci([r[key] for r in res])
        row["delta"], row["delta_ci"] = mean_ci([r["q1_mean"] - r["q2_mean"] for r in res])
        for i in (1, 2):
            row[f"slope{i}"], row[f"slope{i}_ci"] = mean_ci([r[f"q{i}_end"] / a.T for r in res])

        exp = deterministic_expectation(a.T) if kind == "det" else analytic_expectation(params, a.T)
        for key, val in exp.items():
            row["exp_" + key] = val

        # underloaded conveyor: min(Q1, Q2) stays bounded, but the excess
        # |A1 - A2| of one type grows like sqrt(T) (theory.excess_growth_coef);
        # compare the growth of Q1_mean + Q2_mean between T and 5T
        if kind == "random" and exp["rho"] < 1:
            long = replicate_parallel(runs, 5 * a.T, **params)
            for key in ("q1_mean", "q2_mean"):
                row[key + "_5T"], row[key + "_5T_ci"] = mean_ci([r[key] for r in long])
            growth = [(l["q1_mean"] + l["q2_mean"]) - (s_["q1_mean"] + s_["q2_mean"])
                      for s_, l in zip(res, long)]
            row["sum_growth"], row["sum_growth_ci"] = mean_ci(growth)
            coef = excess_growth_coef(params["batch1"], params["interval1"],
                                      params["batch2"], params["interval2"])
            row["exp_sum_growth"] = coef * (math.sqrt(5 * a.T) - math.sqrt(a.T))
        out.append(row)

    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with (RESULTS / "verification.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(out)

    n_events = save_protocol(a.T)

    print(f"Незмінні значення: {BASE_TEXT}")
    print(f"T = {a.T:g} хв; прогонів: {a.runs} (детермінований рядок – 1)\n")
    for r in out:
        print(f"[{r['no']}] {r['change']}  ({r['inputs'] or 'без змін'})")
        for key in METRICS + ["delta", "slope1", "slope2", "items_per_hour"]:
            ci = r.get(key + "_ci", math.nan)
            e = r.get("exp_" + key, math.nan)
            ci_s = "" if math.isnan(ci) else f" ± {ci:.4f}"
            e_s = "" if e is None or (isinstance(e, float) and math.isnan(e)) else f"   очікується {e:.4f}"
            print(f"    {key:15} {r[key]:11.4f}{ci_s}{e_s}")
        for key in ("q1_mean_5T", "q2_mean_5T"):
            if key in r:
                print(f"    {key:15} {r[key]:11.4f} ± {r[key + '_ci']:.4f}   (T = {5 * a.T:g})")
        if "sum_growth" in r:
            print(f"    {'sum_growth':15} {r['sum_growth']:11.4f} ± {r['sum_growth_ci']:.4f}"
                  f"   очікується {r['exp_sum_growth']:.4f}  (приріст Q1+Q2 від T до 5T)")
    print(f"\nПротокол детермінованого сценарію: {n_events} подій -> results/protocol_deterministic.txt")


if __name__ == "__main__":
    main()
