"""Difference of the mean queues Q1 - Q2 compared with renewal theory.

python -m src.experiments.queue_offset [--runs 200] [--T 10000]
"""
import argparse
import csv
from pathlib import Path

from src.experiments.common import mean_ci, replicate_parallel
from src.experiments.theory import queue_difference
from src.petri import unif

RESULTS = Path("results")

CASES = [
    ("базовий: 5 шт. unif(5, 1) / 20 шт. unif(20, 7)", dict(batch2=20, interval2=unif(20, 7))),
    ("обидва типи: 5 шт. unif(5, 1)", dict(batch2=5, interval2=unif(5, 1))),
    ("тип 2: 20 шт. unif(20, 1)", dict(batch2=20, interval2=unif(20, 1))),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--T", type=float, default=10000)
    a = ap.parse_args()

    rows = []
    for name, kw in CASES:
        res = replicate_parallel(a.runs, a.T, **kw)
        m, h = mean_ci([r["q1_mean"] - r["q2_mean"] for r in res])
        th = queue_difference(a.T, 5, unif(5, 1), kw["batch2"], kw["interval2"])
        rows.append({"case": name, "T": a.T, "runs": a.runs, "delta": m, "delta_ci": h, "theory": th})
        print(f"{name:50}  Q1-Q2 = {m:7.3f} ± {h:6.3f}   теорія {th:7.3f}")

    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "queue_offset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
