"""Number of runs by formula (3.5): p = 20 * (sigma / eps)^2, beta = 0.95.

python -m src.experiments.replications

sigma is the standard deviation of a response over pilot runs.
Terminating study (base model and variant 20/20): T_mod = 480 min (one shift),
2400 min (one week) as a sensitivity check.
Buffer model: T_mod = 40000 min, statistics after a 3000-min warm-up, at the
decision point (20, 60) and at the grid corner with the largest buffers (30, 80).
"""
import csv
import math
from pathlib import Path

import numpy as np

from src.experiments.common import replicate_parallel

RESULTS = Path("results")
BETA = 0.95

EPS_TERM = {"p_skip": 0.005, "q1_mean": 1.0, "q2_mean": 1.0, "q1_max": 2.0, "q2_max": 2.0}
EPS_BUF = {"p_skip": 0.002, "q1_mean": 1.0, "q2_mean": 1.0}
T_BUF, WARM_BUF = 40000, 3000

STUDIES = [
    ("10/10", 480, 0.0, 100, dict(), EPS_TERM),
    ("20/20", 480, 0.0, 100, dict(k=20, period=20.0), EPS_TERM),
    ("10/10", 2400, 0.0, 100, dict(), EPS_TERM),
    ("20/20", 2400, 0.0, 100, dict(k=20, period=20.0), EPS_TERM),
    ("буфер 20/60", T_BUF, WARM_BUF, 20, dict(buffer=(20, 60)), EPS_BUF),
    ("буфер 30/80", T_BUF, WARM_BUF, 20, dict(buffer=(30, 80)), EPS_BUF),
]


def runs_needed(sigma, eps, beta=BETA):
    """Formula (3.4): p = sigma^2 / (eps^2 (1 - beta)); (3.5) for beta = 0.95."""
    return math.ceil(sigma ** 2 / (eps ** 2 * (1 - beta)))


def main():
    rows = []
    for name, T, warm, n0, model, eps in STUDIES:
        pilot = replicate_parallel(n0, T, seed=10001, warmup=warm, **model)
        for key, e in eps.items():
            vals = np.array([r[key] for r in pilot], dtype=float)
            sigma = float(vals.std(ddof=1))
            rows.append({"model": name, "T": T, "warmup": warm, "pilot_runs": n0, "response": key,
                         "mean": float(vals.mean()), "sigma": sigma, "eps": e,
                         "p": max(runs_needed(sigma, e), 20)})
    with (RESULTS / "replications.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model']:12} T={r['T']:6} {r['response']:8} середнє {r['mean']:9.4f}  "
              f"σ {r['sigma']:8.4f}  ε {r['eps']:6g}  p = {r['p']}")


if __name__ == "__main__":
    main()
