import math

import numpy as np
from scipy import stats

from src.model.conveyor import build_conveyor_net, summarize
from src.petri import Simulator


def run_once(seed, T, warmup=0.0, protocol=None, **model_kwargs):
    net = build_conveyor_net(**model_kwargs)
    sim = Simulator(net, seed=seed, protocol=protocol).run(T, warmup=warmup)
    return summarize(sim)


def replicate(runs, T, seed=1, warmup=0.0, **model_kwargs):
    """Independent runs with seeds seed, seed+1, ...; list of summaries."""
    return [run_once(seed + i, T, warmup, **model_kwargs) for i in range(runs)]


def mean_ci(values, beta=0.95):
    """Sample mean and half-width of the Student confidence interval."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, math.nan
    half = stats.t.ppf((1 + beta) / 2, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return m, float(half)
