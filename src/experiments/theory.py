"""Analytical estimates used to check the simulation.

Fluid (mean-rate) approximation. Arrival rates
lambda_i = b_i / mu_i items/min, conveyor capacity
c = k / period items/min of each type. Filled sections remove
items of both types at the rate f = min(lambda_1, lambda_2, c), so
    P_skip  ~ 1 - f / c = 1 - lambda_min * period / k
              (if lambda_min < c),
    slope_i = lambda_i - f   (linear queue growth, items/min),
and over [0, T] a linearly growing queue has mean
~ slope * T / 2 and maximum ~ slope * T.

Renewal theory. For batches b arriving after intervals with mean
mu and variance s^2 (first batch after one interval)
    E[A(t)] -> b * t / mu + b * (s^2 - mu^2) / (2 mu^2).
Since Q1(t) - Q2(t) = A1(t) - A2(t), the time-mean difference of
the queues is
    Q1_mean - Q2_mean ~ (lambda_1 - lambda_2) * T / 2
                        + off_1 - off_2.
"""
import math

from src.petri.rng import Const, Unif


def interval_moments(interval):
    if isinstance(interval, Const):
        return interval.d, 0.0
    if isinstance(interval, Unif):
        return interval.d, (2 * interval.s) ** 2 / 12
    raise TypeError(f"no moments for {interval!r}")


def rate(batch, interval):
    mu, _ = interval_moments(interval)
    return batch / mu


def renewal_offset(batch, interval):
    mu, var = interval_moments(interval)
    return batch * (var - mu ** 2) / (2 * mu ** 2)


def fluid(k, period, lam1, lam2):
    c = k / period
    f = min(lam1, lam2, c)
    return {
        "capacity": c,
        "fill_rate": f,
        "p_skip": 1 - f / c,
        "slope1": lam1 - f,
        "slope2": lam2 - f,
        "rho": max(lam1, lam2) / c,
    }


def queue_difference(T, batch1, interval1, batch2, interval2):
    lam1, lam2 = rate(batch1, interval1), rate(batch2, interval2)
    return (lam1 - lam2) * T / 2 + renewal_offset(batch1,
            interval1) - renewal_offset(batch2, interval2)


def difference_variance_rate(batch1, interval1, batch2,
                             interval2):
    """D = sum b^2 s^2 / mu^3: Var[A1(t) - A2(t)] ~ D * t,
    items^2/min.
    """
    d = 0.0
    for b, itv in ((batch1, interval1), (batch2, interval2)):
        mu, var = interval_moments(itv)
        d += b ** 2 * var / mu ** 3
    return d


def excess_growth_coef(batch1, interval1, batch2, interval2):
    """Time-mean over [0, T] of E|A1(t) - A2(t)| ~ coef * sqrt(T).

    With equal arrival rates A1 - A2 is a zero-drift random walk;
    for a normal value with variance D t, E|X| = sqrt(2 D t / pi),
    and the mean of sqrt(t) over [0, T] is 2/3 sqrt(T). Every
    filled section removes the same number of items of both types,
    so this excess stays in one of the queues whatever the
    conveyor speed is: if the conveyor keeps min(Q1, Q2) bounded,
    Q1_mean + Q2_mean ~ coef * sqrt(T) + const.
    """
    d = difference_variance_rate(batch1, interval1, batch2,
                                 interval2)
    return 2 / 3 * math.sqrt(2 * d / math.pi)
