"""Completing conveyor (variant D2) as a stochastic Petri net.

Base scheme: k = 10 items of each type per section, a section every 10 min.
Variant B:   k = 20 items of each type per section, a section every 20 min.
"""
from src.petri import PetriNet, const, unif

ARRIVAL_1 = "Надходження 1"
ARRIVAL_2 = "Надходження 2"
CONVEYOR = "Рух конвеєра"
FILL = "Комплектація"
SKIP = "Пропуск секції"


def build_conveyor_net(k=10, period=10.0,
                       batch1=5, interval1=unif(5, 1),
                       batch2=20, interval2=unif(20, 7),
                       q1_0=0, q2_0=0):
    net = PetriNet("conveyor")
    net.params = {"k": k, "period": period, "batch1": batch1, "interval1": interval1,
                  "batch2": batch2, "interval2": interval2}
    net.add_place("Gen1", 1)
    net.add_place("Gen2", 1)
    net.add_place("Q1", q1_0)
    net.add_place("Q2", q2_0)
    net.add_place("Conv", 1)
    net.add_place("Section", 0)
    net.add_place("Filled", 0)
    net.add_place("Skipped", 0)

    # generators and the conveyor drive are single-channel: the marker in
    # Gen1 / Gen2 / Conv returns only when the transition exits
    net.add_transition(ARRIVAL_1, interval1, {"Gen1": 1}, {"Gen1": 1, "Q1": batch1})
    net.add_transition(ARRIVAL_2, interval2, {"Gen2": 1}, {"Gen2": 1, "Q2": batch2})
    net.add_transition(CONVEYOR, const(period), {"Conv": 1}, {"Conv": 1, "Section": 1})

    # FILL and SKIP compete for the section; FILL has the higher priority,
    # so a section is skipped only when FILL is not enabled
    net.add_transition(FILL, const(0), {"Section": 1, "Q1": k, "Q2": k},
                       {"Filled": 1}, priority=1)
    net.add_transition(SKIP, const(0), {"Section": 1}, {"Skipped": 1}, priority=0)
    return net


def summarize(sim):
    """Model outputs on the statistics window (warmup, T].

    p_skip           = N_skip / (N_fill + N_skip)
    qi_mean, qi_std  - time-weighted mean and std of the queue, items
    qi_max           - maximum queue held for a positive time, items
    sections_per_hour, items_per_hour - throughput (items of each type)
    lambda_i         - observed arrival rate of type i, items/min
    w_i              - mean waiting time of a type-i item by Little's law, min
    """
    params = sim.net.params
    n_fill = sim.window_out[FILL]
    n_skip = sim.window_out[SKIP]
    total = n_fill + n_skip
    span = sim.t - sim.warmup
    q1 = sim.place_stats("Q1")
    q2 = sim.place_stats("Q2")
    lam1 = params["batch1"] * sim.window_out[ARRIVAL_1] / span
    lam2 = params["batch2"] * sim.window_out[ARRIVAL_2] / span
    nan = float("nan")
    return {
        "T": sim.t,
        "warmup": sim.warmup,
        "n_fill": n_fill,
        "n_skip": n_skip,
        "p_skip": n_skip / total if total else nan,
        "q1_mean": q1["mean"], "q1_std": q1["std"], "q1_max": q1["max"],
        "q2_mean": q2["mean"], "q2_std": q2["std"], "q2_max": q2["max"],
        "sections_per_hour": 60.0 * n_fill / span,
        "items_per_hour": 60.0 * params["k"] * n_fill / span,
        "lambda1": lam1,
        "lambda2": lam2,
        "w1": q1["mean"] / lam1 if lam1 > 0 else nan,
        "w2": q2["mean"] / lam2 if lam2 > 0 else nan,
    }
