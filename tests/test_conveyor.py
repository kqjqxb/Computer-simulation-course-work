import math

import pytest

from src.model.conveyor import (ARRIVAL_1, ARRIVAL_2, FILL, SKIP,
                                build_conveyor_net, summarize)
from src.petri import EventProtocol, Simulator, const


def deterministic_net(k=10, period=10):
    return build_conveyor_net(k=k, period=period,
                              interval1=const(5), interval2=const(20))


def test_deterministic_scenario_matches_hand_calculation():
    # type 1: +5 at 5, 10, 15, ...; type 2: +20 at 20, 40, ...; sections at 10, 20, ...
    # t=10: Q1=10, Q2=0  -> skip
    # t=20: Q1=20, Q2=20 -> fill -> 10, 10
    # t=30: Q1=20, Q2=10 -> fill -> 10, 0
    # t=40: Q1=20, Q2=20 -> fill -> 10, 10   (then the pattern repeats)
    net = deterministic_net()
    proto = EventProtocol(places=["Q1", "Q2"])
    Simulator(net, protocol=proto).run(t_end=60)
    sections = [(t, tr, m["Q1"], m["Q2"]) for t, ph, tr, m in proto.rows
                if ph == "IN" and tr in (FILL, SKIP)]
    assert sections == [
        (10, SKIP, 10, 0),
        (20, FILL, 10, 10),
        (30, FILL, 10, 0),
        (40, FILL, 10, 10),
        (50, FILL, 10, 0),
        (60, FILL, 10, 10),
    ]


def test_deterministic_scenario_queue_statistics():
    net = deterministic_net()
    sim = Simulator(net).run(t_end=1000, warmup=100)
    res = summarize(sim, k=10)
    assert res["p_skip"] == 0
    # after t = 20, Q1 cycles 10 -> 15 -> 10 every 10 min (+5 at t+5)
    assert res["q1_mean"] == pytest.approx(12.5)
    assert res["q1_max"] == 15
    # Q2 is 10 on [20, 30), 0 on [30, 40), +20 and a fill at 40, and so on
    assert res["q2_mean"] == pytest.approx(5.0)
    assert res["q2_max"] == 10
    assert res["sections"] == 90  # sections at 110, 120, ..., 1000
    assert res["items_per_min"] == pytest.approx(1.0)


@pytest.mark.parametrize("k,period", [(10, 10), (20, 20)])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_item_conservation(k, period, seed):
    net = build_conveyor_net(k=k, period=period)
    sim = Simulator(net, seed=seed).run(t_end=5000)
    arrived1 = 5 * sim.fired_out[ARRIVAL_1]
    arrived2 = 20 * sim.fired_out[ARRIVAL_2]
    n_fill = sim.fired_out[FILL]
    assert arrived1 == net.places["Q1"].markers + k * n_fill
    assert arrived2 == net.places["Q2"].markers + k * n_fill


@pytest.mark.parametrize("k,period,t_end", [(10, 10, 5000), (20, 20, 5000), (10, 10, 1234.5)])
def test_every_section_is_filled_or_skipped(k, period, t_end):
    net = build_conveyor_net(k=k, period=period)
    sim = Simulator(net, seed=11).run(t_end=t_end)
    assert sim.fired_out[FILL] + sim.fired_out[SKIP] == math.floor(t_end / period)
    assert net.places["Section"].markers == 0


def test_section_skipped_only_when_items_are_missing():
    net = build_conveyor_net()
    proto = EventProtocol(places=["Q1", "Q2"])
    Simulator(net, seed=5, protocol=proto).run(t_end=20_000)
    skips = [m for _, ph, tr, m in proto.rows if ph == "IN" and tr == SKIP]
    assert skips, "expected at least one skipped section"
    for m in skips:
        assert m["Q1"] < 10 or m["Q2"] < 10


def test_first_arrivals_after_first_interval():
    net = build_conveyor_net()
    proto = EventProtocol(places=["Q1", "Q2"])
    Simulator(net, seed=3, protocol=proto).run(t_end=30)
    first1 = next(t for t, ph, tr, _ in proto.rows if ph == "OUT" and tr == ARRIVAL_1)
    first2 = next(t for t, ph, tr, _ in proto.rows if ph == "OUT" and tr == ARRIVAL_2)
    assert 4 <= first1 <= 6
    assert 13 <= first2 <= 27
