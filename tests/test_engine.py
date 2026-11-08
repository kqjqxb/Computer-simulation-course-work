import math

import pytest

from src.petri import EventProtocol, PetriNet, Simulator, const, unif


def test_transition_enabled_needs_arc_weight():
    net = PetriNet()
    net.add_place("P", 2)
    net.add_place("R", 0)
    tr = net.add_transition("T", const(1), {"P": 3}, {"R": 1})
    assert not tr.is_enabled()
    net.places["P"].markers = 3
    assert tr.is_enabled()


def test_multichannel_transition_fires_all_markers_at_once():
    net = PetriNet()
    net.add_place("P", 3)
    net.add_place("R", 0)
    net.add_transition("T", const(2), {"P": 1}, {"R": 1})
    sim = Simulator(net, seed=1).run(t_end=1)
    assert net.transitions["T"].busy == 3
    sim.run(t_end=3)
    assert net.places["R"].markers == 3
    assert sim.fired_out["T"] == 3


def test_priority_resolves_conflict():
    net = PetriNet()
    net.add_place("P", 1)
    net.add_place("A", 0)
    net.add_place("B", 0)
    net.add_transition("high", const(0), {"P": 1}, {"A": 1}, priority=1)
    net.add_transition("low", const(0), {"P": 1}, {"B": 1}, priority=0)
    for seed in range(20):
        Simulator(net, seed=seed).run(t_end=1)
        assert net.places["A"].markers == 1
        assert net.places["B"].markers == 0


def test_probability_resolves_equal_priority_conflict():
    net = PetriNet()
    net.add_place("Src", 1)
    net.add_place("P", 0)
    net.add_place("A", 0)
    net.add_place("B", 0)
    net.add_transition("gen", const(1), {"Src": 1}, {"Src": 1, "P": 1})
    net.add_transition("a", const(0), {"P": 1}, {"A": 1}, probability=0.8)
    net.add_transition("b", const(0), {"P": 1}, {"B": 1}, probability=0.2)
    Simulator(net, seed=7).run(t_end=10_000.5)
    share_a = net.places["A"].markers / 10_000
    assert share_a == pytest.approx(0.8, abs=0.02)


def test_zero_delay_chain_completes_within_instant():
    net = PetriNet()
    net.add_place("Src", 1)
    net.add_place("P1", 0)
    net.add_place("P2", 0)
    net.add_place("P3", 0)
    net.add_transition("gen", const(5), {"Src": 1}, {"Src": 1, "P1": 1})
    net.add_transition("z1", const(0), {"P1": 1}, {"P2": 1})
    net.add_transition("z2", const(0), {"P2": 1}, {"P3": 1})
    sim = Simulator(net, seed=0).run(t_end=12)
    # arrivals at 5 and 10 reach P3 at the same instants; P1, P2 never hold markers
    assert net.places["P3"].markers == 2
    assert net.places["P1"].stat.max == 0
    assert net.places["P2"].stat.max == 0
    assert net.places["P3"].stat.mean == pytest.approx((0 * 5 + 1 * 5 + 2 * 2) / 12)


def test_time_weighted_mean_std_and_warmup():
    net = PetriNet()
    net.add_place("Src", 1)
    net.add_place("P", 0)
    net.add_transition("gen", const(2), {"Src": 1}, {"Src": 1, "P": 1})
    sim = Simulator(net, seed=0).run(t_end=10)
    # P = 0 on [0,2), 1 on [2,4), ..., 4 on [8,10)
    st = net.places["P"].stat
    assert st.mean == pytest.approx(2.0)
    assert st.std == pytest.approx(math.sqrt(2.0))
    assert st.max == 4 and st.min == 0
    sim.run(t_end=10, warmup=6)
    assert net.places["P"].stat.mean == pytest.approx(3.5)
    assert net.places["P"].stat.min == 3
    assert sim.window_out["gen"] == 2  # exits at 8 and 10


def test_same_seed_same_result():
    def run(seed):
        net = PetriNet()
        net.add_place("Src", 1)
        net.add_place("P", 0)
        net.add_transition("gen", unif(5, 2), {"Src": 1}, {"Src": 1, "P": 1})
        Simulator(net, seed=seed).run(t_end=1000)
        return net.places["P"].markers, net.places["P"].stat.mean

    assert run(42) == run(42)
    assert run(42) != run(43)


def test_net_that_never_settles_raises():
    net = PetriNet()
    net.add_place("P", 1)
    net.add_transition("loop", const(0), {"P": 1}, {"P": 1})
    with pytest.raises(RuntimeError):
        Simulator(net).run(t_end=1)


def test_protocol_records_in_and_out():
    net = PetriNet()
    net.add_place("Src", 1)
    net.add_place("P", 0)
    net.add_transition("gen", const(3), {"Src": 1}, {"Src": 1, "P": 1})
    proto = EventProtocol(places=["P"])
    Simulator(net, protocol=proto).run(t_end=7)
    assert [(t, ph) for t, ph, _, _ in proto.rows] == [
        (0, "IN"), (3, "OUT"), (3, "IN"), (6, "OUT"), (6, "IN")]
    assert proto.rows[-1][3] == {"P": 2}


def test_samples_on_a_time_grid_do_not_change_results():
    def build():
        net = PetriNet()
        net.add_place("Src", 1)
        net.add_place("P", 0)
        net.add_transition("gen", unif(5, 2), {"Src": 1}, {"Src": 1, "P": 1})
        return net

    plain = Simulator(build(), seed=3).run(t_end=1000)
    means = []
    net = build()
    sampled = Simulator(net, seed=3).run(
        t_end=1000, sample_every=100,
        on_sample=lambda sim: means.append((sim.t, sim.net.places["P"].stat.mean)))
    assert len(means) == 10 and means[-1][0] == 1000
    assert means[-1][1] == pytest.approx(plain.net.places["P"].stat.mean)
    assert sampled.fired_out == plain.fired_out
