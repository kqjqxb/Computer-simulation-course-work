import pytest

from src.model.conveyor import ARRIVAL_1, ARRIVAL_2, FILL, build_conveyor_net, summarize
from src.petri import EventProtocol, Simulator


@pytest.mark.parametrize("buffer", [20, 30, (15, 40)])
def test_queue_never_exceeds_buffer(buffer):
    l1, l2 = (buffer, buffer) if isinstance(buffer, int) else buffer
    net = build_conveyor_net(buffer=buffer)
    proto = EventProtocol(places=["Q1", "Q2", "Free1", "Free2"])
    Simulator(net, seed=4, protocol=proto).run(t_end=5000)
    for _, _, _, m in proto.rows:
        assert m["Q1"] <= l1 and m["Q2"] <= l2
        assert m["Q1"] + m["Free1"] <= l1 and m["Q2"] + m["Free2"] <= l2


def test_places_are_conserved_in_buffer():
    net = build_conveyor_net(buffer=30)
    sim = Simulator(net, seed=9).run(t_end=3000)
    busy1 = net.transitions[ARRIVAL_1].busy  # batch in production holds its places
    busy2 = net.transitions[ARRIVAL_2].busy
    assert net.places["Q1"].markers + net.places["Free1"].markers + 5 * busy1 == 30
    assert net.places["Q2"].markers + net.places["Free2"].markers + 20 * busy2 == 30
    assert sim.fired_out[FILL] > 0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_huge_buffer_equals_base_model(seed):
    base = summarize(Simulator(build_conveyor_net(), seed=seed).run(t_end=5000))
    big = summarize(Simulator(build_conveyor_net(buffer=10**9), seed=seed).run(t_end=5000))
    assert big["block1"] == 0 and big["block2"] == 0
    for key in base:
        assert big[key] == pytest.approx(base[key], nan_ok=True), key


def test_small_buffer_blocks_suppliers_and_bounds_queues():
    res = summarize(Simulator(build_conveyor_net(buffer=20), seed=1).run(t_end=5000))
    assert res["q1_max"] <= 20 and res["q2_max"] <= 20
    assert res["block1"] > 0 and res["block2"] > 0
    assert res["p_skip"] > 0.05  # little stock -> more empty sections


def test_buffer_smaller_than_batch_is_rejected():
    with pytest.raises(ValueError):
        build_conveyor_net(buffer=15)  # type-2 batch is 20
