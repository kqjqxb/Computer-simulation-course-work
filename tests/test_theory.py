import math

import pytest

from src.experiments.theory import (excess_growth_coef, fluid, queue_difference,
                                    rate, renewal_offset)
from src.petri import const, unif


def test_rates_of_base_model():
    assert rate(5, unif(5, 1)) == pytest.approx(1.0)
    assert rate(20, unif(20, 7)) == pytest.approx(1.0)


def test_renewal_offset_of_fixed_interval_is_minus_half_batch():
    # N(t) = floor(t / mu): mean over time is t / mu - 1/2
    assert renewal_offset(20, const(20)) == pytest.approx(-10.0)


def test_queue_difference_base():
    assert queue_difference(1000, 5, unif(5, 1), 20, unif(20, 7)) == pytest.approx(7.125)


def test_fluid_regimes():
    under = fluid(10, 5, 1.0, 1.0)
    assert under["p_skip"] == pytest.approx(0.5) and under["slope1"] == 0
    over = fluid(10, 20, 1.0, 1.0)
    assert over["p_skip"] == 0 and over["slope1"] == pytest.approx(0.5)
    one_sided = fluid(10, 10, 1.0, 0.5)
    assert one_sided["p_skip"] == pytest.approx(0.5)
    assert one_sided["slope1"] == pytest.approx(0.5) and one_sided["slope2"] == 0


def test_excess_growth_coef_base():
    d = 25 * (1 / 3) / 125 + 400 * (196 / 12) / 8000
    assert excess_growth_coef(5, unif(5, 1), 20, unif(20, 7)) == pytest.approx(
        2 / 3 * math.sqrt(2 * d / math.pi))
