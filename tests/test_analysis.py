import numpy as np
import pytest
from scipy import stats

from src.experiments.anova import diff_ci, oneway_anova
from src.experiments.buffer_series import correlation_index, least_squares, normalize
from src.experiments.compare import paired_ci


def test_oneway_anova_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 50), rng.normal(0.5, 1, 50)
    res = oneway_anova([a, b])
    assert res["F"] == pytest.approx(stats.f_oneway(a, b).statistic)
    assert res["df_f"] == 1 and res["df_r"] == 98
    assert res["significant"] == (res["F"] > res["F_crit"])


def test_anova_identical_groups_not_significant():
    a = np.arange(10.0)
    res = oneway_anova([a, a.copy()])
    assert res["F"] == pytest.approx(0.0) and not res["significant"]


def test_normalize_maps_range_to_minus_one_one():
    x, x0, d = normalize([40, 50, 60, 70, 80])
    assert list(x) == [-1, -0.5, 0, 0.5, 1] and x0 == 60 and d == 20


def test_least_squares_recovers_quadratic():
    x = np.linspace(-1, 1, 7)
    y = 2 - 3 * x + 0.5 * x ** 2
    b, X = least_squares([x, x ** 2], y)
    assert b == pytest.approx([2, -3, 0.5])
    assert correlation_index(y, X @ b) == pytest.approx(1.0)


def test_difference_intervals():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    d, h = paired_ci(a, a + 1.0)
    assert d == pytest.approx(1.0) and h == pytest.approx(0.0)
    d, h = diff_ci(a, a + 1.0)
    assert d == pytest.approx(1.0) and h > 0
