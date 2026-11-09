import math


class TimeWeightedStat:
    """Mean, std, min, max of a piecewise-constant value in time.

    mean = sum(y_s * dt_s) / sum(dt_s)
    var  = sum(y_s^2 * dt_s) / sum(dt_s) - mean^2
    min / max are taken over values held for a positive time.
    """

    def __init__(self):
        self.total_time = 0.0
        self.area = 0.0
        self.area_sq = 0.0
        self.max = None
        self.min = None

    def add(self, value, dt):
        if dt <= 0:
            return
        self.total_time += dt
        self.area += value * dt
        self.area_sq += value * value * dt
        self.max = (value if self.max is None
                    else max(self.max, value))
        self.min = (value if self.min is None
                    else min(self.min, value))

    @property
    def mean(self):
        if self.total_time > 0:
            return self.area / self.total_time
        return math.nan

    @property
    def std(self):
        if self.total_time <= 0:
            return math.nan
        var = self.area_sq / self.total_time - self.mean ** 2
        return math.sqrt(max(var, 0.0))
