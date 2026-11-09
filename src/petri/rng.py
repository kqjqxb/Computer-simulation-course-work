"""Random delays in the guide's notation.

const(d)    - deterministic value d
unif(d, s)  - uniform on (d - s, d + s)
exp(d)      - exponential with mean d
norm(d, s)  - normal with mean d and std s, cut at 0
"""
import numpy as np


def make_rng(seed=None):
    return np.random.default_rng(seed)


class Delay:
    def sample(self, rng):
        raise NotImplementedError

    def __call__(self, rng):
        return self.sample(rng)


class Const(Delay):
    def __init__(self, d):
        self.d = float(d)

    def sample(self, rng):
        return self.d

    def __repr__(self):
        return f"{self.d:g}"


class Unif(Delay):
    def __init__(self, d, s):
        if s < 0 or d - s < 0:
            raise ValueError("unif(d, s) needs 0 <= s <= d")
        self.d, self.s = float(d), float(s)

    def sample(self, rng):
        # unif(d, s) = d - s + 2s * r,  r ~ U(0, 1)
        return self.d - self.s + 2.0 * self.s * rng.random()

    def __repr__(self):
        return f"unif({self.d:g}, {self.s:g})"


class Exp(Delay):
    def __init__(self, d):
        self.d = float(d)

    def sample(self, rng):
        # exp(d) = -d * ln(r)
        return -self.d * np.log(1.0 - rng.random())

    def __repr__(self):
        return f"exp({self.d:g})"


class Norm(Delay):
    def __init__(self, d, s):
        self.d, self.s = float(d), float(s)

    def sample(self, rng):
        return max(0.0, rng.normal(self.d, self.s))

    def __repr__(self):
        return f"norm({self.d:g}, {self.s:g})"


def const(d):
    return Const(d)


def unif(d, s):
    return Unif(d, s)


def exp(d):
    return Exp(d)


def norm(d, s):
    return Norm(d, s)


def as_delay(value):
    """Accept a Delay, a callable rng -> float, or a number."""
    if isinstance(value, Delay) or callable(value):
        return value
    return Const(value)
