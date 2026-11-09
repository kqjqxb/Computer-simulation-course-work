from bisect import insort

from .rng import as_delay


class Transition:
    """Timed transition with a list of exit moments, multichannel.

    priority: larger value fires first when transitions conflict.
    probability: relative weight among enabled transitions of
    equal priority.
    """

    def __init__(self, name, delay=0.0, priority=0,
                 probability=1.0):
        if probability <= 0:
            raise ValueError("probability must be positive")
        self.name = name
        self.delay = as_delay(delay)
        self.priority = int(priority)
        self.probability = float(probability)
        self.inputs = []
        self.outputs = []
        self.exits = []  # sorted exit moments of busy channels

    def is_enabled(self):
        """X(M, T): each input place holds at least the weight."""
        return all(arc.place.markers >= arc.weight
                   for arc in self.inputs)

    def fire_in(self, t, rng):
        """Take markers from input places and schedule an exit."""
        for arc in self.inputs:
            arc.place.markers -= arc.weight
        exit_time = t + float(self.delay(rng))
        insort(self.exits, exit_time)
        return exit_time

    def fire_out(self, t):
        """Release channels with exit moment t; their number."""
        n = 0
        while self.exits and self.exits[0] <= t:
            self.exits.pop(0)
            n += 1
        for arc in self.outputs:
            arc.place.markers += arc.weight * n
        return n

    @property
    def next_exit(self):
        return self.exits[0] if self.exits else float("inf")

    @property
    def busy(self):
        return len(self.exits)

    def reset(self):
        self.exits = []

    def __repr__(self):
        return (f"Transition({self.name!r}, "
                f"delay={self.delay!r}, r={self.priority}, "
                f"b={self.probability:g})")
