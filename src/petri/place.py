from .stats import TimeWeightedStat


class Place:
    """Petri net place: a marker count plus time-weighted statistics."""

    def __init__(self, name, markers=0):
        if markers < 0:
            raise ValueError("marker count must be non-negative")
        self.name = name
        self.initial = int(markers)
        self.markers = int(markers)
        self.stat = TimeWeightedStat()

    def reset(self):
        self.markers = self.initial
        self.stat = TimeWeightedStat()

    def __repr__(self):
        return f"Place({self.name!r}, {self.markers})"
