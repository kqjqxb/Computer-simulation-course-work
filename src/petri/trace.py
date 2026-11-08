class EventProtocol:
    """Event log: time, phase (IN / OUT), transition, marking after the event."""

    def __init__(self, places=None, limit=None):
        self.places = places  # place names to show; None = all
        self.limit = limit
        self.rows = []

    @property
    def full(self):
        return self.limit is not None and len(self.rows) >= self.limit

    def record(self, t, phase, transition, net):
        if self.full:
            return
        marking = net.marking()
        if self.places is not None:
            marking = {p: marking[p] for p in self.places}
        self.rows.append((t, phase, transition.name, marking))

    def to_text(self):
        if not self.rows:
            return ""
        names = list(self.rows[0][3])
        width = max(len(r[2]) for r in self.rows)
        head = f"{'t':>8}  {'phase':<5}  {'transition':<{width}}  " + "  ".join(f"{n:>7}" for n in names)
        lines = [head]
        for t, phase, tr, marking in self.rows:
            lines.append(f"{t:8.3f}  {phase:<5}  {tr:<{width}}  "
                         + "  ".join(f"{marking[n]:>7}" for n in names))
        return "\n".join(lines)
