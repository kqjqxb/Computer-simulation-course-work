class EventProtocol:
    """Event log: time, phase (IN / OUT), transition, marking after the event."""

    def __init__(self, places=None, limit=None,
                 headers=("t", "phase", "transition"), phase_names=None):
        self.places = places  # place names to show; None = all
        self.limit = limit
        self.headers = headers
        self.phase_names = phase_names or {}
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
        h_t, h_phase, h_tr = self.headers
        phase = {ph: self.phase_names.get(ph, ph) for ph in ("IN", "OUT")}
        pw = max(len(h_phase), *(len(v) for v in phase.values()))
        width = max(len(h_tr), *(len(r[2]) for r in self.rows))
        head = f"{h_t:>8}  {h_phase:<{pw}}  {h_tr:<{width}}  " + "  ".join(f"{n:>7}" for n in names)
        lines = [head]
        for t, ph, tr, marking in self.rows:
            lines.append(f"{t:8.3f}  {phase[ph]:<{pw}}  {tr:<{width}}  "
                         + "  ".join(f"{marking[n]:>7}" for n in names))
        return "\n".join(lines)
