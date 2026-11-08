"""Next-event simulation of a stochastic Petri net.

One step:
  1. t_next = min exit moment over all transitions;
  2. time-weighted statistics are collected on [t, t_next);
  3. t := t_next;
  4. out-phase: transitions whose exit moment is t release markers
     (transitions in random order);
  5. in-phase: while some transition is enabled, fire one of the
     enabled transitions with the highest priority (ties broken by
     the probability weights);
  6. zero-delay transitions fired in step 5 exit at the same t, so
     steps 4-5 repeat until no exit is left at t.
"""
from collections import Counter

from .rng import make_rng

MAX_FIRINGS_PER_INSTANT = 10_000


class Simulator:
    def __init__(self, net, seed=None, protocol=None, observer=None):
        self.net = net
        self.rng = make_rng(seed)
        self.protocol = protocol
        self.observer = observer  # called as observer(sim) after each instant
        self.t = 0.0
        self.warmup = 0.0
        self.fired_in = Counter()   # all firings
        self.fired_out = Counter()  # all exits
        self.window_out = Counter()  # exits in (warmup, t_end]

    # --- state change ---------------------------------------------------
    def _out_phase(self):
        ready = [tr for tr in self.net.transitions.values() if tr.next_exit <= self.t]
        self.rng.shuffle(ready)
        for tr in ready:
            n = tr.fire_out(self.t)
            self.fired_out[tr.name] += n
            if self.t > self.warmup or self.warmup == 0:
                self.window_out[tr.name] += n
            if self.protocol is not None:
                for _ in range(n):
                    self.protocol.record(self.t, "OUT", tr, self.net)
        return bool(ready)

    def _choose(self, enabled):
        top = max(tr.priority for tr in enabled)
        group = [tr for tr in enabled if tr.priority == top]
        if len(group) == 1:
            return group[0]
        weights = [tr.probability for tr in group]
        total = sum(weights)
        idx = self.rng.choice(len(group), p=[w / total for w in weights])
        return group[idx]

    def _in_phase(self):
        """Fire enabled transitions one by one; returns how many fired."""
        fired = 0
        while fired <= MAX_FIRINGS_PER_INSTANT:
            enabled = self.net.enabled()
            if not enabled:
                return fired
            tr = self._choose(enabled)
            tr.fire_in(self.t, self.rng)
            self.fired_in[tr.name] += 1
            if self.protocol is not None:
                self.protocol.record(self.t, "IN", tr, self.net)
            fired += 1
        return fired

    def _settle(self):
        """Process the current instant until nothing more happens at t."""
        fired = self._in_phase()
        while self.net.next_event_time() <= self.t:
            if fired > MAX_FIRINGS_PER_INSTANT:
                raise RuntimeError(f"net does not settle at t={self.t}")
            self._out_phase()
            fired += self._in_phase()

    # --- statistics -----------------------------------------------------
    def _collect(self, t_from, t_to):
        lo = max(t_from, self.warmup)
        dt = t_to - lo
        if dt <= 0:
            return
        for place in self.net.places.values():
            place.stat.add(place.markers, dt)

    # --- main loop ------------------------------------------------------
    def run(self, t_end, warmup=0.0, sample_every=None, on_sample=None):
        """Simulate on [0, t_end]; statistics are collected on [warmup, t_end].

        If sample_every is given, on_sample(sim) is called at t = sample_every,
        2 * sample_every, ... <= t_end with the statistics collected up to
        that moment (before the events of that moment are processed). Used
        for running means in the transient-period study.
        """
        if not 0 <= warmup < t_end:
            raise ValueError("need 0 <= warmup < t_end")
        self.warmup = float(warmup)
        self.net.reset()
        self.t = 0.0
        self.fired_in.clear()
        self.fired_out.clear()
        self.window_out.clear()
        next_sample = sample_every if sample_every else float("inf")

        self._settle()
        if self.observer:
            self.observer(self)
        while True:
            t_next = self.net.next_event_time()
            bound = min(t_next, t_end)
            while next_sample <= bound:
                self._collect(self.t, next_sample)
                self.t = next_sample
                on_sample(self)
                next_sample += sample_every
            if t_next > t_end:
                self._collect(self.t, t_end)
                self.t = float(t_end)
                break
            self._collect(self.t, t_next)
            self.t = t_next
            self._out_phase()
            self._settle()
            if self.observer:
                self.observer(self)
        return self

    # --- results --------------------------------------------------------
    def place_stats(self, name):
        st = self.net.places[name].stat
        return {"mean": st.mean, "std": st.std, "min": st.min, "max": st.max}
