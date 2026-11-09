from .arc import Arc
from .place import Place
from .transition import Transition


class PetriNet:
    def __init__(self, name=""):
        self.name = name
        self.places = {}
        self.transitions = {}
        self.params = {}

    def add_place(self, name, markers=0):
        if name in self.places:
            raise ValueError(f"place {name!r} already exists")
        place = Place(name, markers)
        self.places[name] = place
        return place

    def add_transition(self, name, delay=0.0, inputs=None,
                       outputs=None,
                       priority=0, probability=1.0):
        """inputs / outputs: {place_name: arc_weight}."""
        if name in self.transitions:
            raise ValueError(f"transition {name!r} exists")
        tr = Transition(name, delay, priority, probability)
        for pname, w in (inputs or {}).items():
            tr.inputs.append(Arc(self.places[pname], w))
        for pname, w in (outputs or {}).items():
            tr.outputs.append(Arc(self.places[pname], w))
        self.transitions[name] = tr
        return tr

    def marking(self):
        return {name: p.markers
                for name, p in self.places.items()}

    def enabled(self):
        return [tr for tr in self.transitions.values()
                if tr.is_enabled()]

    def next_event_time(self):
        return min((tr.next_exit
                for tr in self.transitions.values()),
                default=float("inf"))

    def reset(self):
        for p in self.places.values():
            p.reset()
        for tr in self.transitions.values():
            tr.reset()
