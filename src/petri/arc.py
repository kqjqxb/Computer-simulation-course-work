from dataclasses import dataclass

from .place import Place


@dataclass(frozen=True)
class Arc:
    """Arc between a place and a transition; weight = arc multiplicity k."""

    place: Place
    weight: int = 1

    def __post_init__(self):
        if self.weight < 1:
            raise ValueError("arc weight must be >= 1")
